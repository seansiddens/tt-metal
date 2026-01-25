// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <random>
#include <chrono>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <bmm_op.hpp>
#include <tt-metalium/device.hpp>
#include <fmt/base.h>
#include <fmt/core.h>

// Trace region size for accurate performance measurement (16MB)
constexpr size_t TRACE_REGION_SIZE = 16 << 20;

uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
// Reference implementation of matrix multiplication.
// Array A is of size MxK, Array B is of size KxN, and the output C is of size MxN.
// The implementation is bare bones and does not include optimizations such as tiling or vectorization.
// This is intended to be used as a golden reference for testing the Metalium implementation.
void golden_matmul(
    std::vector<bfloat16>& a,
    std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K) {
    std::uint32_t idx_c = 0;
    std::uint32_t idx_a = 0;
    std::uint32_t idx_b = 0;

    float c_f;
    float float_tmp;
    std::vector<bfloat16> c_bf(M * N, 0);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            idx_c = j + (i * N);
            idx_a = i * K;
            idx_b = j;
            c_f = 0;
            for (int k_m = 0; k_m < K; k_m++) {
                float_tmp = static_cast<float>(a[idx_a]) * static_cast<float>(b[idx_b]);
                c_f += float_tmp;
                idx_a += 1;
                idx_b += N;
            }
            output.at(idx_c) = bfloat16(c_f);
        }
    }
}

/**
 * @brief Multi-core matrix multiplication using SPMD (Single Program, Multiple Data) parallelization.
 *
 * Performs C = A * B matrix multiplication by distributing output tiles across multiple cores.
 * Each core runs the same program but works on different portions of the output matrix,
 * making this a simple and efficient parallelization scheme.
 *
 * The function uses three types of kernels running in parallel:
 * - Reader: Loads input matrix tiles from DRAM into circular buffers
 * - Compute: Performs tile-wise matrix multiplication (A_tile * B_tile = C_tile)
 * - Writer: Stores computed output tiles back to DRAM
 *
 * Work distribution is handled automatically - if output tiles don't divide evenly
 * across cores, some cores get one extra tile to balance the workload.
 *
 * @param a Input matrix A in row-major format (bfloat16 elements)
 * @param b Input matrix B in row-major format (bfloat16 elements)
 * @param output Output matrix C to store A*B result (bfloat16 elements)
 * @param M Number of rows in matrix A and output matrix C
 * @param N Number of columns in matrix B and output matrix C
 * @param K Number of columns in matrix A and rows in matrix B
 * @param mesh_device Target mesh device (1x1 or larger) for computation
 *
 * @note Matrix dimensions must be divisible by tile size (32x32) for this implementation
 * @note Uses circular buffers with 2 tiles for double-buffering to overlap compute and data movement
 */
void matmul_multi_core(
    std::vector<bfloat16>& a,
    std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t num_cores = 0) {
    // Check if the configuration is valid - matrices must be divisible by tile dimensions
    TT_ASSERT(
        (M * N) % TILE_HW == 0,
        "Matrix dimensions M={} and N={} must be divisible by TILE_HW={} to use this matmul implementation",
        M,
        N,
        TILE_HW);
    fmt::print("\nMatmul Multi-Core ===============================\n");

    // Set up mesh command queue, workload, device range, and program for multi-core execution
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program{};

    // Get the compute grid size to determine how many cores are available
    auto core_grid = mesh_device->compute_with_storage_grid_size();
    auto num_output_tiles_total = (M * N) / TILE_HW;
    fmt::print("Total output tiles: {}\n", num_output_tiles_total);

    // Determine which cores to use based on num_cores parameter
    // If num_cores is 0 or not specified, use all available cores
    // Otherwise, create a CoreRangeSet with exactly num_cores cores
    uint32_t max_cores = core_grid.x * core_grid.y;
    uint32_t target_num_cores = (num_cores == 0 || num_cores > max_cores) ? max_cores : num_cores;

    CoreRangeSet cores_to_use;
    uint32_t actual_num_cores;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t work_per_core1, work_per_core2;

    if (num_cores == 0 || num_cores >= max_cores) {
        // Use all cores via CoreCoord overload
        std::tie(actual_num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2) =
            split_work_to_cores(core_grid, num_output_tiles_total, true);
    } else {
        // Use specific number of cores via CoreRangeSet overload
        cores_to_use = num_cores_to_corerangeset(target_num_cores, core_grid, true);
        std::tie(actual_num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2) =
            split_work_to_cores(cores_to_use, num_output_tiles_total, true);
    }

    fmt::print(
        "Cores: {}, Work per core group 1: {}, Work per core group 2: {}\n",
        actual_num_cores,
        work_per_core1,
        work_per_core2);

    // Extracting Matrix dimensions from input/output vectors and converting to tile coordinates.
    // The accelerator works with 32x32 tiles, so we need to convert from element dimensions
    // to tile dimensions for proper addressing and computation.
    const uint32_t Mt = M / TILE_HEIGHT;  // Number of tiles in M dimension
    const uint32_t Kt = K / TILE_WIDTH;   // Number of tiles in K dimension
    const uint32_t Nt = N / TILE_WIDTH;   // Number of tiles in N dimension
    fmt::print(" Mt: {}, Nt: {}, Kt: {}\n", Mt, Nt, Kt);

    // Create DRAM buffers for input and output matrices (replicated per device across the mesh).
    // We allocate DRAM buffers for the input matrices and output matrix.
    // Setting page_size to single_tile_size is the most common configuration for memory buffers in Metalium
    // as it is generic, works for most cases and achieves good performance.
    // Writing data from input vectors to source buffers.
    constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;  // 2 * 32 * 32 = 2048 bytes

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};

    distributed::ReplicatedBufferConfig buffer_config_A{.size = single_tile_size * Mt * Kt};

    distributed::ReplicatedBufferConfig buffer_config_B{.size = single_tile_size * Nt * Kt};

    distributed::ReplicatedBufferConfig buffer_config_C{.size = single_tile_size * Mt * Nt};

    auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config_A, dram_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config_B, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config_C, dram_config, mesh_device.get());
    // Each handle is a mesh-wide replicated allocation; on a unit mesh this is a single device buffer

    // Configure Circular Buffers
    // Circular buffers act as staging areas for data movement between DRAM and compute units.
    // Using 2 tiles per circular buffer to allow for double buffering (data movement can be reading from one tile while
    // the compute kernel is using the other tile). This number can be adjusted based on the use case, but generally
    // diminishing returns are observed after several tiles.
    // input tiles count is = 2 so one tile can be read while the other is being processed
    const auto cb_data_format = tt::DataFormat::Float16_b;
    uint32_t num_input_tiles = 2;
    tt_metal::CreateCircularBuffer(
        program,
        all_cores,  // create on all cores
        CircularBufferConfig(num_input_tiles * single_tile_size, {{CBIndex::c_0, cb_data_format}})
            .set_page_size(CBIndex::c_0, single_tile_size));

    tt_metal::CreateCircularBuffer(
        program,
        all_cores,  // create on all cores
        CircularBufferConfig(num_input_tiles * single_tile_size, {{CBIndex::c_1, cb_data_format}})
            .set_page_size(CBIndex::c_1, single_tile_size));

    tt_metal::CreateCircularBuffer(
        program,
        all_cores,  // create on all cores
        CircularBufferConfig(num_input_tiles * single_tile_size, {{CBIndex::c_16, cb_data_format}})
            .set_page_size(CBIndex::c_16, single_tile_size));

    // Create Kernels (Reader, Writer, Compute)
    // - Reader kernel: Handles reading input data from DRAM into circular buffers
    // - Writer kernel: Handles writing output data from circular buffers back to DRAM
    // - Compute kernel: Performs the actual matrix multiplication computation
    // All kernels run across all cores to enable parallel execution
    // MathFidelity math_fidelity = MathFidelity::HiFi4;  // High fidelity math for accurate results
    MathFidelity math_fidelity = MathFidelity::LoFi;  // High fidelity math for accurate results
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_dram_buffer).append_to(reader_compile_time_args);
    auto reader_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "streamk/streamk/kernels/dataflow/reader_mm_output_tiles_partitioned.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
    auto writer_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "streamk/streamk/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "streamk/streamk/kernels/compute/mm.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = {}});

    // Set Runtime Arguments for Kernels
    // Each core needs to know which portion of the work it's responsible for. We are parallelizing across output
    // tiles - each core computes different output tiles. Runtime arguments can be changed between program executions
    // without recompilation.
    uint32_t work_offset = 0;
    auto work_groups = {std::make_pair(core_group_1, work_per_core1), std::make_pair(core_group_2, work_per_core2)};

    fmt::print("\nWork distribution across cores:\n");
    uint32_t core_idx = 0;

    // Iterate through each work group and assign work to cores
    for (const auto& [ranges, work_per_core] : work_groups) {
        for (const auto& range : ranges.ranges()) {
            for (const auto& core : range) {
                // Calculate MAC iterations for this core (output tiles * K-tile dimension)
                uint32_t mac_iters = work_per_core * Kt;
                uint32_t end_tile = work_offset + work_per_core - 1;

                // Print the work distribution for this core
                fmt::print(
                    "  Core {} ({}, {}): output tiles [{}, {}] (count: {}), MAC iters: {}\n",
                    core_idx,
                    core.x,
                    core.y,
                    work_offset,
                    end_tile,
                    work_per_core,
                    mac_iters);

                // Set arguments for the reader kernel (data input)
                tt_metal::SetRuntimeArgs(
                    program,
                    reader_id,
                    core,
                    {src0_dram_buffer->address(),  // Address of matrix A in DRAM
                     src1_dram_buffer->address(),  // Address of matrix B in DRAM
                     Mt,                           // Number of tiles in M dimension
                     Kt,                           // Number of tiles in K dimension
                     Nt,                           // Number of tiles in N dimension
                     work_offset,                  // Starting offset for this core's work
                     work_per_core});              // Amount of work for this core

                // Set arguments for the writer kernel (data output)
                tt_metal::SetRuntimeArgs(
                    program, writer_id, core, {dst_dram_buffer->address(), work_per_core, work_offset});

                // Set arguments for the compute kernel
                tt_metal::SetRuntimeArgs(
                    program,
                    compute_kernel_id,
                    core,
                    {work_per_core,            // Amount of work for this core
                     Kt});                     // Number of tiles in K dimension for dot product
                work_offset += work_per_core;  // Update offset for next core
                core_idx++;
            }
        }
    }
    fmt::print("\n");

    // Launch program & read in output buffer result into the host vector
    // 1. Upload input data to DRAM buffers
    // 2. Execute the program (all kernels run in parallel across cores)
    // 3. Read back the result from DRAM to host memory
    // The 'true' parameter in EnqueueReadMeshBuffer ensures we wait for completion (so when the function
    // returns, the output vector is fully populated).
    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, a, false);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, b, false);
    workload.add_program(device_range, std::move(program));

    // Warmup iterations (not recorded)
    constexpr uint32_t warmup_iterations = 10;
    fmt::print("Running {} warmup iterations...\n", warmup_iterations);
    for (uint32_t iter = 0; iter < warmup_iterations; ++iter) {
        distributed::EnqueueMeshWorkload(cq, workload, true);
    }

    // Capture trace for accurate timing (eliminates host dispatch overhead)
    constexpr uint32_t bench_iterations = 10;
    fmt::print("Capturing trace for {} iterations...\n", bench_iterations);

    auto trace_id = distributed::BeginTraceCapture(mesh_device.get(), cq.id());
    for (uint32_t iter = 0; iter < bench_iterations; ++iter) {
        distributed::EnqueueMeshWorkload(cq, workload, false);
    }
    mesh_device->end_mesh_trace(cq.id(), trace_id);

    // Time the trace replay (this excludes host dispatch overhead)
    fmt::print("Running traced benchmark...\n");
    auto start_time = std::chrono::high_resolution_clock::now();
    mesh_device->replay_mesh_trace(cq.id(), trace_id, false);
    distributed::Synchronize(mesh_device.get(), cq.id());
    auto end_time = std::chrono::high_resolution_clock::now();

    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    double avg_us = static_cast<double>(elapsed_us) / bench_iterations;
    fmt::print(
        "Average execution time: {:.2f} us ({:.3f} ms) [traced, {} iterations]\n\n",
        avg_us,
        avg_us / 1000.0,
        bench_iterations);

    // Release trace
    mesh_device->release_mesh_trace(trace_id);

    // Blocking read waits for completion before returning and resizes 'output' as needed
    distributed::EnqueueReadMeshBuffer(cq, output, dst_dram_buffer, true);

    fmt::print("\nMatmul Multi-Core Complete ======================\n");
}

void matmul_streamk(
    std::vector<bfloat16>& a,
    std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t num_cores = 0) {
    // Check if the configuration is valid - matrices must be divisible by tile dimensions
    TT_ASSERT(
        (M * N) % TILE_HW == 0,
        "Matrix dimensions M={} and N={} must be divisible by TILE_HW={} to use this matmul implementation",
        M,
        N,
        TILE_HW);
    fmt::print("\nMatmul StreamK ===============================\n");
    fmt::print("Matrix dimensions: M={}, N={}, K={}\n", M, N, K);
    fmt::print("BLK_M={}, BLK_N={}, BLK_K={}\n", TILE_HEIGHT, TILE_WIDTH, TILE_WIDTH);

    // Set up mesh command queue, workload, device range, and program for multi-core execution
    // distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program{};

    // Get the compute grid size to determine how many cores are available
    auto core_grid = mesh_device->compute_with_storage_grid_size();
    auto num_output_tiles_total = (M * N) / TILE_HW;
    fmt::print("Total output tiles: {}\n", num_output_tiles_total);

    // Determine which cores to use based on num_cores parameter
    // If num_cores is 0 or not specified, use all available cores
    // Otherwise, create a CoreRangeSet with exactly num_cores cores
    uint32_t max_cores = core_grid.x * core_grid.y;
    uint32_t target_num_cores = (num_cores == 0 || num_cores > max_cores) ? max_cores : num_cores;
    fmt::print("Num cores: {}\n", target_num_cores);

    CoreRangeSet all_cores = num_cores_to_corerangeset(target_num_cores, core_grid, true);

    // Extracting Matrix dimensions from input/output vectors and converting to tile coordinates.
    // The accelerator works with 32x32 tiles, so we need to convert from element dimensions
    // to tile dimensions for proper addressing and computation.
    const uint32_t Mt = M / TILE_HEIGHT;  // Number of tiles in M dimension
    const uint32_t Kt = K / TILE_WIDTH;   // Number of tiles in K dimension
    const uint32_t Nt = N / TILE_WIDTH;   // Number of tiles in N dimension
    fmt::print(" Mt: {}, Nt: {}, Kt: {}\n", Mt, Nt, Kt);

    // Compute StreamK work distribution parameters early (needed for CB sizing)
    uint32_t macs_per_tile = Kt;  // MAC iters (A_mk * B_kn) per output tile C_mn
    uint32_t total_macs = Mt * Nt * macs_per_tile;
    uint32_t macs_per_core = ceil_div(total_macs, target_num_cores);

    // Max contributors per tile: a tile can span at most ceil(macs_per_tile / macs_per_core) + 1 cores
    uint32_t max_contributors = std::min(target_num_cores, ceil_div(macs_per_tile, macs_per_core) + 1);
    // Ensure at least 1 for edge cases
    max_contributors = std::max(1u, max_contributors);
    fmt::print(
        "StreamK: macs_per_tile={}, total_macs={}, macs_per_core={}, max_contributors={}\n",
        macs_per_tile,
        total_macs,
        macs_per_core,
        max_contributors);

    // Create DRAM buffers for input and output matrices (replicated per device across the mesh).
    // We allocate DRAM buffers for the input matrices and output matrix.
    // Setting page_size to single_tile_size is the most common configuration for memory buffers in Metalium
    // as it is generic, works for most cases and achieves good performance.
    // Writing data from input vectors to source buffers.
    constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;  // 2 * 32 * 32 = 2048 bytes

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};

    distributed::ReplicatedBufferConfig buffer_config_A{.size = single_tile_size * Mt * Kt};

    distributed::ReplicatedBufferConfig buffer_config_B{.size = single_tile_size * Nt * Kt};

    distributed::ReplicatedBufferConfig buffer_config_C{.size = single_tile_size * Mt * Nt};

    auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config_A, dram_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config_B, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config_C, dram_config, mesh_device.get());
    // Each handle is a mesh-wide replicated allocation; on a unit mesh this is a single device buffer

    // Configure Circular Buffers
    // Circular buffers act as staging areas for data movement between DRAM and compute units.
    // Using 2 tiles per circular buffer to allow for double buffering (data movement can be reading from one tile while
    // the compute kernel is using the other tile). This number can be adjusted based on the use case, but generally
    // diminishing returns are observed after several tiles.
    // input tiles count is = 2 so one tile can be read while the other is being processed
    const auto cb_data_format = tt::DataFormat::Float16_b;
    uint32_t num_input_tiles = 2;
    tt_metal::CreateCircularBuffer(
        program,
        all_cores,  // create on all cores
        CircularBufferConfig(num_input_tiles * single_tile_size, {{CBIndex::c_0, cb_data_format}})
            .set_page_size(CBIndex::c_0, single_tile_size));

    tt_metal::CreateCircularBuffer(
        program,
        all_cores,  // create on all cores
        CircularBufferConfig(num_input_tiles * single_tile_size, {{CBIndex::c_1, cb_data_format}})
            .set_page_size(CBIndex::c_1, single_tile_size));

    tt_metal::CreateCircularBuffer(
        program,
        all_cores,  // create on all cores
        CircularBufferConfig(num_input_tiles * single_tile_size, {{CBIndex::c_16, cb_data_format}})
            .set_page_size(CBIndex::c_16, single_tile_size));

    // Partials CB: used to exchange partial tiles between cores
    // Sized to hold max_contributors tiles to support N-way tile splits
    uint32_t partials_cb_tiles = std::max(2u, max_contributors);
    tt_metal::CreateCircularBuffer(
        program,
        all_cores,  // create on all cores
        CircularBufferConfig(partials_cb_tiles * single_tile_size, {{CBIndex::c_2, cb_data_format}})
            .set_page_size(CBIndex::c_2, single_tile_size));

    // Semaphore: per-core synchronization for partials handling
    auto partials_ready_sem = tt_metal::CreateSemaphore(program, all_cores, 0);

    // Create Kernels (Reader, Writer, Compute)
    // - Reader kernel: Handles reading input data from DRAM into circular buffers
    // - Writer kernel: Handles writing output data from circular buffers back to DRAM
    // - Compute kernel: Performs the actual matrix multiplication computation
    // All kernels run across all cores to enable parallel execution
    MathFidelity math_fidelity = MathFidelity::LoFi;  // Match baseline for fair comparison
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_dram_buffer).append_to(reader_compile_time_args);
    auto reader_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "streamk/streamk/kernels/dataflow/streamk_reader.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
    auto writer_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "streamk/streamk/kernels/dataflow/streamk_writer.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "streamk/streamk/kernels/compute/streamk_mm.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = {}});

    // Set Runtime Arguments for Kernels
    // Each core needs to know which portion of the work it's responsible for. We are parallelizing across output
    // tiles - each core computes different output tiles. Runtime arguments can be changed between program executions
    // without recompilation.
    fmt::print("\nWork distribution across cores:\n");

    // Track which cores touch each tile and compute per-tile metadata for general N-way splits
    std::vector<uint32_t> tile_num_contributors(num_output_tiles_total, 0);
    std::vector<uint32_t> tile_starter_core_idx(num_output_tiles_total, UINT32_MAX);
    std::vector<CoreCoord> tile_starter_physical(num_output_tiles_total);

    // (macs_per_tile, total_macs, macs_per_core already computed above for CB sizing)
    fmt::print("MACs per tile: {}\n", macs_per_tile);
    fmt::print("Total MACs: {}\n", total_macs);
    fmt::print("MACs per core: {}\n", macs_per_core);

    // First pass: collect core physical coordinates and compute per-tile metadata
    std::vector<CoreCoord> core_physical_coords;
    std::vector<uint32_t> core_mac_starts;
    std::vector<uint32_t> core_mac_ends;

    uint32_t core_idx = 0;
    for (const auto& range : all_cores.ranges()) {
        for (const auto& core : range) {
            CoreCoord core_physical = mesh_device->worker_core_from_logical_core(core);
            core_physical_coords.push_back(core_physical);

            uint32_t mac_start = core_idx * macs_per_core;
            uint32_t mac_end = std::min(mac_start + macs_per_core, total_macs);
            core_mac_starts.push_back(mac_start);
            core_mac_ends.push_back(mac_end);

            // Determine which output tiles this core works on
            uint32_t first_tile = mac_start / macs_per_tile;
            uint32_t last_tile = (mac_end > 0) ? (mac_end - 1) / macs_per_tile : first_tile;

            // For each tile, track contributors and identify starter
            for (uint32_t tile_idx = first_tile; tile_idx <= last_tile; ++tile_idx) {
                uint32_t tile_mac_start = tile_idx * macs_per_tile;
                uint32_t tile_mac_end = tile_mac_start + macs_per_tile;

                uint32_t core_tile_mac_start = std::max(mac_start, tile_mac_start);
                uint32_t core_tile_mac_end = std::min(mac_end, tile_mac_end);

                if (core_tile_mac_end > core_tile_mac_start) {
                    tile_num_contributors[tile_idx]++;

                    // Check if this core is the starter (first to work on this tile)
                    bool tile_started = (core_tile_mac_start == tile_mac_start);
                    if (tile_started) {
                        tile_starter_core_idx[tile_idx] = core_idx;
                        tile_starter_physical[tile_idx] = core_physical;
                    }
                }
            }
            core_idx++;
        }
    }

    // Verify max_contributors bound computed earlier
    uint32_t actual_max_contributors = 1;
    for (uint32_t t = 0; t < num_output_tiles_total; ++t) {
        actual_max_contributors = std::max(actual_max_contributors, tile_num_contributors[t]);
    }
    fmt::print("Actual max contributors per tile: {} (bound: {})\n", actual_max_contributors, max_contributors);
    TT_ASSERT(actual_max_contributors <= max_contributors, "max_contributors bound exceeded");

    // Second pass: set runtime args for each core
    core_idx = 0;
    CoreCoord prev_core_logical = CoreCoord{0, 0};
    CoreCoord prev_core_physical = CoreCoord{0, 0};
    for (const auto& range : all_cores.ranges()) {
        for (const auto& core : range) {
            CoreCoord core_physical = core_physical_coords[core_idx];
            uint32_t my_x = core_physical.x;
            uint32_t my_y = core_physical.y;

            uint32_t mac_start = core_mac_starts[core_idx];
            uint32_t mac_end = core_mac_ends[core_idx];

            fmt::print(
                "Core {} logical ({}, {}), physical ({}, {})\n",
                core_idx,
                core.x,
                core.y,
                core_physical.x,
                core_physical.y);
            fmt::print("  MAC range: [{}, {})\n", mac_start, mac_end);

            // Determine which output tiles this core works on
            uint32_t first_tile = mac_start / macs_per_tile;
            uint32_t last_tile = (mac_end > 0) ? (mac_end - 1) / macs_per_tile : first_tile;

            fmt::print(
                "  Output tiles touched: {} to {} (count: {})\n", first_tile, last_tile, last_tile - first_tile + 1);

            // For each tile, print debug info
            for (uint32_t tile_idx = first_tile; tile_idx <= last_tile; ++tile_idx) {
                uint32_t tile_mac_start = tile_idx * macs_per_tile;
                uint32_t tile_mac_end = tile_mac_start + macs_per_tile;

                uint32_t core_tile_mac_start = std::max(mac_start, tile_mac_start);
                uint32_t core_tile_mac_end = std::min(mac_end, tile_mac_end);

                bool tile_started = (core_tile_mac_start == tile_mac_start);
                bool tile_finished = (core_tile_mac_end == tile_mac_end);

                uint32_t k_start = core_tile_mac_start - tile_mac_start;
                uint32_t k_end = core_tile_mac_end - tile_mac_start;

                fmt::print(
                    "    Tile {}: K-iters [{}, {}), started={}, finished={}, num_contributors={}\n",
                    tile_idx,
                    k_start,
                    k_end,
                    tile_started,
                    tile_finished,
                    tile_num_contributors[tile_idx]);
            }

            // Compute first_tile metadata (for non-starter case)
            uint32_t first_tile_starter_idx = tile_starter_core_idx[first_tile];
            CoreCoord first_tile_starter = tile_starter_physical[first_tile];
            bool is_first_tile_starter = (first_tile_starter_idx == core_idx);

            // Compute last_tile metadata (for starter-not-finisher case)
            uint32_t last_tile_mac_end = (last_tile + 1) * macs_per_tile;
            bool is_last_tile_finisher = (mac_end >= last_tile_mac_end);
            uint32_t last_tile_other_contributors = is_last_tile_finisher ? 0 : (tile_num_contributors[last_tile] - 1);

            // Compute contributor index for first tile (if not starter)
            uint32_t first_tile_contributor_idx = is_first_tile_starter ? 0 : (core_idx - first_tile_starter_idx);

            // Set arguments for the reader kernel (data input)
            tt_metal::SetRuntimeArgs(
                program,
                reader_id,
                core,
                {
                    src0_dram_buffer->address(),  // Address of matrix A in DRAM
                    src1_dram_buffer->address(),  // Address of matrix B in DRAM
                    Mt,                           // Number of tiles in M dimension
                    Kt,                           // Number of tiles in K dimension
                    Nt,                           // Number of tiles in N dimension
                    target_num_cores,             // Total number of cores
                    num_output_tiles_total,
                    macs_per_tile,                    // MAC iters per output tile
                    macs_per_core,                    // MAC iters per core
                    total_macs,                       // Total number of MAC iterations across whole problem
                    mac_start,                        // Starting MAC iteration for this core
                    mac_end,                          // Ending MAC iteration for this core
                    my_x,                             // This core's x
                    my_y,                             // This core's y
                    first_tile_starter.x,             // Starter core's x for first tile
                    first_tile_starter.y,             // Starter core's y for first tile
                    partials_ready_sem,               // partials_ready_sem id
                    is_first_tile_starter ? 1u : 0u,  // Is this core the starter for its first tile?
                    first_tile_contributor_idx,       // Contributor index for first tile (0 if starter)
                    last_tile_other_contributors,     // Num other contributors for last tile (0 if finisher)
                    max_contributors                  // Max contributors per tile (for CB sizing)
                });

            tt_metal::SetRuntimeArgs(
                program,
                compute_kernel_id,
                core,
                {
                    src0_dram_buffer->address(),  // Address of matrix A in DRAM
                    src1_dram_buffer->address(),  // Address of matrix B in DRAM
                    Mt,                           // Number of tiles in M dimension
                    Kt,                           // Number of tiles in K dimension
                    Nt,                           // Number of tiles in N dimension
                    target_num_cores,             // Total number of cores
                    num_output_tiles_total,
                    macs_per_tile,                    // MAC iters per output tile
                    macs_per_core,                    // MAC iters per core
                    total_macs,                       // Total number of MAC iterations across whole problem
                    mac_start,                        // Starting MAC iteration for this core
                    mac_end,                          // Ending MAC iteration for this core
                    my_x,                             // This core's x
                    my_y,                             // This core's y
                    first_tile_starter.x,             // Starter core's x for first tile
                    first_tile_starter.y,             // Starter core's y for first tile
                    partials_ready_sem,               // partials_ready_sem id
                    is_first_tile_starter ? 1u : 0u,  // Is this core the starter for its first tile?
                    first_tile_contributor_idx,       // Contributor index for first tile (0 if starter)
                    last_tile_other_contributors,     // Num other contributors for last tile (0 if finisher)
                    max_contributors                  // Max contributors per tile (for CB sizing)
                });

            tt_metal::SetRuntimeArgs(
                program,
                writer_id,
                core,
                {
                    dst_dram_buffer->address(),  // Address of output matrix C in DRAM
                    Mt,                          // Number of tiles in M dimension
                    Kt,                          // Number of tiles in K dimension
                    Nt,                          // Number of tiles in N dimension
                    target_num_cores,            // Total number of cores
                    num_output_tiles_total,
                    macs_per_tile,                    // MAC iters per output tile
                    macs_per_core,                    // MAC iters per core
                    total_macs,                       // Total number of MAC iterations across whole problem
                    mac_start,                        // Starting MAC iteration for this core
                    mac_end,                          // Ending MAC iteration for this core
                    my_x,                             // This core's x
                    my_y,                             // This core's y
                    first_tile_starter.x,             // Starter core's x for first tile
                    first_tile_starter.y,             // Starter core's y for first tile
                    partials_ready_sem,               // partials_ready_sem id
                    is_first_tile_starter ? 1u : 0u,  // Is this core the starter for its first tile?
                    first_tile_contributor_idx,       // Contributor index for first tile (0 if starter)
                    last_tile_other_contributors,     // Num other contributors for last tile (0 if finisher)
                    max_contributors                  // Max contributors per tile (for CB sizing)
                });

            // Update previous core for next iteration (both logical and physical)
            prev_core_logical = core;
            prev_core_physical = core_physical;
            core_idx++;
        }
    }

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, a, false);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, b, false);
    workload.add_program(device_range, std::move(program));

    // Warmup iterations (not recorded)
    constexpr uint32_t warmup_iterations = 10;
    fmt::print("Running {} warmup iterations...\n", warmup_iterations);
    for (uint32_t iter = 0; iter < warmup_iterations; ++iter) {
        distributed::EnqueueMeshWorkload(cq, workload, true);
    }

    // Capture trace for accurate timing (eliminates host dispatch overhead)
    constexpr uint32_t bench_iterations = 10;
    fmt::print("Capturing trace for {} iterations...\n", bench_iterations);

    auto trace_id = distributed::BeginTraceCapture(mesh_device.get(), cq.id());
    for (uint32_t iter = 0; iter < bench_iterations; ++iter) {
        distributed::EnqueueMeshWorkload(cq, workload, false);
    }
    mesh_device->end_mesh_trace(cq.id(), trace_id);

    // Time the trace replay (this excludes host dispatch overhead)
    fmt::print("Running traced benchmark...\n");
    auto start_time = std::chrono::high_resolution_clock::now();
    mesh_device->replay_mesh_trace(cq.id(), trace_id, false);
    distributed::Synchronize(mesh_device.get(), cq.id());
    auto end_time = std::chrono::high_resolution_clock::now();

    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    double avg_us = static_cast<double>(elapsed_us) / bench_iterations;
    fmt::print(
        "Average execution time: {:.2f} us ({:.3f} ms) [traced, {} iterations]\n",
        avg_us,
        avg_us / 1000.0,
        bench_iterations);

    // Calculate and print TFLOPs
    double flops = 2.0 * M * N * K;  // multiply-add
    double time_s = avg_us / 1e6;
    double tflops = flops / (time_s * 1e12);
    fmt::print("Average throughput:     {:.3f} TFLOPs\n\n", tflops);

    // Release trace
    mesh_device->release_mesh_trace(trace_id);

    // Read back the result from DRAM to host memory
    distributed::EnqueueReadMeshBuffer(cq, output, dst_dram_buffer, true);

    fmt::print("Execution complete!\n");

    fmt::print("\nMatmul StreamK Complete ======================\n");
}

///////////////////////////////////////

int main(int argc, char** argv) {
    bool pass = true;

    try {
        constexpr int device_id = 0;
        // Create mesh device with trace region enabled for accurate performance measurement
        // The trace region allows capturing kernel execution without host dispatch overhead
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(
            device_id,
            DEFAULT_L1_SMALL_SIZE,  // l1_small_size
            TRACE_REGION_SIZE       // trace_region_size (16MB for tracing)
        );

        // Parse command-line arguments for matrix dimensions (defaults: M=640, N=640, K=640)
        // Optional named arguments:
        //   --num-cores <value> (0 = use all cores)
        uint32_t M = 640;        // Number of rows in matrix A (user-defined)
        uint32_t N = 640;        // Number of columns in matrix B (user-defined)
        uint32_t K = 640;        // Inner dimension for multiplication (user-defined)
        uint32_t num_cores = 0;  // Number of cores to use (0 = use all available)

        // Parse positional arguments for M, N, K
        if (argc >= 4) {
            M = std::stoul(argv[1]);
            N = std::stoul(argv[2]);
            K = std::stoul(argv[3]);
        }

        // Parse named arguments
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--num-cores" && i + 1 < argc) {
                num_cores = std::stoul(argv[i + 1]);
                ++i;  // Skip next argument
            }
        }

        fmt::print("M: {}, N: {}, K: {}\n", M, N, K);
        fmt::print("MT: {}x{},", TILE_HEIGHT, TILE_WIDTH);

        // Ensure that the matrix dimensions are compatible with the tile size
        TT_FATAL(M % TILE_HEIGHT == 0, "M must be divisible by TILE_HEIGHT");
        TT_FATAL(N % TILE_WIDTH == 0, "N must be divisible by TILE_WIDTH");
        TT_FATAL(K % TILE_WIDTH == 0, "K must be divisible by TILE_WIDTH");

        // Calculate matrix dimensions in tiles for the accelerator
        uint32_t Mt = M / TILE_HEIGHT;
        uint32_t Nt = N / TILE_WIDTH;

        // Calculate buffer sizes needed for each matrix in bytes
        constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;  // 2 * 32 * 32 = 2048 bytes
        uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt;                           // num_tiles of FP16_B

        // Create random input vectors for matrices A and B
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        std::vector<bfloat16> src0_vec(M * K, 0);  // Matrix A (MxK)
        std::vector<bfloat16> src1_vec(K * N, 0);  // Matrix B (KxN)
        // // Fill with random bfloat16 values
        for (bfloat16& v : src0_vec) {
            v = bfloat16(dist(rng));
        }
        for (bfloat16& v : src1_vec) {
            v = bfloat16(dist(rng));
        }

        // Golden Matmul running on CPU (Float) - reference implementation for verification
        std::vector<bfloat16> golden_vec(M * N, 0);
        golden_matmul(src0_vec, src1_vec, golden_vec, M, N, K);

        // Input vector tilizing to match device expected tiled layout
        // The Tenstorrent hardware operates on data in 32x32 tiles rather than standard row-major format.
        // tilize_nfaces() converts the input matrices from row-major layout to the tiled layout expected by the device.
        // This transformation groups elements into 32x32 blocks and reorders them in memory so that each tile
        // (32x32 elements) is stored contiguously. This matches the native data access patterns of the matrix engine
        // and enables efficient operations on the accelerator.
        src0_vec = tilize_nfaces(src0_vec, M, K);
        src1_vec = tilize_nfaces(src1_vec, K, N);

        /* Run both StreamK and Baseline for comparison */
        std::vector<bfloat16> streamk_vec(dram_buffer_C_size / sizeof(bfloat16));
        std::vector<bfloat16> baseline_vec(dram_buffer_C_size / sizeof(bfloat16));

        fmt::print("\n=== Running StreamK Algorithm ===\n");
        matmul_streamk(src0_vec, src1_vec, streamk_vec, M, N, K, mesh_device, num_cores);
        streamk_vec = untilize_nfaces(streamk_vec, M, N);

        fmt::print("\n=== Running Baseline Multi-Core for Comparison ===\n");
        matmul_multi_core(src0_vec, src1_vec, baseline_vec, M, N, K, mesh_device, num_cores);
        baseline_vec = untilize_nfaces(baseline_vec, M, N);

        fmt::print("\nStreamK output vector size: {}\n", streamk_vec.size());
        fmt::print("Baseline output vector size: {}\n", baseline_vec.size());

        // Detailed error analysis
        fmt::print("\n=== Validation Results ===\n");

        // 1. StreamK vs Baseline
        float pcc_baseline = check_bfloat16_vector_pcc(baseline_vec, streamk_vec);
        fmt::print("StreamK vs Baseline -- PCC = {:.8f}\n", pcc_baseline);

        // 2. StreamK vs Golden (for reference)
        float pcc_golden = check_bfloat16_vector_pcc(golden_vec, streamk_vec);
        fmt::print("StreamK vs Golden   -- PCC = {:.8f}\n", pcc_golden);

        // 3. Baseline vs Golden (sanity check)
        float pcc_baseline_golden = check_bfloat16_vector_pcc(golden_vec, baseline_vec);
        fmt::print("Baseline vs Golden  -- PCC = {:.8f}\n", pcc_baseline_golden);

        // 4. Element-wise error statistics (StreamK vs Baseline)
        double max_abs_error = 0.0;
        double sum_abs_error = 0.0;
        uint32_t num_mismatches = 0;

        for (size_t i = 0; i < streamk_vec.size(); ++i) {
            float baseline_val = static_cast<float>(baseline_vec[i]);
            float streamk_val = static_cast<float>(streamk_vec[i]);
            float abs_error = std::abs(baseline_val - streamk_val);
            sum_abs_error += abs_error;
            max_abs_error = std::max(max_abs_error, static_cast<double>(abs_error));

            // Count exact mismatches (bfloat16 comparison)
            if (baseline_vec[i] != streamk_vec[i]) {
                num_mismatches++;
            }
        }

        double mean_abs_error = sum_abs_error / streamk_vec.size();
        fmt::print("\nError Statistics (StreamK vs Baseline):\n");
        fmt::print("  Max absolute error:  {:.6f}\n", max_abs_error);
        fmt::print("  Mean absolute error: {:.6f}\n", mean_abs_error);
        fmt::print(
            "  Exact mismatches:    {} / {} ({:.2f}%)\n",
            num_mismatches,
            streamk_vec.size(),
            100.0 * num_mismatches / streamk_vec.size());

        // 5. Per-tile error analysis (for small matrices)
        if (Mt * Nt <= 16) {
            fmt::print("\nPer-Tile Error Analysis (StreamK vs Baseline):\n");
            for (uint32_t tile_idx = 0; tile_idx < Mt * Nt; ++tile_idx) {
                uint32_t tile_m = tile_idx / Nt;
                uint32_t tile_n = tile_idx % Nt;

                double tile_max_error = 0.0;
                double tile_sum_error = 0.0;
                uint32_t tile_mismatches = 0;

                for (uint32_t row = 0; row < TILE_HEIGHT; ++row) {
                    for (uint32_t col = 0; col < TILE_WIDTH; ++col) {
                        uint32_t global_row = tile_m * TILE_HEIGHT + row;
                        uint32_t global_col = tile_n * TILE_WIDTH + col;
                        uint32_t idx = global_row * N + global_col;

                        float baseline_val = static_cast<float>(baseline_vec[idx]);
                        float streamk_val = static_cast<float>(streamk_vec[idx]);
                        float abs_error = std::abs(baseline_val - streamk_val);
                        tile_sum_error += abs_error;
                        tile_max_error = std::max(tile_max_error, static_cast<double>(abs_error));

                        if (baseline_vec[idx] != streamk_vec[idx]) {
                            tile_mismatches++;
                        }
                    }
                }

                double tile_mean_error = tile_sum_error / TILE_HW;
                fmt::print(
                    "  Tile {:2d} (M={}, N={}): max_err={:.6f}, mean_err={:.6f}, mismatches={}\n",
                    tile_idx,
                    tile_m,
                    tile_n,
                    tile_max_error,
                    tile_mean_error,
                    tile_mismatches);
            }
        }

        // Validation: StreamK should match baseline very closely
        constexpr float required_pcc = 0.99;  // StreamK should match baseline almost perfectly
        fmt::print("\nValidation: ");
        if (pcc_baseline >= required_pcc) {
            fmt::print("PASSED (PCC >= {})\n", required_pcc);
        } else {
            fmt::print("FAILED (PCC < {})\n", required_pcc);
            TT_FATAL(false, "StreamK vs Baseline PCC too low: {:.8f} < {}", pcc_baseline, required_pcc);
        }

        pass &= mesh_device->close();

    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());

        throw;
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
