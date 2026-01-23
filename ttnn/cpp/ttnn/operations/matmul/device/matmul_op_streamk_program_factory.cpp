// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_op_streamk_program_factory.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::matmul {

tt::tt_metal::operation::ProgramWithCallbacks matmul_streamk(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    Tensor& output_tensor,
    const MatmulStreamKProgramConfig& program_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    Program program{};

    // ========================================================================
    // EXTRACT DIMENSIONS
    // ========================================================================
    const auto& a_shape = input_tensor_a.padded_shape();
    const auto& b_shape = input_tensor_b.padded_shape();

    uint32_t M = a_shape[-2];
    uint32_t K = a_shape[-1];
    uint32_t N = b_shape[-1];

    // Convert to tile dimensions
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;

    TT_FATAL(M % TILE_HEIGHT == 0, "M must be multiple of tile height");
    TT_FATAL(K % TILE_WIDTH == 0, "K must be multiple of tile width");
    TT_FATAL(N % TILE_WIDTH == 0, "N must be multiple of tile width");

    // ========================================================================
    // SETUP DEVICE AND GRID
    // ========================================================================
    IDevice* device = input_tensor_a.device();
    auto grid_size = program_config.compute_with_storage_grid_size;
    uint32_t num_cores = grid_size.x * grid_size.y;

    CoreRangeSet all_cores = num_cores_to_corerangeset(num_cores, grid_size, true);

    log_debug(tt::LogOp, "StreamK GEMM: M={}, K={}, N={}", M, K, N);
    log_debug(tt::LogOp, "StreamK Grid: {}x{} ({} cores)", grid_size.x, grid_size.y, num_cores);

    // ========================================================================
    // STREAMK WORK DISTRIBUTION
    // ========================================================================
    uint32_t macs_per_tile = Kt;  // Each output tile requires Kt MAC iterations
    uint32_t total_output_tiles = Mt * Nt;
    uint32_t total_macs = total_output_tiles * macs_per_tile;
    uint32_t macs_per_core = (total_macs + num_cores - 1) / num_cores;

    log_debug(tt::LogOp, "StreamK: {} total MACs, {} per core", total_macs, macs_per_core);

    // ========================================================================
    // BUFFER SETUP
    // ========================================================================
    auto src0_buffer = input_tensor_a.buffer();
    auto src1_buffer = input_tensor_b.buffer();
    auto dst_buffer = output_tensor.buffer();

    TT_FATAL(src0_buffer != nullptr, "Input A buffer is null");
    TT_FATAL(src1_buffer != nullptr, "Input B buffer is null");
    TT_FATAL(dst_buffer != nullptr, "Output buffer is null");

    // ========================================================================
    // CIRCULAR BUFFERS
    // ========================================================================
    const auto cb_data_format = tt::DataFormat::Float16_b;
    constexpr uint32_t single_tile_size = 2 * 1024;  // bfloat16: 32*32*2
    uint32_t num_buffer_tiles = 2;                   // Double buffering

    // CB 0: Input A tiles
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(num_buffer_tiles * single_tile_size, {{CBIndex::c_0, cb_data_format}})
            .set_page_size(CBIndex::c_0, single_tile_size));

    // CB 1: Input B tiles
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(num_buffer_tiles * single_tile_size, {{CBIndex::c_1, cb_data_format}})
            .set_page_size(CBIndex::c_1, single_tile_size));

    // CB 16: Output tiles
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(num_buffer_tiles * single_tile_size, {{CBIndex::c_16, cb_data_format}})
            .set_page_size(CBIndex::c_16, single_tile_size));

    // CB 2: Partial accumulation tiles (for cross-core partial handling)
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(num_buffer_tiles * single_tile_size, {{CBIndex::c_2, cb_data_format}})
            .set_page_size(CBIndex::c_2, single_tile_size));

    // ========================================================================
    // SEMAPHORES
    // ========================================================================
    // Used for signaling when partial tiles are ready for accumulation
    auto partials_ready_sem = CreateSemaphore(program, all_cores, 0);

    // ========================================================================
    // EXTRACT MATH FIDELITY FROM COMPUTE CONFIG
    // ========================================================================
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    std::visit(
        [&](auto&& config) {
            using T = std::decay_t<decltype(config)>;
            if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
                math_fidelity = config.math_fidelity;
            } else if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
                math_fidelity = config.math_fidelity;
            }
        },
        compute_kernel_config);

    log_debug(tt::LogOp, "StreamK Math Fidelity: {}", static_cast<int>(math_fidelity));

    // ========================================================================
    // CREATE READER KERNEL
    // ========================================================================
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);

    auto reader_id = CreateKernel(
        program,
        "tt_metal/programming_examples/streamk/streamk/kernels/dataflow/streamk_reader.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    // ========================================================================
    // CREATE WRITER KERNEL
    // ========================================================================
    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    auto writer_id = CreateKernel(
        program,
        "tt_metal/programming_examples/streamk/streamk/kernels/dataflow/streamk_writer.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    // ========================================================================
    // CREATE COMPUTE KERNEL
    // ========================================================================
    auto compute_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/streamk/streamk/kernels/compute/streamk_mm.cpp",
        all_cores,
        ComputeConfig{.math_fidelity = math_fidelity, .compile_args = {}});

    // ========================================================================
    // SET RUNTIME ARGUMENTS PER CORE
    // ========================================================================
    uint32_t core_idx = 0;
    CoreCoord prev_core_physical = CoreCoord{0, 0};

    for (const auto& core_range : all_cores.ranges()) {
        for (const auto& core : core_range) {
            // Get physical coordinates for this core
            CoreCoord core_physical = device->worker_core_from_logical_core(core);
            uint32_t my_x = core_physical.x;
            uint32_t my_y = core_physical.y;

            // Peer core is the previous core (for partial accumulation)
            uint32_t peer_x = (core_idx == 0) ? core_physical.x : prev_core_physical.x;
            uint32_t peer_y = (core_idx == 0) ? core_physical.y : prev_core_physical.y;

            // Calculate MAC iteration range for this core
            uint32_t mac_start = core_idx * macs_per_core;
            uint32_t mac_end = std::min(mac_start + macs_per_core, total_macs);

            // Common args for all kernels
            std::vector<uint32_t> common_args = {
                Mt,
                Kt,
                Nt,
                num_cores,
                total_output_tiles,
                macs_per_tile,
                macs_per_core,
                total_macs,
                mac_start,
                mac_end,
                my_x,
                my_y,
                peer_x,
                peer_y,
                partials_ready_sem};

            // Reader runtime args
            std::vector<uint32_t> reader_args = {src0_buffer->address(), src1_buffer->address()};
            reader_args.insert(reader_args.end(), common_args.begin(), common_args.end());
            SetRuntimeArgs(program, reader_id, core, reader_args);

            // Compute runtime args (also needs buffer addresses at the start)
            std::vector<uint32_t> compute_args = {src0_buffer->address(), src1_buffer->address()};
            compute_args.insert(compute_args.end(), common_args.begin(), common_args.end());
            SetRuntimeArgs(program, compute_kernel_id, core, compute_args);

            // Writer runtime args
            std::vector<uint32_t> writer_args = {dst_buffer->address()};
            writer_args.insert(writer_args.end(), common_args.begin(), common_args.end());
            SetRuntimeArgs(program, writer_id, core, writer_args);

            prev_core_physical = core_physical;
            core_idx++;
        }
    }

    // ========================================================================
    // OVERRIDE RUNTIME ARGUMENTS CALLBACK
    // ========================================================================
    auto override_runtime_args_callback = [reader_id, compute_kernel_id, writer_id, all_cores](
                                              const void* operation,
                                              const Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto src0_buffer = input_tensors.at(0).buffer();
        auto src1_buffer = input_tensors.at(1).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        for (const auto& core_range : all_cores.ranges()) {
            for (const auto& core : core_range) {
                // Update reader args (buffer addresses)
                {
                    auto& runtime_args = GetRuntimeArgs(program, reader_id, core);
                    runtime_args[0] = src0_buffer->address();
                    runtime_args[1] = src1_buffer->address();
                }

                // Update compute args (buffer addresses)
                {
                    auto& runtime_args = GetRuntimeArgs(program, compute_kernel_id, core);
                    runtime_args[0] = src0_buffer->address();
                    runtime_args[1] = src1_buffer->address();
                }

                // Update writer args (buffer address)
                {
                    auto& runtime_args = GetRuntimeArgs(program, writer_id, core);
                    runtime_args[0] = dst_buffer->address();
                }
            }
        }
    };

    // ========================================================================
    // RETURN PROGRAM
    // ========================================================================
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::matmul
