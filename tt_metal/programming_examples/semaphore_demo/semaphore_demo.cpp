// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/constants.hpp>
#include <fmt/core.h>

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main(int /*argc*/, char** /*argv*/) {
    // Encourage kernel DPRINT visibility.
    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        fmt::print("NOTE: Set TT_METAL_DPRINT_CORES='(0,0)-(1,0)' to see kernel logs.\n");
    }

    // Create a unit mesh (single device) and basic workload objects.
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());

    // Choose two cores on the compute-with-storage grid (logical coordinates).
    const auto grid = mesh_device->compute_with_storage_grid_size();
    CoreCoord waiter_core_logical{0, 0};
    CoreCoord signaler_core_logical{1, 0};
    if (grid.x < 2) {
        // Fallback to vertical pair if only one column.
        TT_FATAL(grid.y >= 2, "Need at least two cores in the grid");
        signaler_core_logical = CoreCoord{0, 1};
    }

    // Convert logical coordinates to physical coordinates for runtime args.
    CoreCoord waiter_core_physical = mesh_device->worker_core_from_logical_core(waiter_core_logical);
    CoreCoord signaler_core_physical = mesh_device->worker_core_from_logical_core(signaler_core_logical);

    Program program = CreateProgram();

    // Create a semaphore on both cores with initial value 0.
    std::vector<CoreRange> core_ranges = {
        CoreRange(waiter_core_logical, waiter_core_logical), CoreRange(signaler_core_logical, signaler_core_logical)};
    CoreRangeSet both_cores(std::move(core_ranges));
    uint32_t sem_id = tt_metal::CreateSemaphore(program, both_cores, 0);

    // Create dataflow kernels: one waits on a local semaphore, the other remotely increments it.
    KernelHandle waiter_k = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "semaphore_demo/kernels/dataflow/waiter.cpp",
        waiter_core_logical,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    KernelHandle signaler_k = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "semaphore_demo/kernels/dataflow/signaler.cpp",
        signaler_core_logical,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Runtime args: Use PHYSICAL coordinates for NoC addressing.
    // Waiter gets its physical coordinates and semaphore id to wait on.
    SetRuntimeArgs(program, waiter_k, waiter_core_logical, {waiter_core_physical.x, waiter_core_physical.y, sem_id});
    // Signaler gets peer's physical coordinates (waiter) and semaphore id to increment remotely.
    SetRuntimeArgs(
        program, signaler_k, signaler_core_logical, {waiter_core_physical.x, waiter_core_physical.y, sem_id});

    // Dispatch and finish.
    fmt::print("Dispatched work...\n");
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    fmt::print("Semaphore demo finished. If it hangs, remote inc failed.\n");
    mesh_device->close();
    return 0;
}
