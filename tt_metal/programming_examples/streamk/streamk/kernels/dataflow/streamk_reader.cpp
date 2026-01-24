#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    uint32_t arg_idx = 0;
    uint32_t src0_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t src1_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t Mt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t Kt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t Nt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
    uint32_t total_output_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t macs_per_tile = get_arg_val<uint32_t>(arg_idx++);
    uint32_t macs_per_core = get_arg_val<uint32_t>(arg_idx++);
    uint32_t total_macs = get_arg_val<uint32_t>(arg_idx++);
    uint32_t mac_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t mac_end = get_arg_val<uint32_t>(arg_idx++);
    uint32_t my_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t my_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t first_tile_starter_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t first_tile_starter_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t partials_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    uint32_t is_first_tile_starter = get_arg_val<uint32_t>(arg_idx++);
    uint32_t first_tile_contributor_idx = get_arg_val<uint32_t>(arg_idx++);
    uint32_t last_tile_other_contributors = get_arg_val<uint32_t>(arg_idx++);
    uint32_t max_contributors = get_arg_val<uint32_t>(arg_idx++);

    // Note: Reader doesn't use most of these args, but must consume them to match runtime args

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_partials = tt::CBIndex::c_2;

    // Declare address in which we stored the source matrices. We have set the exact same format between CBs and DRAM
    // buffers in the host code, so we can use the same address for both DRAM and CBs.
    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);

    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, src0_addr, in0_tile_bytes);

    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, src1_addr, in1_tile_bytes);

    // Stream-K: Iterate through MAC iterations assigned to this core.
    // Each MAC iteration corresponds to one K iteration of one output tile.
    uint32_t mac_iter = mac_start;
    while (mac_iter < mac_end) {
        // Determine which output tile this MAC iteration belongs to
        uint32_t tile_idx = mac_iter / macs_per_tile;
        uint32_t tile_iter_start = tile_idx * macs_per_tile;
        uint32_t tile_iter_end = tile_iter_start + macs_per_tile;

        // Determine the range of MAC iters for this tile that this core needs to process
        uint32_t local_iter = mac_iter - tile_iter_start;
        uint32_t local_iter_end =
            (mac_end < tile_iter_end) ? (mac_end - tile_iter_start) : (tile_iter_end - tile_iter_start);

        // Convert linear tile index to 2D output coordinates
        uint32_t out_row = tile_idx / Nt;  // Which row in output matrix
        uint32_t out_col = tile_idx % Nt;  // Which column in output matrix

        // Inner loop: fetch input tiles for each K iteration in the local range
        for (uint32_t k_idx = local_iter; k_idx < local_iter_end; k_idx++) {
            // Calculate which A and B tiles are needed for this (out_row, out_col, k_idx) computation
            // A is laid out as [Mt x Kt], B is laid out as [Kt x Nt]
            uint32_t tile_A = out_row * Kt + k_idx;  // A tile at (out_row, k_idx)
            uint32_t tile_B = k_idx * Nt + out_col;  // B tile at (k_idx, out_col)

            // Read A tile from DRAM into circular buffer
            {
                cb_reserve_back(cb_id_in0, 1);
                uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                noc_async_read_tile(tile_A, a, l1_write_addr_in0);
                noc_async_read_barrier();
                cb_push_back(cb_id_in0, 1);
                // DPRINT << "Sent tile A for tile_idx " << tile_idx << ", k_idx " << k_idx << ENDL();
            }

            // Read B tile from DRAM into circular buffer
            {
                cb_reserve_back(cb_id_in1, 1);
                uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                noc_async_read_tile(tile_B, b, l1_write_addr_in1);
                noc_async_read_barrier();
                cb_push_back(cb_id_in1, 1);
                // DPRINT << "Sent tile B for tile_idx " << tile_idx << ", k_idx " << k_idx << ENDL();
            }
        }

        mac_iter = tile_iter_end;
    }
    DPRINT << "Reader finished all work." << ENDL();
}
