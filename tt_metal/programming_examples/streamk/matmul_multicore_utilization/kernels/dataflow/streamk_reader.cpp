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

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;

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
    for (uint32_t mac_iter = mac_start; mac_iter < mac_end; mac_iter++) {
        // Determine which output tile this MAC iteration belongs to
        uint32_t tile_idx = mac_iter / macs_per_tile;

        // Determine which K iteration within that tile (MAC index within the tile)
        uint32_t k_idx = mac_iter % macs_per_tile;

        // Convert linear tile index to 2D output coordinates
        uint32_t out_row = tile_idx / Nt;  // Which row in output matrix
        uint32_t out_col = tile_idx % Nt;  // Which column in output matrix

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
        }

        // Read B tile from DRAM into circular buffer
        {
            cb_reserve_back(cb_id_in1, 1);
            uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
            noc_async_read_tile(tile_B, b, l1_write_addr_in1);
            noc_async_read_barrier();
            cb_push_back(cb_id_in1, 1);
        }
    }
}
