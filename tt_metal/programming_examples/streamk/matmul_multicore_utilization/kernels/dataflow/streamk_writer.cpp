#include "dataflow_api.h"

void kernel_main() {
    // Runtime arguments to write data back into the output buffer.
    // uint32_t dst_addr = get_arg_val<uint32_t>(0);
    // uint32_t num_tiles = get_arg_val<uint32_t>(1);  // number of output tiles to write
    // uint32_t start_id = get_arg_val<uint32_t>(2);   // starting tile ID for output tiles

    uint32_t arg_idx = 0;
    uint32_t dst_addr = get_arg_val<uint32_t>(arg_idx++);
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

    constexpr uint32_t cb_id_out = tt::CBIndex::c_16;

    // Set up the TensorAccessor for the output buffer
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);

    constexpr auto c_args = TensorAccessorArgs<0>();
    const auto c = TensorAccessor(c_args, dst_addr, tile_bytes);

    // Stream-K writer: Iterate through MAC space to identify which output tiles this core completes.
    // In the simple case (no partial sums), each complete tile is written to DRAM.
    uint32_t prev_tile_idx = UINT32_MAX;

    for (uint32_t mac_iter = mac_start; mac_iter < mac_end; mac_iter++) {
        uint32_t tile_idx = mac_iter / macs_per_tile;
        uint32_t tile_iter_end = (tile_idx + 1) * macs_per_tile;

        // Check if this MAC iteration completes an output tile
        bool tile_ended = (mac_iter + 1 == mac_end || mac_iter + 1 == tile_iter_end);

        if (tile_ended) {
            // Wait for the compute kernel to produce this output tile
            cb_wait_front(cb_id_out, onetile);

            // Write the output tile to DRAM at the correct position
            uint32_t l1_read_addr = get_read_ptr(cb_id_out);
            noc_async_write_tile(tile_idx, c, l1_read_addr);
            noc_async_write_barrier();

            // Pop the tile from the circular buffer
            cb_pop_front(cb_id_out, onetile);
        }
    }
}
