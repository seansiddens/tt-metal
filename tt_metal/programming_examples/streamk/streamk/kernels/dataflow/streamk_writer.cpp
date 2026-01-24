#include "dataflow_api.h"

void kernel_main() {
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
    uint32_t my_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t my_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t first_tile_starter_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t first_tile_starter_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t partials_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    uint32_t is_first_tile_starter = get_arg_val<uint32_t>(arg_idx++);
    uint32_t first_tile_contributor_idx = get_arg_val<uint32_t>(arg_idx++);
    uint32_t last_tile_other_contributors = get_arg_val<uint32_t>(arg_idx++);
    uint32_t max_contributors = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t cb_id_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_id_partials = tt::CBIndex::c_2;

    // Set up the TensorAccessor for the output buffer
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);

    constexpr auto c_args = TensorAccessorArgs<0>();
    const auto c = TensorAccessor(c_args, dst_addr, tile_bytes);

    // Stream-K writer: Iterate through MAC space to identify which output tiles this core processes
    uint32_t iter = mac_start;
    uint32_t iter_end = mac_end;

    // Determine first and last tiles this core works on
    uint32_t first_tile = mac_start / macs_per_tile;
    uint32_t last_tile = (mac_end > 0) ? (mac_end - 1) / macs_per_tile : first_tile;

    while (iter < iter_end) {
        // Determine which output tile this iteration belongs to
        uint32_t tile_idx = iter / macs_per_tile;
        uint32_t tile_iter = tile_idx * macs_per_tile;
        uint32_t tile_iter_end = tile_iter + macs_per_tile;

        // Check tile transition flags
        bool tile_started = (iter == tile_iter);
        bool tile_ended = (iter_end >= tile_iter_end);

        if (!tile_started) {
            // PARTIAL REMOTE-SEND PHASE
            // This core didn't start the tile, so send partials to the starter core
            // The starter core will accumulate all partials

            // Wait for partial from compute kernel
            cb_wait_front(cb_id_out, onetile);

            // Send partial to starter core at indexed offset
            uint32_t l1_read_addr = get_read_ptr(cb_id_out);

            // Determine starter core coords (only first tile uses pre-computed coords)
            uint32_t starter_x, starter_y;
            uint32_t contrib_idx;
            if (tile_idx == first_tile) {
                starter_x = first_tile_starter_x;
                starter_y = first_tile_starter_y;
                contrib_idx = first_tile_contributor_idx;
            } else {
                // For tiles after first, this core is the starter (since we just finished
                // the previous tile, this tile must start at our mac_start boundary)
                // This case shouldn't happen in !tile_started branch
                // But if it does, something is wrong
                DPRINT << "ERROR: non-first tile " << tile_idx << " but not starter?" << ENDL();
                starter_x = my_x;
                starter_y = my_y;
                contrib_idx = 0;
            }

            // Write to indexed slot in starter's cb_partials
            // contrib_idx is 1-based for non-starters (starter is index 0)
            uint32_t local_cb_partials_addr = get_write_ptr(cb_id_partials);
            uint32_t offset = (contrib_idx - 1) * tile_bytes;  // contrib_idx >= 1 for non-starters

            DPRINT << "SENDER tile " << tile_idx << ": sending to starter (" << starter_x << ", " << starter_y
                   << ") at offset " << offset << " (contrib_idx=" << contrib_idx << ")" << ENDL();

            uint64_t peer_noc_addr = get_noc_addr(starter_x, starter_y, local_cb_partials_addr + offset);
            noc_async_write(l1_read_addr, peer_noc_addr, tile_bytes);
            noc_async_write_barrier();

            // Signal to starter that this partial is ready
            uint32_t sem_addr = get_semaphore(partials_ready_sem);
            uint64_t peer_sem_noc = get_noc_addr(starter_x, starter_y, sem_addr);
            noc_semaphore_inc(peer_sem_noc, 1);
            noc_async_atomic_barrier();

            DPRINT << "SENDER tile " << tile_idx << ": signaled starter" << ENDL();

            cb_pop_front(cb_id_out, onetile);
        } else {
            // This core started the tile
            if (!tile_ended) {
                // PARTIALS REMOTE-RECEIVE PHASE
                // This core started but didn't finish the tile
                // Wait for all partials from other cores, then compute will accumulate

                // Determine how many partials to wait for
                uint32_t num_partials_to_receive;
                if (tile_idx == last_tile) {
                    num_partials_to_receive = last_tile_other_contributors;
                } else {
                    // For tiles before last, we should be both starter and finisher
                    // This case indicates a problem
                    DPRINT << "ERROR: non-last tile " << tile_idx << " but started and not ended?" << ENDL();
                    num_partials_to_receive = 0;
                }

                DPRINT << "REDUCER tile " << tile_idx << ": waiting for " << num_partials_to_receive << " partials"
                       << ENDL();

                // Wait for all partials to arrive
                uint32_t sem_addr = get_semaphore(partials_ready_sem);
                volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

                // Wait for semaphore to reach expected count
                noc_semaphore_wait(sem_ptr, num_partials_to_receive);
                noc_semaphore_set(sem_ptr, 0);  // Reset for future tiles

                DPRINT << "REDUCER tile " << tile_idx << ": all partials arrived" << ENDL();

                // Push all received partials to compute for accumulation
                // Compute will pop them as it accumulates
                cb_push_back(cb_id_partials, num_partials_to_receive);

                // Wait for compute to finish accumulation and output final tile
                cb_wait_front(cb_id_out, onetile);

                // Write final tile to DRAM
                uint32_t l1_read_addr = get_read_ptr(cb_id_out);
                noc_async_write_tile(tile_idx, c, l1_read_addr);
                noc_async_write_barrier();

                DPRINT << "REDUCER tile " << tile_idx << ": wrote to DRAM" << ENDL();

                cb_pop_front(cb_id_out, onetile);
            } else {
                // tile_started && tile_ended: this core both starts and finishes the tile
                cb_wait_front(cb_id_out, onetile);

                uint32_t l1_read_addr = get_read_ptr(cb_id_out);
                noc_async_write_tile(tile_idx, c, l1_read_addr);
                noc_async_write_barrier();

                cb_pop_front(cb_id_out, onetile);
            }
        }

        iter = tile_iter_end;
    }
    DPRINT << "Writer finished." << ENDL();
}
