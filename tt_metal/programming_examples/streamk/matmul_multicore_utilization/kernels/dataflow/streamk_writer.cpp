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
    uint32_t peer_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t peer_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t partials_ready_sem = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t cb_id_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_id_partials = tt::CBIndex::c_2;

    // Set up the TensorAccessor for the output buffer
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);

    constexpr auto c_args = TensorAccessorArgs<0>();
    const auto c = TensorAccessor(c_args, dst_addr, tile_bytes);

    // Stream-K writer: Iterate through MAC space to identify which output tiles this core completes.
    uint32_t iter = mac_start;
    uint32_t iter_end = mac_end;

    while (iter < iter_end) {
        // Determine which output tile this iteration belongs to
        uint32_t tile_idx = iter / macs_per_tile;
        uint32_t tile_iter = tile_idx * macs_per_tile;
        uint32_t tile_iter_end = tile_iter + macs_per_tile;

        // Determine the range of MAC iterations for this tile that this core processes
        uint32_t local_iter = iter - tile_iter;
        uint32_t local_iter_end = (iter_end < tile_iter_end) ? (iter_end - tile_iter) : (tile_iter_end - tile_iter);

        // Check tile transition flags
        bool tile_started = (iter == tile_iter);
        bool tile_ended = (iter_end >= tile_iter_end);

        if (!tile_started) {
            // PARTIAL REMOTE-SEND PHASE
            // This core didn't start the tile, so it needs to send its locally-accumulated partials
            // to the core that started the tile.

            // Wait for partials from compute core.
            cb_wait_front(cb_id_out, onetile);

            // Send partials to peer core.
            uint32_t l1_read_addr = get_read_ptr(cb_id_out);
            uint32_t local_cb_partials_addr = get_write_ptr(cb_id_partials);
            uint64_t peer_noc_addr = get_noc_addr(peer_x, peer_y, local_cb_partials_addr);
            noc_async_write(l1_read_addr, peer_noc_addr, tile_bytes);
            noc_async_write_barrier();
            DPRINT << "Sent partials to peer (" << peer_x << ", " << peer_y << ") for tile " << tile_idx << ENDL();

            // Signal to peer that the partial is ready.
            uint32_t my_sem_addr = get_semaphore(partials_ready_sem);
            uint64_t peer_sem_noc = get_noc_addr(peer_x, peer_y, my_sem_addr);
            uint32_t val = 1;
            // noc_semaphore_set_remote(reinterpret_cast<uint32_t>(&val), peer_sem_noc);
            noc_semaphore_inc(peer_sem_noc, 1);
            DPRINT << "Signaled peer (" << peer_x << ", " << peer_y << ") for tile " << tile_idx
                   << "on semaphore addr: << " << peer_sem_noc << ENDL();
            noc_async_atomic_barrier();

            cb_pop_front(cb_id_out, onetile);
        } else {
            // This core started the tile
            if (!tile_ended) {
                // PARTIALs REMOTE-RECEIVE PHASE
                // This core started but didn't finish the tile
                // Wait for peer's semaphore signal.
                uint32_t sem_addr = get_semaphore(partials_ready_sem);
                volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
                uint64_t my_sem_addr = get_noc_addr(my_x, my_y, sem_addr);
                DPRINT << "Writer: Before noc_semaphore_wait for tile " << tile_idx << "on semaphore addr: << "
                       << my_sem_addr << ENDL();
                noc_semaphore_wait(sem_ptr, 1);
                DPRINT << "Writer: After noc_semaphore_wait for tile " << tile_idx << ENDL();
                noc_semaphore_set(sem_ptr, 0);

                // Partials have arrived from peer core into partials CB, now publish to compute.
                // DPRINT << "Writer: Before cb_reserve_back(cb_id_partials) for tile " << tile_idx << ENDL();
                // DPRINT << "Writer: After cb_reserve_back(cb_id_partials) for tile " << tile_idx << ENDL();
                // Push to compute to accumulate with local partials.
                cb_push_back(cb_id_partials, onetile);

                // Wait for accumulation to finish.
                cb_wait_front(cb_id_out, onetile);
                DPRINT << "Writer: After cb_wait_front(cb_id_out) for tile " << tile_idx << ENDL();

                // Push final tile to DRAM.
                uint32_t l1_read_addr = get_read_ptr(cb_id_out);
                noc_async_write_tile(tile_idx, c, l1_read_addr);
                noc_async_write_barrier();
                DPRINT << "Writer: Wrote final tile " << tile_idx << " to DRAM." << ENDL();

                cb_pop_front(cb_id_out, onetile);
            } else {
                // tile_started && tile_ended: this core both starts and finishes the tile (simple case)
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
