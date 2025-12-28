#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"

using std::uint32_t;

namespace NAMESPACE {
/**
 * @brief Main kernel function for multi-core matrix multiplication (BMM).
 *
 * This function performs a blocked outer product matrix multiplication using tiles.
 * It initializes the matrix engine (FPU) and sets up circular buffers for input and output.
 * For each output tile (indexed by i), it:
 *   - Acquires the destination buffer.
 *   - Iterates over the K dimension (kt), waiting for input tiles to be available in the circular buffers.
 *   - Performs a tile-wise matrix multiplication using `matmul_tiles`.
 *   - Pops the used tiles from the input buffers.
 *   - After processing all K tiles, reserves space in the output buffer, packs the result tile, and pushes it to the
 *     output buffer.
 *   - Releases the destination buffer.
 *
 * Runtime arguments:
 *   - num_porduced_tiles: Number of output tiles to produce.
 *   - Kt: Number of tiles in the reduction dimension.
 *
 * Circular buffers:
 *   - cb_in0: Input buffer for matrix A tiles.
 *   - cb_in1: Input buffer for matrix B tiles.
 *   - cb_out: Output buffer for result tiles.
 *
 * Assumes that input tiles are provided in the correct order and that the reader is responsible for supplying
 * the appropriate tiles for each output tile computation.
 */
void MAIN {
    // uint32_t num_output_tiles = get_arg_val<uint32_t>(0);  // number of output tiles to produce
    // uint32_t Kt = get_arg_val<uint32_t>(1);                // number of tiles in K dimension for dot product

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
    uint32_t peer_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t peer_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t partials_ready_sem = get_arg_val<uint32_t>(arg_idx++);

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;
    constexpr tt::CBIndex cb_id_partials = tt::CBIndex::c_2;

    // Setup the FPU (matrix engine) for the matmul operation. And specify the input
    // and output circular buffers.
    mm_init(cb_in0, cb_in1, cb_out);

    // Stream-K compute kernel: Process MAC iterations assigned to this core.
    // Use while loop based on tile boundaries, matching the pseudocode structure.
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

        // MacLoop: Perform the range of MAC iterations for this tile
        // Initialize accumulator for this tile
        tile_regs_acquire();
        for (uint32_t k_idx = local_iter; k_idx < local_iter_end; k_idx++) {
            // Wait for the input tiles (A and B) from the reader
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);

            // Perform the tile-wise matrix multiplication (acumulates into registers)
            matmul_tiles(cb_in0, cb_in1, 0, 0, 0);

            // Pop the input tiles since we've consumed them
            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);
        }

        // Consolidate partial-sums across cores
        bool tile_started = (iter == tile_iter);
        bool tile_ended = (iter_end >= tile_iter_end);

        if (!tile_started) {
            // This core didn't start the tile, so send partials to the writer
            // Writer will forward these to the core that started this tile
            // DPRINT_PACK(DPRINT << "Compute: Before tile_regs_commit (sender) for tile " << tile_idx << ENDL());
            tile_regs_commit();
            tile_regs_wait();

            // DPRINT_PACK(DPRINT << "Compute: Before cb_reserve_back(cb_out) for tile " << tile_idx << ENDL());
            cb_reserve_back(cb_out, 1);
            // DPRINT_PACK(DPRINT << "Compute: After cb_reserve_back(cb_out) for tile " << tile_idx << ENDL());
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);

            tile_regs_release();
        } else {
            // This core started the tile
            if (!tile_ended) {
                // ▷ accumulate partial sums from other cores contributing to this tile
                // This core started but didn't finish the tile, so wait for partials from later cores
                // TODO: cta_end ← tile_iter_end / macs_per_tile

                // DPRINT_PACK(DPRINT << "Compute: Before cb_wait_front(cb_id_partials) for tile " << tile_idx <<
                // ENDL());
                cb_wait_front(cb_id_partials, 1);
                // DPRINT_PACK(DPRINT << "Compute: After cb_wait_front(cb_id_partials) for tile " << tile_idx <<
                // ENDL());
                cb_pop_front(cb_id_partials, 1);

                // After accumulating all partials, store the final tile
                tile_regs_commit();
                tile_regs_wait();

                // DPRINT_PACK(DPRINT << "Compute: Before cb_reserve_back(cb_out) for tile " << tile_idx << ENDL());
                cb_reserve_back(cb_out, 1);
                // DPRINT_PACK(DPRINT << "Compute: After cb_reserve_back(cb_out) for tile " << tile_idx << ENDL());
                pack_tile(0, cb_out);
                cb_push_back(cb_out, 1);

                tile_regs_release();
            } else {
                // tile_started && tile_ended: this core both starts and finishes the tile (simple case)
                tile_regs_commit();
                tile_regs_wait();

                cb_reserve_back(cb_out, 1);
                pack_tile(0, cb_out);
                cb_push_back(cb_out, 1);

                tile_regs_release();
            }
        }

        iter = tile_iter_end;
    }
    DPRINT << "All Compute Finished" << ENDL();
}
}  // namespace NAMESPACE
