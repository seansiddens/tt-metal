#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "debug/dprint.h"

using std::uint32_t;

namespace NAMESPACE {
/**
 * @brief StreamK compute kernel with N-way partial tile accumulation support.
 *
 * This kernel handles matrix multiplication with arbitrary tile splits across cores.
 * For each output tile, the kernel:
 *   - Computes its assigned portion of MAC iterations
 *   - If not the starter: outputs partial for writer to send to starter
 *   - If starter but not finisher: accumulates partials from all other contributors
 *   - If starter and finisher: outputs directly
 */
void MAIN {
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

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;
    constexpr tt::CBIndex cb_id_partials = tt::CBIndex::c_2;

    constexpr uint32_t dst_reg = 0;

    // Setup the FPU (matrix engine) for the matmul operation
    mm_init(cb_in0, cb_in1, cb_out);

    // Determine first and last tiles this core works on
    uint32_t first_tile = mac_start / macs_per_tile;
    uint32_t last_tile = (mac_end > 0) ? (mac_end - 1) / macs_per_tile : first_tile;

    // Stream-K compute kernel: Process MAC iterations assigned to this core
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
        tile_regs_acquire();
        for (uint32_t k_idx = local_iter; k_idx < local_iter_end; k_idx++) {
            // Wait for the input tiles (A and B) from the reader
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);

            // Perform the tile-wise matrix multiplication (accumulates into registers)
            matmul_tiles(cb_in0, cb_in1, 0, 0, dst_reg);

            // Pop the input tiles since we've consumed them
            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);
        }

        // Determine tile transition flags
        bool tile_started = (iter == tile_iter);
        bool tile_ended = (iter_end >= tile_iter_end);

        if (!tile_started) {
            // PARTIAL REMOTE-SEND PHASE
            // This core didn't start the tile, so send partials to the writer
            // Writer will forward these to the starter core
            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);

            tile_regs_release();
        } else {
            // This core started the tile
            if (!tile_ended) {
                // PARTIALS REMOTE-RECEIVE PHASE
                // Accumulate partial sums from other cores contributing to this tile
                // Writer signals how many partials to expect

                // Determine how many partials to accumulate
                uint32_t num_partials;
                if (tile_idx == last_tile) {
                    num_partials = last_tile_other_contributors;
                } else {
                    // Should not happen for tiles before last if started but not ended
                    DPRINT_PACK(DPRINT << "ERROR: non-last tile but started and not ended" << ENDL());
                    num_partials = 0;
                }

                DPRINT_PACK(
                    DPRINT << "Compute: tile " << tile_idx << " accumulating " << num_partials << " partials"
                           << ENDL());

                // Initialize the binary op ONCE before the loop (not per-partial!)
                binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_id_partials);

                // Wait for all partials and accumulate them one by one
                for (uint32_t p = 0; p < num_partials; p++) {
                    // Wait for this partial to arrive from writer
                    cb_wait_front(cb_id_partials, 1);

                    // Accumulate: dst_reg += partial (init already done outside loop)
                    binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                        cb_id_partials, 0, dst_reg);

                    cb_pop_front(cb_id_partials, 1);
                }

                // Pack final accumulated result
                tile_regs_commit();
                tile_regs_wait();

                cb_reserve_back(cb_out, 1);
                pack_tile(dst_reg, cb_out);
                cb_push_back(cb_out, 1);

                tile_regs_release();

                // Reset SFPU state for mm (next iteration)
                mm_init(cb_in0, cb_in1, cb_out);
            } else {
                // tile_started && tile_ended: this core both starts and finishes the tile
                tile_regs_commit();
                tile_regs_wait();

                cb_reserve_back(cb_out, 1);
                pack_tile(dst_reg, cb_out);
                cb_push_back(cb_out, 1);

                tile_regs_release();
            }
        }

        iter = tile_iter_end;
    }
    DPRINT << "Compute finished" << ENDL();
}
}  // namespace NAMESPACE
