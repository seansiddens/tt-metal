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

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    // Setup the FPU (matrix engine) for the matmul operation. And specify the input
    // and output circular buffers.
    mm_init(cb_in0, cb_in1, cb_out);

    // Stream-K compute kernel: Process MAC iterations assigned to this core.
    uint32_t prev_tile_idx = UINT32_MAX;  // Track previous tile to detect transitions

    for (uint32_t mac_iter = mac_start; mac_iter < mac_end; mac_iter++) {
        // Determine which output tile and K iteration this MAC belongs to
        uint32_t tile_idx = mac_iter / macs_per_tile;
        uint32_t k_idx = mac_iter % macs_per_tile;

        // Check if we're transitioning to a new output tile
        if (tile_idx != prev_tile_idx) {
            // Finish and write the previous tile (if not first iteration)
            if (prev_tile_idx != UINT32_MAX) {
                tile_regs_commit();
                tile_regs_wait();

                cb_reserve_back(cb_out, 1);
                pack_tile(0, cb_out);
                cb_push_back(cb_out, 1);

                tile_regs_release();
            }

            // Start accumulating a new output tile
            tile_regs_acquire();
            prev_tile_idx = tile_idx;
        }

        // Wait for the input tiles (A and B) from the reader
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        // Perform the tile-wise matrix multiplication (accumulates into registers)
        matmul_tiles(cb_in0, cb_in1, 0, 0, 0);

        // Pop the input tiles since we've consumed them
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }

    // Write the final output tile
    if (prev_tile_idx != UINT32_MAX) {
        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        tile_regs_release();
    }
}
}  // namespace NAMESPACE
