#include "dataflow_api.h"

void kernel_main() {
    uint32_t arg_idx = 0;
    uint32_t peer_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t peer_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t sem_id = get_arg_val<uint32_t>(arg_idx++);

    // Compute remote semaphore NoC address using the same L1 offset as local.
    uint32_t local_sem_addr = get_semaphore(sem_id);
    uint64_t peer_sem_noc = get_noc_addr(peer_x, peer_y, local_sem_addr);

    DPRINT << "Signaler: incrementing remote semaphore at " << peer_sem_noc << ENDL();
    noc_semaphore_inc(peer_sem_noc, 1);
    noc_async_atomic_barrier();
    DPRINT << "Signaler: inc done" << ENDL();
}
