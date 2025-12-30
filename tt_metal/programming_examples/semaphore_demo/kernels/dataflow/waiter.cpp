#include "dataflow_api.h"

void kernel_main() {
    uint32_t arg_idx = 0;
    uint32_t my_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t my_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t sem_id = get_arg_val<uint32_t>(arg_idx++);

    uint32_t sem_addr = get_semaphore(sem_id);
    volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    uint64_t my_sem_noc = get_noc_addr(my_x, my_y, sem_addr);

    DPRINT << "Waiter: waiting on semaphore at " << my_sem_noc << ENDL();
    noc_semaphore_wait(sem_ptr, 1);
    DPRINT << "Waiter: received signal" << ENDL();
}
