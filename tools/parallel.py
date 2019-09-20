# usage:
#   simply replace `map` with `parallel_map`


import multiprocessing
import threading


lock = threading.Lock()
context = {}
job_id_seq = 0


def parallel_map(f, items, n_procs=None):
    global job_id_seq

    if n_procs == 1:
        return map(f, items)
    else:
        with lock:
            job_id = job_id_seq
            job_id_seq += 1
        context[job_id] = (f, items)
        pool = multiprocessing.Pool(n_procs)
        try:
            return pool.map(wrapper, ((job_id, i) for i in range(len(items))))
        finally:
            pool.close()
            del context[job_id]


def wrapper(args):
    job_id, index = args
    f, items = context[job_id]
    return f(items[index])
