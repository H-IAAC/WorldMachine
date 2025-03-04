import random
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import (
    Lock, Pool, RLock, current_process, freeze_support, set_start_method)
from time import sleep

from tqdm import tqdm


def initializer(the_lock):
    tqdm.set_lock(the_lock)


def progresser(n):

    text = f'#{n}'
    sampling_counts = 10
    current = current_process()
    pos = current._identity[0]-1

    print(current._identity)

    with tqdm(total=sampling_counts, desc=text, position=pos) as pbar:
        for i in range(sampling_counts):
            sleep(random.uniform(0, 1))
            pbar.update(1)


if __name__ == '__main__':
    n_worker = 2

    current = current_process()
    print("MAIN", current._identity)

    L = list(range(n_worker))  # works until 23, breaks starting at 24

    lock = RLock()

    executor = ProcessPoolExecutor(
        n_worker, initializer=initializer, initargs=(lock,))

    futures = []
    for i in range(n_worker):
        f = executor.submit(progresser, i)

        futures.append(f)

    wait(futures)
