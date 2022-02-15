import ray
import time
import sys
import os

import numpy as np


@ray.remote
def calcnp(n):
    for _ in range(n):
        a = np.arange(1e6)
        b = sum(a**2)


@ray.remote
def calcwait(n):
    for _ in range(n):
        a = sum(x**2 for x in range(1000000))


@ray.remote
class RayClass:
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    # useray = bool(int(sys.argv[1]))
    # nump = bool(int(sys.argv[2]))

    for useray in [False, True]:
        for nump in [True, False]:
            n = 1 if useray else 10
            ps = 10 if useray else 1
            func = calcnp if nump else calcwait

            start_t = time.time()
            cs = [func.remote(n) for _ in range(ps)]
            results = [ray.get(c) for c in cs]
            time_taken = time.time() - start_t

            print(
                f"Running fn: {func._function.__name__:10} across {ps:3} processes. Time taken: {time_taken:5.3f}s"
            )
