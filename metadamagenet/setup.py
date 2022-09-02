import os
import random
import numpy as np


def single_thread_numpy():
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"


def set_random_seeds():
    random.seed(1)
    np.random.seed(1)



