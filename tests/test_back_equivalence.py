"""
Test against some frozen results from a previous iteration of the package
"""

import numpy as np
from coaldecoder import CoalescentDecoder
import time

def test_coalescence_rates():
    migr_mat = np.load("data/demographic_parameters.npy")
    admx_mat = np.load("data/admixture_coefficients.npy")
    time_vec = np.load("data/epoch_durations.npy")
    exp_rate = np.load("data/expected_coalescence_rates.npy")
    decoder = CoalescentDecoder(3, True)
    states = decoder.initial_state_vectors()
    st = time.time()
    _, check_exp_rate = decoder.forward(states, migr_mat, admx_mat, time_vec)
    en = time.time()
    print(en - st)
    np.testing.assert_allclose(exp_rate, check_exp_rate)
    print(decoder.emission_states(["0","1","2"]))

test_coalescence_rates()

