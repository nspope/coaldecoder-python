"""
Test against some frozen results from a previous iteration of the package
"""

import numpy as np
from coaldecoder import TrioCoalescenceRateModel, PairCoalescenceRateModel
import time

def test_trio_coalescence_rates():
    migr_mat = np.load("data/demographic_parameters.npy")
    admx_mat = np.load("data/admixture_coefficients.npy")
    time_vec = np.load("data/epoch_durations.npy")
    exp_rate = np.load("data/expected_coalescence_rates.npy")
    migr_gra = np.load("data/demographic_parameters_gradient.npy")

    decoder = TrioCoalescenceRateModel(3)
    check_exp_rate = decoder.forward(migr_mat, admx_mat, time_vec)
    np.testing.assert_allclose(exp_rate, check_exp_rate)

    target = np.ones_like(exp_rate)
    weight = np.ones_like(exp_rate)
    resid = (target - exp_rate) * weight
    check_migr_gra, *_ = decoder.backward(resid * weight)
    np.testing.assert_allclose(migr_gra, check_migr_gra)

    #DEBUG -- move into a test gradients routine with smaller arrays
    import numdifftools as nd
    idx = [0, 1, 0]

    def replace(arr, val):
        out = arr.copy()
        out[idx[0], idx[1], idx[2]] = val
        return out

    def obj(x):
        m = replace(migr_mat, x)
        f = decoder.forward(m, admx_mat, time_vec)
        r = (target - f) * weight
        return np.sum(-0.5 * r ** 2)

    der = nd.Derivative(obj, n=1, step=migr_mat[idx[0], idx[1], idx[2]] * 1e-2)
    print("-->", der(migr_mat[idx[0], idx[1], idx[2]]), check_migr_gra[idx[0], idx[1], idx[2]])
    #/DEBUG

    print("Rate labels", decoder.labels(["0","1","2"]))

test_trio_coalescence_rates()


#----------------
def test_pair_coalescence_rates():
    grid_size = 1000
    time_grid = np.linspace(0, 5e4, grid_size)
    epoch_start = time_grid[:-1]
    demographic_parameters = np.zeros((3, 3, grid_size - 1))
    demographic_parameters[0,0] = 1e5 + 9e4 * np.cos(2 * np.pi * (epoch_start + 0) * 1/15e3)
    demographic_parameters[0,1] = 1e-6 + np.exp(-(epoch_start - 1e4) ** 2 / 1e3 ** 2) * (1e-4 - 1e-6)
    demographic_parameters[1,0] = 1e-6 + np.exp(-(epoch_start - 4e4) ** 2 / 1e3 ** 2) * (1e-3 - 1e-6)
    demographic_parameters[1,1] = 1e5 + 9e4 * np.cos(2 * np.pi * (epoch_start + 5e3) * 1/15e3)
    demographic_parameters[2,2] = np.inf
    admixture_coefficients = np.zeros((3, 3, grid_size - 1))
    for i in range(grid_size - 1): admixture_coefficients[:, :, i] = np.eye(3, 3)

    decoder = TrioCoalescenceRateModel(3)
    expected_rates = decoder.forward(demographic_parameters, admixture_coefficients, np.diff(time_grid))
    state_labels = np.array(decoder.labels(['0','1','2']))
    pair_subset = np.isin(state_labels, ["t1::((0,0),2)", "t1::((0,1),2)", "t1::((1,1),2)"])
    expected_rates = expected_rates[pair_subset]

    decoder_pair = PairCoalescenceRateModel(2)
    check_exp_rate = decoder_pair.forward(demographic_parameters[:2, :2],  admixture_coefficients[:2, :2], np.diff(time_grid))
    print(decoder_pair.labels(['0','1']))

    np.testing.assert_allclose(expected_rates, check_exp_rate)

#test_pair_coalescence_rates()
