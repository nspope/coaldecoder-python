import numpy as np
import nlopt
from coaldecoder import CoalescentDecoder
import matplotlib
import matplotlib.pyplot as plt

def rates_and_demography(
        intercept=[1e5, 1e5], 
        phase=[0, 5e3], 
        frequency=[1/1e4, 1/1e4], 
        amplitude=[9e4, 9e4], 
        pulse_mode=[2e4, 4e4], 
        pulse_sd=[1e3, 1e3], 
        pulse_on=[1e-4, 1e-3], 
        pulse_off=[1e-6, 1.1e-6], 
        grid_size=1000,
        pairs_only=True,
    ):
    """
    Pair rates for two oscillating populations with pulse migration
    """

    time_grid = np.linspace(0, 5e4, grid_size)
    epoch_start = time_grid[:-1]
    demographic_parameters = np.zeros((3, 3, grid_size - 1))
    demographic_parameters[0,0] = intercept[0] + amplitude[0] * np.cos(2 * np.pi * (epoch_start + phase[0]) * frequency[0])
    demographic_parameters[0,1] = pulse_off[0] + np.exp(-(epoch_start - pulse_mode[0]) ** 2 / pulse_sd[0] ** 2) * (pulse_on[0] - pulse_off[0])
    demographic_parameters[1,0] = pulse_off[1] + np.exp(-(epoch_start - pulse_mode[1]) ** 2 / pulse_sd[1] ** 2) * (pulse_on[1] - pulse_off[1])
    demographic_parameters[1,1] = intercept[1] + amplitude[1] * np.cos(2 * np.pi * (epoch_start + phase[1]) * frequency[1])
    demographic_parameters[2,2] = 10000
    admixture_coefficients = np.zeros((3, 3, grid_size - 1))
    for i in range(grid_size - 1): admixture_coefficients[:, :, i] = np.eye(3, 3)

    decoder = CoalescentDecoder(3, True)
    states = decoder.initial_state_vectors()
    _, expected_rates = decoder.forward(states, demographic_parameters, admixture_coefficients, np.diff(time_grid))

    state_labels = np.array(decoder.emission_states(['0','1','2']))
    pair_subset = np.isin(state_labels, ["t1::((0,0),2)", "t1::((0,1),2)", "t1::((1,1),2)"])
    trio_subset = np.isin(state_labels, 
        ["t2::((0,0),0)", "t2::((0,0),1)", "t2::((0,1),0)", "t2::((0,1),1)", "t2::((1,1),0)", "t2::((1,1),1)"]
    )
    subset = pair_subset if pairs_only else trio_subset

    return epoch_start, expected_rates[subset], demographic_parameters[:2, :2]


# ---
def unconstrained_optimize(target, duration, starting_value, lower_bound, upper_bound):
    assert np.logical_and(starting_value >= lower_bound, starting_value <= upper_bound)

    decoder = CoalescentDecoder(starting_value.shape[0])
    mapping = np.arange(starting_value.size).reshape(starting_value.shape)
    initial_state = decoder.initial_state_vectors()

    # temporary
    weights = np.ones(target.shape)
    admixture = np.zeros(starting_value.shape)
    for i in range(admixture.shape[3]): admixture[:, :, i] = np.eye(admixture[:, :, i].shape)
    # /temporary

    def objective(par, grad):
        demography = np.exp(par[mapping])
        _, fitted = decoder.forward(initial_state, demography, admixture, duration)
        resid = (target - fitted) * weights
        if grad.size:
            _, gradient, _ = decoder.backward(np.zeros(initial_state.shape), resid * weights) 
            gradient = numpy.bincount(mapping.flatten(), weights=gradient.flatten())
            grad[:] = gradient * np.exp(par)  # logspace
        return 0.5 * np.mean(resid ** 2)

    lower_bound = np.log(lower_bound).flatten()
    upper_bound = np.log(upper_bound).flatten()
    starting_value = np.log(starting_value).flatten()

    optimizer = nlopt.opt("NLOPT_LD_LBFGS", starting_value.size)
    optimizer.set_min_objective(objective)
    optimizer.set_lower_bound(lower_bound)
    optimizer.set_upper_bound(upper_bound)
    optimizer.set_vector_storage(50)
    convergence = optimizer.optimize(starting_value)
    parameters = optimizer.last_optimum_result()
    loglik = optimizer.last_optimum_value()

    demography = np.exp(parameters[mapping])
    _, fitted = decoder.forward(initial_state, demography, admixture, duration)

    return convergence, loglik, parameters, fitted



# ---
start, rates, params = rates_and_demography(pairs_only=True, **kwargs)
