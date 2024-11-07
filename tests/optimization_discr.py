import numpy as np
import tskit
import os
import pickle
import nlopt
from coaldecoder import TrioCoalescenceRateModel, PairCoalescenceRateModel
from coaldecoder import TrioCoalescenceRates, PairCoalescenceRates
import matplotlib
import matplotlib.pyplot as plt

def rates_and_demography(
        time_grid,
        intercept=[1e5, 1e5], 
        phase=[0, 5e3], 
        frequency=[1/1e4, 1/1e4], 
        amplitude=[9e4, 9e4], 
        pulse_mode=[2e4, 4e4], 
        pulse_sd=[1e3, 1e3], 
        pulse_on=[1e-4, 1e-3], 
        pulse_off=[1e-6, 1.1e-6], 
        pairs_only=True,
    ):
    """
    Pair rates for two oscillating populations with pulse migration
    """

    epoch_start = time_grid[:-1]
    demographic_parameters = np.zeros((3, 3, grid_size - 1))
    demographic_parameters[0,0] = intercept[0] + amplitude[0] * np.cos(2 * np.pi * (epoch_start + phase[0]) * frequency[0])
    demographic_parameters[0,1] = pulse_off[0] + np.exp(-(epoch_start - pulse_mode[0]) ** 2 / pulse_sd[0] ** 2) * (pulse_on[0] - pulse_off[0])
    demographic_parameters[1,0] = pulse_off[1] + np.exp(-(epoch_start - pulse_mode[1]) ** 2 / pulse_sd[1] ** 2) * (pulse_on[1] - pulse_off[1])
    demographic_parameters[1,1] = intercept[1] + amplitude[1] * np.cos(2 * np.pi * (epoch_start + phase[1]) * frequency[1])
    demographic_parameters[2,2] = np.inf
    admixture_coefficients = np.zeros((3, 3, grid_size - 1))
    for i in range(grid_size - 1): admixture_coefficients[:, :, i] = np.eye(3, 3)

    decoder = TrioCoalescenceRateModel(3)
    expected_rates = decoder.forward(demographic_parameters, admixture_coefficients, np.diff(time_grid))

    state_labels = np.array(decoder.labels(['0','1','2']))
    pair_subset = np.isin(state_labels, ["t1::((0,0),2)", "t1::((0,1),2)", "t1::((1,1),2)"])
    trio_subset = np.isin(state_labels, 
        ["t2::((0,0),0)", "t2::((0,0),1)", "t2::((0,1),0)", "t2::((0,1),1)", "t2::((1,1),0)", "t2::((1,1),1)"]
    )
    subset = pair_subset if pairs_only else trio_subset

    return np.diff(time_grid), expected_rates[subset], demographic_parameters[:2, :2]


# ---
def unconstrained_optimize(target, weights, duration, starting_value, lower_bound, upper_bound):
    assert np.logical_and(np.all(starting_value >= lower_bound), np.all(starting_value <= upper_bound))

    num_populations = starting_value.shape[0]
    decoder = PairCoalescenceRateModel(num_populations)
    mapping = np.arange(starting_value.size).reshape(starting_value.shape)

    # temporary
    sampling_interval = 10
    max_samples = 100
    admixture = np.zeros(starting_value.shape)
    for i in range(admixture.shape[-1]): admixture[:, :, i] = np.eye(*admixture[:, :, i].shape)
    # /temporary

    opt_trajectory = []
    loss_trajectory = []

    def objective(par, grad):
        demography = np.exp(par[mapping])
        fitted = decoder.forward(demography, admixture, duration)
        resid = (target - fitted) * weights
        if grad.size:
            d_demography, *_ = decoder.backward(resid * weights) 
            d_par = np.bincount(mapping.flatten(), weights=d_demography.flatten())
            grad[:] = d_par * np.exp(par)  # logspace
        loglik = -0.5 * np.sum(resid ** 2)
        loss_trajectory.append(loglik)
        if len(loss_trajectory) % sampling_interval == 0 and len(opt_trajectory) < max_samples:
            opt_trajectory.append((fitted, demography))
            print(f"{len(loss_trajectory)} loglik {-0.5 * np.sum(resid ** 2)}, grad norm {np.linalg.norm(grad)}")
        return loglik

    lower_bound = np.log(lower_bound).flatten()
    upper_bound = np.log(upper_bound).flatten()
    starting_value = np.log(starting_value).flatten()

    # initialize trajectory with starting state
    objective(starting_value, np.zeros(starting_value.size))

    optimizer = nlopt.opt(nlopt.LD_LBFGS, starting_value.size)
    optimizer.set_max_objective(objective)
    optimizer.set_lower_bounds(lower_bound)
    optimizer.set_upper_bounds(upper_bound)
    optimizer.set_vector_storage(50)
    optimizer.set_ftol_rel(1e-4)
    parameters = optimizer.optimize(starting_value)
    convergence = optimizer.last_optimize_result()
    loglik = optimizer.last_optimum_value()

    demography = np.exp(parameters[mapping])
    fitted = decoder.forward(demography, admixture, duration)

    return demography, fitted, opt_trajectory


# --- get empirical rates
grid_size = 25
time_grid = np.linspace(0, 5e4, grid_size)

# wrap into func, return pairs or trios depending
ts = tskit.load("/sietch_colab/natep/trio-coal/sims/osc/oscillating_1.ts")
#ts = ts.keep_intervals([[0, ts.sequence_length / 10]]).trim()

population_map = {i:ts.population(i).metadata["name"] for i in range(ts.num_populations)}
sample_population = np.array([population_map[i] for i in ts.nodes_population[:ts.num_samples]])
population_names = np.unique(sample_population)
sample_sets = []
for p in population_names:
    sample_sets.append(np.flatnonzero(sample_population == p))

#time_breaks = np.append(np.linspace(0, 50000, 25), np.inf)
windows = np.linspace(0, ts.sequence_length, 10)

rates_calculator = PairCoalescenceRates(ts, sample_sets, time_grid, sample_set_names=population_names, bootstrap_blocks=windows)
emp_rates = rates_calculator.rates()
std_rates = rates_calculator.std_dev(num_replicates=10)


# --- fit model
duration, rates, params = rates_and_demography(time_grid, pairs_only=True)
st = params.copy()
lb = params.copy()
ub = params.copy()
st[:] = 1e-5
lb[:] = 1e-10
ub[:] = 1e-2
for i in range(ub.shape[0]):
    st[i,i] = 1e4
    lb[i,i] = 1e2
    ub[i,i] = 1e6
traj_store = "/home/natep/public_html/trio-pres/tmp/test_pair_demog_discr_optimize.p"
if True: #not os.path.exists(traj_store):
    params_fit, rates_fit, opt_traj = unconstrained_optimize(emp_rates, 1./std_rates, duration, st, lb, ub)
    pickle.dump((params_fit, rates_fit, opt_traj), open(traj_store, "wb"))
else:
    params_fit, rates_fit, opt_traj = pickle.load(open(traj_store, "rb"))


# --- look at optimized model
def make_plot(duration, params, rates, path, highlight=None, pairs_only=True):
    start = np.cumsum(np.append(0, duration))[:-1]
    rate_names = ["(A,A)", "(A,B)", "(B,B)"] if pairs_only else \
        ["((A,A),A)", "((A,A),B)", "((A,B),A)", "((A,B),B)", "((B,B),A)", "((B,B),B)"]
    
    fig = plt.figure(figsize=(8, 4))
    if highlight is not None:
        def focal_highlight():
            return matplotlib.patches.Rectangle((highlight[0], 1e-30), highlight[1] - highlight[0], 1e30, fc = 'gray', alpha=0.3)
    
    ne_ax = plt.subplot2grid((2, 2), (0, 0))
    ne_ax.plot(start / 1e3, params[0,0], label=r"$N_{A}$", color="dodgerblue")
    ne_ax.plot(start / 1e3, params[1,1], label=r"$N_{B}$", color="firebrick")
    ne_ax.set_ylim(8e3, 8e5)
    ne_ax.set_yscale('log')
    ne_ax.set_ylabel("Haploid $N_e$")
    ne_ax.legend(ncol=2, loc='upper left')
    if highlight is not None:
        ne_ax.add_patch(focal_highlight())
    
    mi_ax = plt.subplot2grid((2, 2), (1, 0))
    mi_ax.plot(start / 1e3, params[0,1], label=r"$M_{A \rightarrow B}$", color="dodgerblue", linestyle='dashed')
    mi_ax.plot(start / 1e3, params[1,0], label=r"$M_{B \rightarrow A}$", color="firebrick", linestyle='dashed')
    mi_ax.set_yscale('log')
    mi_ax.set_ylabel("Migration rate")
    mi_ax.legend(ncol=1, loc='upper left')
    if highlight is not None:
        mi_ax.add_patch(focal_highlight())
    
    ra_ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    for i, label in enumerate(rate_names):
        ra_ax.plot(start / 1e3, rates[i], label=label)
    ra_ax.set_yscale('log')
    ra_ax.set_ylabel("Pair coalescence rate")
    ra_ax.legend(ncol=1, loc='lower right')
    if highlight is not None:
        ra_ax.add_patch(focal_highlight())
    
    fig.supxlabel("Thousands of generations in past")
    fig.tight_layout()
    plt.savefig(path)
    plt.clf()

make_plot(duration, params_fit, rates_fit, path="/home/natep/public_html/trio-pres/tmp/test_pair_demog_discr_optimize.png")

assert False

# --- make animation of optimization process
def make_anim(duration, target, optimization_traj, path, pairs_only=True):
    start = np.cumsum(np.append(0, duration))[:-1]
    x_min = 0
    x_max = 50
    y_min_ne = 8e3
    y_max_ne = 8e5
    y_min_mi = 1e-6
    y_max_mi = 1e-3
    y_min_ra = 1e-9 if pairs_only else 1e-13
    y_max_ra = 1e-1 if pairs_only else 1e-3
    rate_names = ["(A,A)", "(A,B)", "(B,B)"] if pairs_only else \
        ["((A,A),A)", "((A,A),B)", "((A,B),A)", "((A,B),B)", "((B,B),A)", "((B,B),B)"]
    
    fig = plt.figure(figsize=(8, 4))
    
    ne_ln = {}
    mi_ln = {}
    ra_ln = {}
    ne_ax = plt.subplot2grid((2, 2), (0, 0))
    ne_ln["A"], *_ = ne_ax.plot([], [], label=r"$N_{A}$", color="dodgerblue")
    ne_ln["B"], *_ = ne_ax.plot([], [], label=r"$N_{B}$", color="firebrick")
    ne_ax.set_ylim(y_min_ne, y_max_ne)
    ne_ax.set_xlim(x_min, x_max)
    ne_ax.set_yscale('log')
    ne_ax.set_ylabel("Haploid $N_e$")
    ne_ax.legend(ncol=2, loc='upper left')
    
    mi_ax = plt.subplot2grid((2, 2), (1, 0))
    mi_ln["AB"], *_ = mi_ax.plot([], [], label=r"$M_{A \rightarrow B}$", color="dodgerblue", linestyle='dashed')
    mi_ln["BA"], *_ = mi_ax.plot([], [], label=r"$M_{B \rightarrow A}$", color="firebrick", linestyle='dashed')
    mi_ax.set_yscale('log')
    mi_ax.set_ylim(y_min_mi, y_max_mi)
    mi_ax.set_xlim(x_min, x_max)
    mi_ax.set_ylabel("Migration rate")
    mi_ax.legend(ncol=1, loc='upper left')
    
    ra_ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    ra_ln = {}
    for i, label in enumerate(rate_names):
        ra_ln[label], *_ = ra_ax.plot([], [], label=label)
    ra_ta = {}
    for i, label in enumerate(rate_names):
        ra_ta[label], *_ = ra_ax.plot(start / 1e3, target[i], label=label, linestyle='dashed')
    ra_ax.set_yscale('log')
    ra_ax.set_ylim(y_min_ra, y_max_ra)
    ra_ax.set_xlim(x_min, x_max)
    if pairs_only:
        ra_ax.set_ylabel("Pair coalescence rate")
    else:
        ra_ax.set_ylabel("Trio coalescence rate")
    ra_ax.legend(ncol=1, loc='lower right')

    fig.supxlabel("Thousands of generations in past")
    fig.tight_layout()

    num_frames = len(optimization_traj)

    def update(frame):
        rates, params = optimization_traj[frame]
        ne_ln["A"].set_data(start / 1e3, params[0,0])
        ne_ln["B"].set_data(start / 1e3, params[1,1])
        mi_ln["AB"].set_data(start / 1e3, params[0,1])
        mi_ln["BA"].set_data(start / 1e3, params[1,0])
        for i, label in enumerate(rate_names):
            ra_ln[label].set_data(start / 1e3, rates[i])

    from matplotlib.animation import FuncAnimation
    ani = FuncAnimation(fig, update, repeat=True, frames=num_frames, interval=100)
    ani.save(path, writer="imagemagick")

#make_anim(duration, rates, opt_traj, path="/home/natep/public_html/trio-pres/tmp/test_pair_demog_discr_optimize.gif", pairs_only=True)
