import numpy as np
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



#--- 
def make_plot(path, highlight=None, pairs_only=True, **kwargs):
    start, rates, params = rates_and_demography(pairs_only=pairs_only, **kwargs)
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

make_plot("/home/natep/public_html/trio-pres/tmp/test_pair_demog.png", highlight=[0, 10])
make_plot("/home/natep/public_html/trio-pres/tmp/test_trio_demog.png", pairs_only=False, highlight=[0, 10])

#---
def make_anim(path, pairs_only=True, **kwargs):
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
    ra_ax.set_yscale('log')
    ra_ax.set_ylim(y_min_ra, y_max_ra)
    ra_ax.set_xlim(x_min, x_max)
    ra_ax.set_ylabel("Pair coalescence rate")
    ra_ax.legend(ncol=1, loc='lower right')

    fig.supxlabel("Thousands of generations in past")
    fig.tight_layout()

    # smooth deformation of parameters
    num_frames = 100
    assert num_frames % 2 == 0
    phase_A = np.linspace(0, 3e4, num_frames)
    amplt_B = np.linspace(9e4, -9e4, num_frames // 2)
    amplt_B = np.append(amplt_B, amplt_B[::-1])
    pulse_B = np.logspace(-6, -3, num_frames // 2)
    pulse_B = np.append(pulse_B, pulse_B[::-1])
    mode_A = np.linspace(0, 2e4, num_frames // 2)[::-1]
    mode_A = np.append(mode_A, mode_A[::-1])

    def update(frame):
        phase = [phase_A[frame], 5e3]
        amplitude = [9e4, amplt_B[frame]]
        pulse_on = [1e-4, pulse_B[frame]]
        pulse_mode = [mode_A[frame], 4e4]
        start, rates, params = rates_and_demography(pairs_only=pairs_only, phase=phase, amplitude=amplitude, pulse_on=pulse_on, pulse_mode=pulse_mode)
        ne_ln["A"].set_data(start / 1e3, params[0,0])
        ne_ln["B"].set_data(start / 1e3, params[1,1])
        mi_ln["AB"].set_data(start / 1e3, params[0,1])
        mi_ln["BA"].set_data(start / 1e3, params[1,0])
        for i, label in enumerate(rate_names):
            ra_ln[label].set_data(start / 1e3, rates[i])


    from matplotlib.animation import FuncAnimation
    ani = FuncAnimation(fig, update, repeat=True, frames=num_frames, interval=100)
    ani.save(path, writer="imagemagick")


make_anim("/home/natep/public_html/trio-pres/tmp/test_pair_demog.gif", pairs_only=True)
