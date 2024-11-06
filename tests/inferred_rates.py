import coaldecoder
from coaldecoder import TrioCoalescenceRateModel
import numpy as np
import tskit

# might need an easier example
# such as, OOA
ts = tskit.load("/sietch_colab/natep/trio-coal/sims/osc-log/singer-snakemake/results/osclog_1/trees/osclog_1.99.trees").trim()

sample_population = np.full(ts.num_samples, '')
for i in ts.individuals():
    sample_population[i.nodes] = i.metadata['population']
population_names = np.unique(sample_population)
sample_sets = []
for p in population_names:
    sample_sets.append(np.flatnonzero(sample_population == p))

breaks_size = 50
log_breaks = np.linspace(2, 5, breaks_size)
time_breaks = 10 ** log_breaks
time_breaks[0] = 0

windows = np.linspace(0, ts.sequence_length, 10)

rates_calculator = coaldecoder.TrioCoalescenceRates(ts, sample_sets, time_breaks, sample_set_names=population_names, bootstrap_blocks=None) #windows)
for i in range(2, 6):
    ts = tskit.load(f"/sietch_colab/natep/trio-coal/sims/osc-log/singer-snakemake/results/osclog_{i}/trees/osclog_{i}.99.trees").trim()
    tmp = coaldecoder.TrioCoalescenceRates(ts, sample_sets, time_breaks, sample_set_names=population_names, bootstrap_blocks=None) #windows)
    rates_calculator.join(tmp)
emp_rates = rates_calculator.rates()


#----
#def rates_and_demography(
#        intercept=[1e5, 1e5], 
#        phase=[0, 5e3], 
#        frequency=[1/1e4, 1/1e4], 
#        amplitude=[9e4, 9e4], 
#        pulse_mode=[2e4, 4e4], 
#        pulse_sd=[1e3, 1e3], 
#        pulse_on=[1e-4, 1e-3], 
#        pulse_off=[1e-6, 1.1e-6], 
#        grid_size=1000,
#        pairs_only=True,
#    ):
#    """
#    Pair rates for two oscillating populations with pulse migration
#    """
#
#    time_grid = np.linspace(0, 5e4, grid_size)
#    epoch_start = time_grid[:-1]
#    demographic_parameters = np.zeros((3, 3, grid_size - 1))
#    demographic_parameters[0,0] = intercept[0] + amplitude[0] * np.cos(2 * np.pi * (epoch_start + phase[0]) * frequency[0])
#    demographic_parameters[0,1] = pulse_off[0] + np.exp(-(epoch_start - pulse_mode[0]) ** 2 / pulse_sd[0] ** 2) * (pulse_on[0] - pulse_off[0])
#    demographic_parameters[1,0] = pulse_off[1] + np.exp(-(epoch_start - pulse_mode[1]) ** 2 / pulse_sd[1] ** 2) * (pulse_on[1] - pulse_off[1])
#    demographic_parameters[1,1] = intercept[1] + amplitude[1] * np.cos(2 * np.pi * (epoch_start + phase[1]) * frequency[1])
#    demographic_parameters[2,2] = np.inf
#    admixture_coefficients = np.zeros((3, 3, grid_size - 1))
#    for i in range(grid_size - 1): admixture_coefficients[:, :, i] = np.eye(3, 3)
#
#    decoder = coaldecoder.TrioCoalescenceRateModel(3)
#    expected_rates = decoder.forward(demographic_parameters, admixture_coefficients, np.diff(time_grid))
#
#    state_labels = np.array(decoder.labels(['0','1','2']))
#    pair_subset = np.isin(state_labels, ["t1::((0,0),2)", "t1::((0,1),2)", "t1::((1,1),2)"])
#    trio_subset = np.isin(state_labels, 
#        ["t1::((0,0),0)", "t1::((0,0),1)", "t1::((0,1),0)", "t1::((0,1),1)", "t1::((1,1),0)", "t1::((1,1),1)"] +
#        ["t2::((0,0),0)", "t2::((0,0),1)", "t2::((0,1),0)", "t2::((0,1),1)", "t2::((1,1),0)", "t2::((1,1),1)"]
#    )
#    subset = pair_subset if pairs_only else trio_subset
#
#    return demographic_parameters[:2, :2], admixture_coefficients[:2, :2], time_grid, expected_rates[subset]


def rates_and_demography(
        intercept=[2e4, 2e4], 
        phase=[0, 1.0], 
        frequency=[0.50, 0.50], 
        amplitude=[15e3, 15e3], 
        pulse_mode=[3.5, 4.7], 
        pulse_sd=[0.1, 0.1], 
        pulse_on=[1e-4, 1e-3], 
        pulse_off=[1e-6, 1.1e-6], 
        grid_size=1000,
        log_grid=True,
        pairs_only=True,
    ):
    """
    Pair rates for two oscillating populations with pulse migration
    """

    log_grid = np.linspace(2, 5, grid_size)
    time_grid = 10 ** log_grid
    time_grid[0] = 0.0
    log_start = log_grid[:-1]
    epoch_start = time_grid[:-1]
    demographic_parameters = np.zeros((3, 3, grid_size - 1))
    demographic_parameters[0,0] = intercept[0] + amplitude[0] * np.cos(2 * np.pi * (log_start + phase[0]) * frequency[0])
    demographic_parameters[0,1] = pulse_off[0] + np.exp(-(log_start - pulse_mode[0]) ** 2 / pulse_sd[0] ** 2) * (pulse_on[0] - pulse_off[0])
    demographic_parameters[1,0] = pulse_off[1] + np.exp(-(log_start - pulse_mode[1]) ** 2 / pulse_sd[1] ** 2) * (pulse_on[1] - pulse_off[1])
    demographic_parameters[1,1] = intercept[1] + amplitude[1] * np.cos(2 * np.pi * (log_start + phase[1]) * frequency[1])
    demographic_parameters[2,2] = np.inf
    admixture_coefficients = np.zeros((3, 3, grid_size - 1))
    for i in range(grid_size - 1): admixture_coefficients[:, :, i] = np.eye(3, 3)

    decoder = TrioCoalescenceRateModel(3)
    expected_rates = decoder.forward(demographic_parameters, admixture_coefficients, np.diff(time_grid))

    state_labels = np.array(decoder.labels(['0','1','2']))
    pair_subset = np.isin(state_labels, ["t1::((0,0),2)", "t1::((0,1),2)", "t1::((1,1),2)"])
    trio_subset = np.isin(state_labels, 
        ["t1::((0,0),0)", "t1::((0,0),1)", "t1::((0,1),0)", "t1::((0,1),1)", "t1::((1,1),0)", "t1::((1,1),1)"] +
        ["t2::((0,0),0)", "t2::((0,0),1)", "t2::((0,1),0)", "t2::((0,1),1)", "t2::((1,1),0)", "t2::((1,1),1)"]
    )
    subset = pair_subset if pairs_only else trio_subset

    return demographic_parameters[:2, :2], admixture_coefficients[:2, :2], time_grid, expected_rates[subset]



_, _, time_grid, exp_rates = rates_and_demography(pairs_only=False)

print(exp_rates.shape, emp_rates.shape)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(exp_rates.shape[0] // 2, 2, figsize=(2*4, exp_rates.shape[0]//2*4))
axs[0,0].plot(time_grid[:-1], exp_rates[0])
axs[0,0].set_xscale('log')
axs[0,0].set_yscale('log')
axs[1,0].plot(time_grid[:-1], exp_rates[1])
axs[1,0].set_xscale('log')
axs[1,0].set_yscale('log')
axs[2,0].plot(time_grid[:-1], exp_rates[2])
axs[2,0].set_xscale('log')
axs[2,0].set_yscale('log')
axs[3,0].plot(time_grid[:-1], exp_rates[3])
axs[3,0].set_xscale('log')
axs[3,0].set_yscale('log')
axs[4,0].plot(time_grid[:-1], exp_rates[4])
axs[4,0].set_xscale('log')
axs[4,0].set_yscale('log')
axs[5,0].plot(time_grid[:-1], exp_rates[5])
axs[5,0].set_xscale('log')
axs[5,0].set_yscale('log')
axs[0,1].plot(time_grid[:-1], exp_rates[6])
axs[0,1].set_xscale('log')
axs[0,1].set_yscale('log')
axs[1,1].plot(time_grid[:-1], exp_rates[7])
axs[1,1].set_xscale('log')
axs[1,1].set_yscale('log')
axs[2,1].plot(time_grid[:-1], exp_rates[8])
axs[2,1].set_xscale('log')
axs[2,1].set_yscale('log')
axs[3,1].plot(time_grid[:-1], exp_rates[9])
axs[3,1].set_xscale('log')
axs[3,1].set_yscale('log')
axs[4,1].plot(time_grid[:-1], exp_rates[10])
axs[4,1].set_xscale('log')
axs[4,1].set_yscale('log')
axs[5,1].plot(time_grid[:-1], exp_rates[11])
axs[5,1].set_xscale('log')
axs[5,1].set_yscale('log')
axs[0,0].step(time_breaks[:-1], emp_rates[0], color='black', where='post')
axs[1,0].step(time_breaks[:-1], emp_rates[1], color='black', where='post')
axs[2,0].step(time_breaks[:-1], emp_rates[2], color='black', where='post')
axs[3,0].step(time_breaks[:-1], emp_rates[3], color='black', where='post')
axs[4,0].step(time_breaks[:-1], emp_rates[4], color='black', where='post')
axs[5,0].step(time_breaks[:-1], emp_rates[5], color='black', where='post')
axs[0,1].step(time_breaks[:-1], emp_rates[6], color='black', where='post')
axs[1,1].step(time_breaks[:-1], emp_rates[7], color='black', where='post')
axs[2,1].step(time_breaks[:-1], emp_rates[8], color='black', where='post')
axs[3,1].step(time_breaks[:-1], emp_rates[9], color='black', where='post')
axs[4,1].step(time_breaks[:-1], emp_rates[10], color='black', where='post')
axs[5,1].step(time_breaks[:-1], emp_rates[11], color='black', where='post')
plt.tight_layout()
plt.savefig("/home/natep/public_html/trio-pres/tmp/inferred_trio_rates_check_2.png")

