import numpy as np
from coaldecoder import TrioCoalescenceRateModel
import msprime


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
    demographic_parameters[2,2] = np.inf
    admixture_coefficients = np.zeros((3, 3, grid_size - 1))
    for i in range(grid_size - 1): admixture_coefficients[:, :, i] = np.eye(3, 3)

    # mass migration
    admixture_coefficients[0, 0, 100] = 0.3
    admixture_coefficients[0, 1, 100] = 0.0
    admixture_coefficients[1, 0, 100] = 0.7
    admixture_coefficients[1, 1, 100] = 1.0

    decoder = TrioCoalescenceRateModel(3)
    expected_rates = decoder.forward(demographic_parameters, admixture_coefficients, np.diff(time_grid))

    state_labels = np.array(decoder.labels(['0','1','2']))
    pair_subset = np.isin(state_labels, ["t1::((0,0),2)", "t1::((0,1),2)", "t1::((1,1),2)"])
    trio_subset = np.isin(state_labels, 
        ["t2::((0,0),0)", "t2::((0,0),1)", "t2::((0,1),0)", "t2::((0,1),1)", "t2::((1,1),0)", "t2::((1,1),1)"]
    )
    subset = pair_subset if pairs_only else trio_subset

    return demographic_parameters[:2, :2], admixture_coefficients[:2, :2], np.diff(time_grid), expected_rates[subset]


def to_msprime(demographic_parameters, admixture_coefficients, time_step, population_names):
    assert demographic_parameters.shape == admixture_coefficients.shape
    assert len(population_names) == demographic_parameters.shape[0] == demographic_parameters.shape[1]
    assert len(time_step) == demographic_parameters.shape[2]

    demography = msprime.Demography()
    for i, p in enumerate(population_names):
        demography.add_population(initial_size=np.inf, name=p)

    start_time = np.cumsum(np.append(0, time_step))
    demographic_parameters = demographic_parameters.transpose(2, 0, 1)
    admixture_coefficients = admixture_coefficients.transpose(2, 0, 1)
    for M, A, t in zip(demographic_parameters, admixture_coefficients, start_time):
        for i, p in enumerate(population_names):
            for j, q in enumerate(population_names):
                if i != j and A[j, i] > 0:
                    demography.add_mass_migration(time=t, source=p, dest=q, proportion=A[j, i])

        for i, p in enumerate(population_names):
            for j, q in enumerate(population_names):
                if i == j:
                    demography.add_population_parameters_change(time=t, initial_size=M[i, i] / 2, population=p)
                else:
                    demography.add_migration_rate_change(time=t, rate=M[i, j], source=p, dest=q)

    return demography


#--- sanity check
def sanity_check():
    M, A, T, rates = rates_and_demography()
    demography = to_msprime(M, A, T, ["A", "B"])
    debugger = demography.debug()
    time_steps = np.append(0, np.cumsum(T))
    
    import matplotlib.pyplot as plt
    rates_check_AA = debugger.coalescence_rate_trajectory(lineages={"A":2}, steps=time_steps)[1]
    rates_check_AA = np.diff(1 - rates_check_AA) / rates_check_AA[:-1] / np.diff(time_steps)
    rates_check_AB = debugger.coalescence_rate_trajectory(lineages={"A":1,"B":1}, steps=time_steps)[1]
    rates_check_AB = np.diff(1 - rates_check_AB) / rates_check_AB[:-1] / np.diff(time_steps)
    rates_check_BB = debugger.coalescence_rate_trajectory(lineages={"B":2}, steps=time_steps)[1]
    rates_check_BB = np.diff(1 - rates_check_BB) / rates_check_BB[:-1] / np.diff(time_steps)
    fig, axs = plt.subplots(1, 3, figsize=(3*4, 4))
    axs[0].plot(time_steps[:-1], rates_check_AA, label='msprime', c='red')
    axs[0].plot(time_steps[:-1], rates[0], label='coaldecoder', c='black', ls='dashed')
    axs[0].legend()
    axs[1].plot(time_steps[:-1], rates_check_AB, label='msprime', c='blue')
    axs[1].plot(time_steps[:-1], rates[1], label='coaldecoder', c='black', ls='dashed')
    axs[1].legend()
    axs[2].plot(time_steps[:-1], rates_check_BB, label='msprime', c='green')
    axs[2].plot(time_steps[:-1], rates[2], label='coaldecoder', c='black', ls='dashed')
    axs[2].legend()
    plt.savefig("/home/natep/public_html/trio-pres/tmp/msprime_sanity_check.png")

sanity_check()
