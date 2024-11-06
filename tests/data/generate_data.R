# Using the R version of coaldecoder at `github.com/nspope/coaldecoder`

library(coaldecoder)
reticulate::use_condaenv("base")
reticulate::source_python(system.file("python", "calculate_rates.py", package = "coaldecoder"))
np <- reticulate::import("numpy")

num_pops <- 3
haps <- 10

time_breaks <- seq(0.0, 50000, length.out=1001)
epoch_start <- time_breaks[2:length(time_breaks)-1]
M <- array(0, c(num_pops,num_pops,length(epoch_start)))
M[1,1,] <- 100000 + 50000*cos(2*pi*epoch_start/15000)
M[1,2,] <- 1e-4 * exp(log(1e-6/1e-4)/40000 * epoch_start)
M[1,3,] <- 1e-5
M[2,1,] <- 1e-6 * exp(log(1e-4/1e-6)/40000 * epoch_start)
M[2,2,] <- 100000 + 50000*cos(2*pi*(epoch_start+15000)/15000)
M[2,3,] <- ifelse(epoch_start > 15000 & epoch_start < 25000, 1e-5, 1e-4)
M[3,1,] <- 1e-5
M[3,2,] <- ifelse(epoch_start > 15000 & epoch_start < 25000, 1e-5, 1e-6)
M[3,3,] <- 100000

pop_model <- PopulationTree$new("((A:30000,C:20000):10000,B:40000);", time_breaks=time_breaks)
pop_model$set_demographic_parameters(M)

if(!file.exists("coaldecoder_example_chr1.ts")) {
  pop_model$msprime_simulate(
    outfile="coaldecoder_example", 
    sample_sizes=c(haps,haps,haps), 
    chromosomes=1, 
    chromosome_length=1e8, 
    recombination_rate=1.15e-8,
    mutation_rate=1.29e-8,
    random_seed=1024, 
    what="tree_sequence"
  )
}

if(!file.exists("observed_rates_values.npy")){
  # calculate "observed" trio coalescence rates within twenty-five 2500-generation epochs
  obs_time_breaks <- seq(0, 50000, 2500)
  sample_sets <- list( #sample indices in the tree sequence for each population
    "A" = c(0:9),
    "B" = c(10:19),
    "C" = c(20:29)
  )
  obs_rates <- ObservedTrioRates(
    ts = "coaldecoder_example_chr1.ts",
    sample_sets = sample_sets,
    time_breaks = obs_time_breaks,
    bootstrap_blocks = 100,
    mask = NULL,
    threads = 1 #currently ignored
  )

  # extract rates and bootstrap precision
  rates <- obs_rates$rates()
  rates_sd <- obs_rates$std_dev(num_replicates=100, random_seed=1)
  rownames(rates) <- rownames(rates_sd) <- obs_rates$emission_states()
  colnames(rates) <- colnames(rates_sd) <- obs_rates$epochs()
  np$save("observed_rates_names.npy", obs_rates$emission_states())
  np$save("observed_rates_values.npy", rates)
  np$save("observed_rates_stddev.npy", rates_sd)
  np$save("observed_rates_breaks.npy", obs_time_breaks)
}

decoder <- CoalescentDecoder$new(num_pops, pop_model$epoch_durations(), TRUE)
labels <- decoder$emission_states(c("A", "B", "C"))
dummy_rates <- matrix(1, length(labels), length(time_breaks) - 1)
dummy_weights <- matrix(1, length(labels), length(time_breaks) - 1)
out <- decoder$loglikelihood(dummy_rates, dummy_weights, decoder$initial_state_vectors(), pop_model$demographic_parameters(), pop_model$admixture_coefficients())
np$save("demographic_parameters_gradient.npy", out$gradient$M)
exit()

np$save("expected_coalescence_rates.npy", pop_model$expected_coalescence_rates())
np$save("epoch_durations.npy", pop_model$epoch_durations())
np$save("demographic_parameters.npy", pop_model$demographic_parameters())
np$save("admixture_coefficients.npy", pop_model$admixture_coefficients())

