# Using the R version of coaldecoder at `github.com/nspope/coaldecoder`

library(coaldecoder)
reticulate::use_condaenv("base")
reticulate::source_python(system.file("python", "calculate_rates.py", package = "coaldecoder"))

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

expected_rates <- pop_model$expected_coalescence_rates()
np <- reticulate::import("numpy")
np$save("expected_rates.npy", expected_rates)

