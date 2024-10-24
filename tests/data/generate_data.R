# Using the R version of coaldecoder at `github.com/nspope/coaldecoder`

library(reticulate)
library(coaldecoder)

set.seed(1024)
nP <- 4
nT <- 30

epoch_len <- runif(nT, 1e2, 1e3)
migr_mat <- array(runif(nP * nP * nT, 0, 1e-4), c(nP, nP, nT))
for (i in 1:nT) { diag(migr_mat[,,i]) <- runif(nP, 1e3, 1e5) }
admix_mat <- array(runif(nP * nP * nT, 0, 1), c(nP, nP, nT))

decoder <- CoalescentDecoder(nP, nT, TRUE)
decoder$expected_coalescence_rates(...)

# simulate some data, get empirical coalescence rates, dump these
