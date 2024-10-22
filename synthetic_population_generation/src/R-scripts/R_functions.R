# Import required R libraries
library(bnlearn)
library(Rfast)
library(emplik)
library(doParallel)
library(parallel)


BN_function <- function(data, blacklist){
    # BN function using R library bnlearn
    # 
    # Args
    #----------
    # data: dataFrame
    #     dataFrame with the data to be fitted
    # blacklist: blacklist_type
    #     forbidden directions of the variables
    # 
    # Returns
    #----------
    # bayesian_network: 
    #     bayesian network model

    # Convert all columns to float
    data <- lapply(data, as.numeric)

    # Build and fit the bayesian network
    bayesian_netowrk = bnlearn::mmhc(as.data.frame(data), blacklist=blacklist)

    return(bayesian_netowrk)
}

bnmat_function <- function(dag) {
    # bnmat function in R
    #
    # Args
    #----------
    # dag: Bayesian network learned by Hybrid methods
    #    bayesian network object trained
    # Returns
    #----------
    # G: matrix
    #   adjacency matrix representing the DAG

    G <- bnlearn::amat(dag)

    return(G)
}

topological_sort_function <- function(G){
    
    ela <- topological_sort(G)

    return(ela)
}

el_test_function <- function(x, m){
    
    b <- el.test(x, m)

    return(b)
}

normalize_weights <- function(b){
    
    weights_norm <- b$wts / sum(b$wts)

    return(weights_norm)
}

ceiling_wei1_function <- function(weights_norm, N_farms){

    B <- ceiling(N_farms*max(weights_norm))

    return(B)
}
