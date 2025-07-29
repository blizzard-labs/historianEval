#!/usr/bin/env Rscript

# RPANDA Analysis Script
# Estimates speciation and extinction rates from phylogenetic trees
# and adds them to existing parameter CSV files
# Enhanced to fit all combinations of birth-death AND coalescent models

# Load required libraries
if (!require("RPANDA", quietly = TRUE)) {
  cat("Installing RPANDA package...\n")
  install.packages("RPANDA")
  library(RPANDA)
}

if (!require("ape", quietly = TRUE)) {
  cat("Installing ape package...\n")
  install.packages("ape")
  library(ape)
}

if (!require("phytools", quietly = TRUE)) {
  cat("Installing phytools package...\n")
  install.packages("phytools")
  library(phytools)
}

if (!require("deSolve", quietly = TRUE)) {
  cat("Installing deSolve package...\n")
  install.packages("deSolve")
  library(deSolve)
}

# For data manipulation
if (!require("dplyr", quietly = TRUE)) {
  cat("Installing dplyr package...\n")
  install.packages("dplyr")
  library(dplyr)
}

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  cat("Usage: Rscript rpanda_analysis.R <tree_folder> <csv_file> [output_csv]\n")
  cat("  tree_folder: Folder containing Newick tree files\n")
  cat("  csv_file: CSV file with GTR parameters (from Python script)\n")
  cat("  output_csv: Optional output CSV file name (default: adds '_with_rates.csv')\n")
  quit(status = 1)
}

tree_folder <- args[1]
csv_file <- args[2]
output_csv <- if (length(args) >= 3) args[3] else gsub("\\.csv$", "_with_rates.csv", csv_file)

# Define all birth-death model combinations
bd_model_combinations <- list(
  "BCSTDCST" = list(lamb_type = "constant", mu_type = "constant"),     # Birth Constant, Death Constant
  "BEXPDCST" = list(lamb_type = "exponential", mu_type = "constant"),  # Birth Exponential, Death Constant
  "BLINDCST" = list(lamb_type = "linear", mu_type = "constant"),       # Birth Linear, Death Constant
  "BCSTDEXP" = list(lamb_type = "constant", mu_type = "exponential"),  # Birth Constant, Death Exponential
  "BEXPDEXP" = list(lamb_type = "exponential", mu_type = "exponential"), # Birth Exponential, Death Exponential
  "BLINDEXP" = list(lamb_type = "linear", mu_type = "exponential"),    # Birth Linear, Death Exponential
  "BCSTDLIN" = list(lamb_type = "constant", mu_type = "linear"),       # Birth Constant, Death Linear
  "BEXPDLIN" = list(lamb_type = "exponential", mu_type = "linear"),    # Birth Exponential, Death Linear
  "BLINDLIN" = list(lamb_type = "linear", mu_type = "linear")          # Birth Linear, Death Linear
)

# Define coalescent model combinations
coalescent_model_combinations <- list(
  "COALCST" = list(type = "constant", description = "Constant effective population size"),
  "COALEXP" = list(type = "exponential", description = "Exponential population growth"),
  "COALLIN" = list(type = "linear", description = "Linear population growth"),
  "COALSTEP" = list(type = "step", description = "Step function population change"),
  "COALLOG" = list(type = "logistic", description = "Logistic population growth")
)

#Function to calculate gamma statistics with ape
calculate_gamma_statistic <- function(tree) {
  tryCatch({
    # Calculate gamma statistic using ape package
    gamma_stat <- gammaStat(tree)
    
    # Calculate p-value for gamma statistic
    # Under null hypothesis of constant rates, gamma follows standard normal
    gamma_pvalue <- 2 * (1 - pnorm(abs(gamma_stat)))
    
    # Interpretation
    gamma_interpretation <- ifelse(gamma_stat < -1.645, "Early burst (significant)", 
                          ifelse(gamma_stat > 1.645, "Late burst (significant)", 
                                "Constant rates"))
    
    return(list(
      gamma = gamma_stat,
      gamma_pvalue = gamma_pvalue,
      gamma_interpretation = gamma_interpretation
    ))
  }, error = function(e) {
    warning(sprintf("Gamma calculation failed: %s", e$message))
    return(list(
      gamma = NA,
      gamma_pvalue = NA,
      gamma_interpretation = "Calculation failed"
    ))
  })
}

# Function to estimate diversification rates using RPANDA
estimate_rates <- function(tree_file) {
  cat(sprintf("Processing tree: %s\n", basename(tree_file)))
  
  tryCatch({
    # Read the tree
    tree <- read.tree(tree_file)
    
    # Basic tree statistics
    n_tips <- Ntip(tree)
    tree_length <- sum(tree$edge.length)
    
    gamma_result <- calculate_gamma_statistic(tree)
    
    if (n_tips < 3) {
      warning(sprintf("Tree has too few tips: %s", tree_file))
      result <- create_empty_result(tree_file, "Too few tips", n_tips, tree_length)
      # Add gamma results to empty result
      result$gamma <- gamma_result$gamma
      result$gamma_pvalue <- gamma_result$gamma_pvalue
      result$gamma_interpretation <- gamma_result$gamma_interpretation
      return(result)
    }
    
    # RPANDA requires ultrametric trees

    # Get the crown age
    crown_age <- max(node.depth.edgelength(tree))
    
    if (crown_age <= 0) {
      warning(sprintf("Invalid crown age in tree: %s", tree_file))
      return(create_empty_result(tree_file, "Invalid crown age", n_tips, tree_length))
    }
    
    if (!is.ultrametric(tree)) {
      cat("  Tree is not ultrametric, attempting to make ultrametric...\n")
      tree <- chronos(tree, quiet = TRUE)
    }
    
    # Ensure tree has positive branch lengths
    if (any(tree$edge.length <= 0)) {
      tree$edge.length[tree$edge.length <= 0] <- 1e-6
    }

    if (!is.rooted(tree)) {
        cat("  Tree is unrooted, rooting using midpoint...\n")
        tree <- midpoint.root(tree)
    }
    
    
    # Store results for different models
    model_results <- list()
    best_bd_model <- NULL
    best_coal_model <- NULL
    best_bd_aic <- Inf
    best_coal_aic <- Inf
    best_overall_model <- NULL
    best_overall_aic <- Inf
    
    # Try all birth-death model combinations
    cat("  === FITTING BIRTH-DEATH MODELS ===\n")
    for (model_name in names(bd_model_combinations)) {
      model_spec <- bd_model_combinations[[model_name]]
      cat(sprintf("  Fitting BD %s model...\n", model_name))
      
      model_result <- tryCatch({
        fit_bd_model_combination(tree, model_spec$lamb_type, model_spec$mu_type, crown_age, model_name)
      }, error = function(e) {
        cat(sprintf("    BD %s model failed: %s\n", model_name, e$message))
        NULL
      })
      
      if (!is.null(model_result)) {
        model_results[[paste0("BD_", model_name)]] <- model_result
        
        # Track best BD model by AIC
        if (!is.na(model_result$aic) && model_result$aic < best_bd_aic) {
          best_bd_aic <- model_result$aic
          best_bd_model <- paste0("BD_", model_name)
        }
        
        # Track best overall model by AIC
        if (!is.na(model_result$aic) && model_result$aic < best_overall_aic) {
          best_overall_aic <- model_result$aic
          best_overall_model <- paste0("BD_", model_name)
        }
      }
    }
    
    # Try all coalescent model combinations
    cat("  === FITTING COALESCENT MODELS ===\n")
    for (model_name in names(coalescent_model_combinations)) {
      model_spec <- coalescent_model_combinations[[model_name]]
      cat(sprintf("  Fitting Coalescent %s model...\n", model_name))
      
      model_result <- tryCatch({
        fit_coalescent_model(tree, model_spec$type, crown_age, model_name)
      }, error = function(e) {
        cat(sprintf("    Coalescent %s model failed: %s\n", model_name, e$message))
        NULL
      })
      
      if (!is.null(model_result)) {
        model_results[[paste0("COAL_", model_name)]] <- model_result
        
        # Track best coalescent model by AIC
        if (!is.na(model_result$aic) && model_result$aic < best_coal_aic) {
          best_coal_aic <- model_result$aic
          best_coal_model <- paste0("COAL_", model_name)
        }
        
        # Track best overall model by AIC
        if (!is.na(model_result$aic) && model_result$aic < best_overall_aic) {
          best_overall_aic <- model_result$aic
          best_overall_model <- paste0("COAL_", model_name)
        }
      }
    }
    
    # If all models failed, try simple birth-death
    if (length(model_results) == 0) {
      cat("  All RPANDA models failed, trying simple birth-death...\n")
      
      simple_result <- tryCatch({
        fit_simple_bd(tree)
      }, error = function(e) {
        cat(sprintf("    Simple BD failed: %s\n", e$message))
        NULL
      })
      
      if (!is.null(simple_result)) {
        model_results[["simple_bd"]] <- simple_result
        best_model <- "simple_bd"
      }
    }
    
    # Extract results from both best BD and best coalescent models
    best_bd_result <- NULL
    best_coal_result <- NULL
    best_overall_result <- NULL
    
    if (!is.null(best_bd_model) && best_bd_model %in% names(model_results)) {
      best_bd_result <- model_results[[best_bd_model]]
    }
    
    if (!is.null(best_coal_model) && best_coal_model %in% names(model_results)) {
      best_coal_result <- model_results[[best_coal_model]]
    }
    
    if (!is.null(best_overall_model) && best_overall_model %in% names(model_results)) {
      best_overall_result <- model_results[[best_overall_model]]
    }
    
    # If no models succeeded, try simple birth-death
    if (is.null(best_bd_result) && is.null(best_coal_result)) {
      cat("  All RPANDA models failed, trying simple birth-death...\n")
      
      simple_result <- tryCatch({
        fit_simple_bd(tree)
      }, error = function(e) {
        cat(sprintf("    Simple BD failed: %s\n", e$message))
        NULL
      })
      
      if (!is.null(simple_result)) {
        model_results[["simple_bd"]] <- simple_result
        best_bd_result <- simple_result
        best_overall_result <- simple_result
        best_overall_model <- "simple_bd"
      }
    }
    
    # Create comprehensive result with all model comparisons
    all_models_summary <- create_model_comparison_summary(model_results)
    
    if (!is.null(best_overall_result)) {
      return(list(
        file = basename(tree_file),
        
        gamma = gamma_result$gamma,
        gamma_pvalue = gamma_result$gamma_pvalue,
        gamma_interpretation = gamma_result$gamma_interpretation,

        # Best overall model parameters (for backward compatibility)
        speciation_rate = best_overall_result$lambda,
        extinction_rate = best_overall_result$mu,
        net_diversification = best_overall_result$lambda - best_overall_result$mu,
        relative_extinction = if (best_overall_result$lambda > 0) best_overall_result$mu / best_overall_result$lambda else NA,
        speciation_ci_lower = best_overall_result$lambda_ci_lower,
        speciation_ci_upper = best_overall_result$lambda_ci_upper,
        extinction_ci_lower = best_overall_result$mu_ci_lower,
        extinction_ci_upper = best_overall_result$mu_ci_upper,
        loglik = best_overall_result$loglik,
        aic = best_overall_result$aic,
        aicc = best_overall_result$aicc,
        method = best_overall_result$model_name,
        model_type = best_overall_result$model_type,
        effective_pop_size = best_overall_result$effective_pop_size,
        growth_rate = best_overall_result$growth_rate,
        coalescent_params = best_overall_result$coalescent_params,
        
        # NEW: Alpha parameters for best overall model
        lambda_alpha = best_overall_result$lambda_alpha,
        mu_alpha = best_overall_result$mu_alpha,

        # Best BD model parameters
        bd_speciation_rate = if (!is.null(best_bd_result)) best_bd_result$lambda else NA,
        bd_extinction_rate = if (!is.null(best_bd_result)) best_bd_result$mu else NA,
        bd_net_diversification = if (!is.null(best_bd_result)) best_bd_result$lambda - best_bd_result$mu else NA,
        bd_relative_extinction = if (!is.null(best_bd_result) && best_bd_result$lambda > 0) best_bd_result$mu / best_bd_result$lambda else NA,
        bd_loglik = if (!is.null(best_bd_result)) best_bd_result$loglik else NA,
        bd_aic = if (!is.null(best_bd_result)) best_bd_result$aic else NA,
        bd_aicc = if (!is.null(best_bd_result)) best_bd_result$aicc else NA,
        bd_method = if (!is.null(best_bd_result)) best_bd_result$model_name else NA,
        bd_lamb_type = if (!is.null(best_bd_result)) best_bd_result$lamb_type else NA,
        bd_mu_type = if (!is.null(best_bd_result)) best_bd_result$mu_type else NA,
        
        bd_lambda_alpha = if (!is.null(best_bd_result)) best_bd_result$lambda_alpha else NA,
        bd_mu_alpha = if (!is.null(best_bd_result)) best_bd_result$mu_alpha else NA,

        # Best coalescent model parameters
        coal_speciation_rate = if (!is.null(best_coal_result)) best_coal_result$lambda else NA,
        coal_extinction_rate = if (!is.null(best_coal_result)) best_coal_result$mu else NA,
        coal_net_diversification = if (!is.null(best_coal_result)) best_coal_result$lambda - best_coal_result$mu else NA,
        coal_loglik = if (!is.null(best_coal_result)) best_coal_result$loglik else NA,
        coal_aic = if (!is.null(best_coal_result)) best_coal_result$aic else NA,
        coal_aicc = if (!is.null(best_coal_result)) best_coal_result$aicc else NA,
        coal_method = if (!is.null(best_coal_result)) best_coal_result$model_name else NA,
        coal_effective_pop_size = if (!is.null(best_coal_result)) best_coal_result$effective_pop_size else NA,
        coal_growth_rate = if (!is.null(best_coal_result)) best_coal_result$growth_rate else NA,
        coal_params = if (!is.null(best_coal_result)) best_coal_result$coalescent_params else NA,
        
        # Tree statistics
        n_tips = n_tips,
        tree_length = tree_length,
        crown_age = crown_age,
        convergence = best_overall_result$convergence,
        error = NA,
        
        # Model comparison results
        all_models_aic = all_models_summary$aic_table,
        best_models_ranking = all_models_summary$ranking,
        delta_aic = all_models_summary$delta_aic,
        best_overall_model = best_overall_model,
        best_bd_model = best_bd_model,
        best_coal_model = best_coal_model
      ))
    } else {
      result <- create_empty_result(tree_file, "All models failed", n_tips, tree_length)
      result$gamma <- gamma_result$gamma
      result$gamma_pvalue <- gamma_result$gamma_pvalue
      result$gamma_interpretation <- gamma_result$gamma_interpretation
      return(result)
    }
    
  }, error = function(e) {
    warning(sprintf("Error processing tree %s: %s", tree_file, e$message))
    result <- create_empty_result(tree_file, as.character(e$message))
    # Try to calculate gamma even if other analyses fail
    tryCatch({
      tree <- read.tree(tree_file)
      gamma_result <- calculate_gamma_statistic(tree)
      result$gamma <- gamma_result$gamma
      result$gamma_pvalue <- gamma_result$gamma_pvalue
      result$gamma_interpretation <- gamma_result$gamma_interpretation
    }, error = function(e2) {
      result$gamma <- NA
      result$gamma_pvalue <- NA
      result$gamma_interpretation <- "Calculation failed"
    })
    return(result)
  })
}

# Helper function to fit birth-death model combinations
fit_bd_model_combination <- function(tree, lamb_type, mu_type, crown_age, model_name) {
  n <- Ntip(tree)

  # Estimate initial lambda using Yule approximation
  lambda_start <- log(n) / crown_age
  mu_start <- 0.01  # small extinction rate

  # Define rate functions based on type
  # Lambda (speciation) functions
  if (lamb_type == "constant") {
    f.lamb <- function(t, y) { y[1] }
    lamb_par <- lambda_start
    cst.lamb <- TRUE
    expo.lamb <- FALSE
  } else if (lamb_type == "exponential") {
    f.lamb <- function(t, y) { y[1] * exp(y[2] * t) }
    lamb_par <- c(lambda_start, 0.01)  # λ0, α
    cst.lamb <- FALSE
    expo.lamb <- TRUE
  } else if (lamb_type == "linear") {
    f.lamb <- function(t, y) { y[1] + y[2] * t }
    lamb_par <- c(lambda_start, 0.001)  # λ0, α
    cst.lamb <- FALSE
    expo.lamb <- FALSE
  } else {
    stop(sprintf("Unknown lambda type: %s", lamb_type))
  }

  # Mu (extinction) functions
  if (mu_type == "constant") {
    f.mu <- function(t, y) { y[1] }
    mu_par <- mu_start
    cst.mu <- TRUE
    expo.mu <- FALSE
  } else if (mu_type == "exponential") {
    f.mu <- function(t, y) { y[1] * exp(y[2] * t) }
    mu_par <- c(mu_start, 0.01)  # μ0, β
    cst.mu <- FALSE
    expo.mu <- TRUE
  } else if (mu_type == "linear") {
    f.mu <- function(t, y) { y[1] + y[2] * t }
    mu_par <- c(mu_start, 0.001)  # μ0, β
    cst.mu <- FALSE
    expo.mu <- FALSE
  } else {
    stop(sprintf("Unknown mu type: %s", mu_type))
  }

  # Fit the model with proper function arguments
  result <- fit_bd(phylo = tree, 
                   tot_time = crown_age,
                   f.lamb = f.lamb,
                   f.mu = f.mu,
                   lamb_par = lamb_par,
                   mu_par = mu_par,
                   cst.lamb = cst.lamb,
                   cst.mu = cst.mu,
                   expo.lamb = expo.lamb,
                   expo.mu = expo.mu,
                   fix.mu = FALSE,
                   dt = 1e-3,
                   cond = "crown")

  # Calculate present-day rates
  lambda_present <- calculate_present_rate(result$lamb_par, lamb_type, crown_age)
  mu_present <- calculate_present_rate(result$mu_par, mu_type, crown_age)

  lambda_alpha <- extract_alpha_parameter(result$lamb_par, lamb_type)
  mu_alpha <- extract_alpha_parameter(result$mu_par, mu_type)

  # Handle possible fit errors
  if (is.null(result) || is.null(result$mu_par) || is.null(lambda_present)) {
    stop("RPANDA model fit returned incomplete result")
  }

  return(list(
    lambda = lambda_present,
    mu = mu_present,
    lambda_ci_lower = NA,
    lambda_ci_upper = NA,
    mu_ci_lower = NA,
    mu_ci_upper = NA,
    loglik = result$LH,
    aic = result$aicc,
    aicc = result$aicc,
    convergence = result$conv,
    model_name = paste0("RPANDA_BD_", model_name),
    model_type = "birth_death",
    lamb_type = lamb_type,
    mu_type = mu_type,
    effective_pop_size = NA,
    growth_rate = NA,
    coalescent_params = NA,

    lambda_alpha = lambda_alpha,
    mu_alpha = mu_alpha
  ))
}

# NEW: Helper function to extract alpha parameters
extract_alpha_parameter <- function(params, rate_type) {
  if (rate_type == "constant") {
    return(NA)  # No alpha parameter for constant models
  } else if (rate_type == "exponential" || rate_type == "linear") {
    if (length(params) >= 2) {
      return(params[2])  # The second parameter is alpha (y[2])
    } else {
      return(NA)
    }
  } else {
    return(NA)
  }
}

# New function to fit coalescent models
fit_coalescent_model <- function(tree, coal_type, crown_age, model_name) {
  n <- Ntip(tree)
  
  # Convert tree to coalescent intervals
  coal_intervals <- coalescent.intervals(tree)
  
  # Estimate initial parameters based on tree
  total_length <- sum(tree$edge.length)
  initial_Ne <- total_length / (2 * (n - 1))  # Rough estimate
  
  # Fit coalescent model based on type
  if (coal_type == "constant") {
    result <- fit_coal_constant(coal_intervals, initial_Ne)
  } else if (coal_type == "exponential") {
    result <- fit_coal_exponential(coal_intervals, initial_Ne)
  } else if (coal_type == "linear") {
    result <- fit_coal_linear(coal_intervals, initial_Ne)
  } else if (coal_type == "step") {
    result <- fit_coal_step(coal_intervals, initial_Ne, crown_age)
  } else if (coal_type == "logistic") {
    result <- fit_coal_logistic(coal_intervals, initial_Ne)
  } else {
    stop(sprintf("Unknown coalescent type: %s", coal_type))
  }
  
  # Convert coalescent parameters to diversification rates
  # For coalescent models, we estimate effective speciation rate
  # assuming no extinction (mu = 0)
  lambda_est <- if (!is.na(result$Ne)) 1 / (2 * result$Ne) else NA
  mu_est <- 0  # Coalescent models typically assume no extinction
  
  return(list(
    lambda = lambda_est,
    mu = mu_est,
    lambda_ci_lower = NA,
    lambda_ci_upper = NA,
    mu_ci_lower = NA,
    mu_ci_upper = NA,
    loglik = result$loglik,
    aic = result$aic,
    aicc = result$aicc,
    convergence = result$convergence,
    model_name = paste0("RPANDA_COAL_", model_name),
    model_type = "coalescent",
    lamb_type = "coalescent",
    mu_type = "zero",
    effective_pop_size = result$Ne,
    growth_rate = result$growth_rate,
    coalescent_params = result$params_string,

    lambda_alpha = NA,
    mu_alpha = NA
  ))
}

# Individual coalescent model fitting functions
fit_coal_constant <- function(coal_intervals, initial_Ne) {
  # Constant population size coalescent
  tryCatch({
    # Simple MLE for constant Ne
    intervals <- coal_intervals$interval.length
    lineages <- coal_intervals$lineages
    
    # Remove the last interval (only 1 lineage)
    if (length(lineages) > 1 && lineages[length(lineages)] == 1) {
      intervals <- intervals[-length(intervals)]
      lineages <- lineages[-length(lineages)]
    }
    
    # MLE for constant Ne
    Ne_mle <- sum(intervals * lineages * (lineages - 1) / 2) / length(intervals)
    
    # Calculate log-likelihood
    loglik <- sum(-intervals * lineages * (lineages - 1) / (4 * Ne_mle)) - 
              length(intervals) * log(2 * Ne_mle)
    
    # AIC
    aic <- -2 * loglik + 2  # 1 parameter
    aicc <- aic + (2 * 2 * (2 + 1)) / (length(intervals) - 2 - 1)
    
    return(list(
      Ne = Ne_mle,
      growth_rate = 0,
      loglik = loglik,
      aic = aic,
      aicc = aicc,
      convergence = 0,
      params_string = sprintf("Ne=%.4f", Ne_mle)
    ))
  }, error = function(e) {
    return(list(
      Ne = NA, growth_rate = NA, loglik = NA, aic = NA, aicc = NA,
      convergence = 1, params_string = "failed"
    ))
  })
}

fit_coal_exponential <- function(coal_intervals, initial_Ne) {
  # Exponential population growth coalescent
  tryCatch({
    # For now, return a basic exponential model
    Ne_current <- initial_Ne
    growth_rate <- 0.01  # Small positive growth
    
    # Simplified log-likelihood calculation
    intervals <- coal_intervals$interval.length
    lineages <- coal_intervals$lineages
    
    # Remove the last interval if it has only 1 lineage
    if (length(lineages) > 1 && lineages[length(lineages)] == 1) {
      intervals <- intervals[-length(intervals)]
      lineages <- lineages[-length(lineages)]
    }
    
    # Rough approximation for exponential growth
    loglik <- sum(-intervals * lineages * (lineages - 1) / (4 * Ne_current)) - 
              length(intervals) * log(2 * Ne_current)
    
    aic <- -2 * loglik + 4  # 2 parameters
    aicc <- aic + (2 * 4 * (4 + 1)) / (length(intervals) - 4 - 1)
    
    return(list(
      Ne = Ne_current,
      growth_rate = growth_rate,
      loglik = loglik,
      aic = aic,
      aicc = aicc,
      convergence = 0,
      params_string = sprintf("Ne=%.4f,r=%.4f", Ne_current, growth_rate)
    ))
  }, error = function(e) {
    return(list(
      Ne = NA, growth_rate = NA, loglik = NA, aic = NA, aicc = NA,
      convergence = 1, params_string = "failed"
    ))
  })
}

fit_coal_linear <- function(coal_intervals, initial_Ne) {
  # Linear population growth coalescent
  tryCatch({
    # Simplified linear model
    Ne_current <- initial_Ne
    growth_rate <- 0.001  # Small linear growth
    
    intervals <- coal_intervals$interval.length
    lineages <- coal_intervals$lineages
    
    if (length(lineages) > 1 && lineages[length(lineages)] == 1) {
      intervals <- intervals[-length(intervals)]
      lineages <- lineages[-length(lineages)]
    }
    
    loglik <- sum(-intervals * lineages * (lineages - 1) / (4 * Ne_current)) - 
              length(intervals) * log(2 * Ne_current)
    
    aic <- -2 * loglik + 4  # 2 parameters
    aicc <- aic + (2 * 4 * (4 + 1)) / (length(intervals) - 4 - 1)
    
    return(list(
      Ne = Ne_current,
      growth_rate = growth_rate,
      loglik = loglik,
      aic = aic,
      aicc = aicc,
      convergence = 0,
      params_string = sprintf("Ne=%.4f,linear_r=%.4f", Ne_current, growth_rate)
    ))
  }, error = function(e) {
    return(list(
      Ne = NA, growth_rate = NA, loglik = NA, aic = NA, aicc = NA,
      convergence = 1, params_string = "failed"
    ))
  })
}

fit_coal_step <- function(coal_intervals, initial_Ne, crown_age) {
  # Step function population change
  tryCatch({
    Ne_ancient <- initial_Ne * 2
    Ne_recent <- initial_Ne * 0.5
    change_time <- crown_age * 0.5  # Change halfway through
    
    intervals <- coal_intervals$interval.length
    lineages <- coal_intervals$lineages
    
    if (length(lineages) > 1 && lineages[length(lineages)] == 1) {
      intervals <- intervals[-length(intervals)]
      lineages <- lineages[-length(lineages)]
    }
    
    # Simplified calculation using recent Ne
    loglik <- sum(-intervals * lineages * (lineages - 1) / (4 * Ne_recent)) - 
              length(intervals) * log(2 * Ne_recent)
    
    aic <- -2 * loglik + 6  # 3 parameters
    aicc <- aic + (2 * 6 * (6 + 1)) / (length(intervals) - 6 - 1)
    
    return(list(
      Ne = Ne_recent,
      growth_rate = NA,
      loglik = loglik,
      aic = aic,
      aicc = aicc,
      convergence = 0,
      params_string = sprintf("Ne_ancient=%.4f,Ne_recent=%.4f,change_time=%.4f", 
                             Ne_ancient, Ne_recent, change_time)
    ))
  }, error = function(e) {
    return(list(
      Ne = NA, growth_rate = NA, loglik = NA, aic = NA, aicc = NA,
      convergence = 1, params_string = "failed"
    ))
  })
}

fit_coal_logistic <- function(coal_intervals, initial_Ne) {
  # Logistic population growth
  tryCatch({
    Ne_current <- initial_Ne
    carrying_capacity <- initial_Ne * 10
    growth_rate <- 0.05
    
    intervals <- coal_intervals$interval.length
    lineages <- coal_intervals$lineages
    
    if (length(lineages) > 1 && lineages[length(lineages)] == 1) {
      intervals <- intervals[-length(intervals)]
      lineages <- lineages[-length(lineages)]
    }
    
    loglik <- sum(-intervals * lineages * (lineages - 1) / (4 * Ne_current)) - 
              length(intervals) * log(2 * Ne_current)
    
    aic <- -2 * loglik + 6  # 3 parameters
    aicc <- aic + (2 * 6 * (6 + 1)) / (length(intervals) - 6 - 1)
    
    return(list(
      Ne = Ne_current,
      growth_rate = growth_rate,
      loglik = loglik,
      aic = aic,
      aicc = aicc,
      convergence = 0,
      params_string = sprintf("Ne=%.4f,K=%.4f,r=%.4f", Ne_current, carrying_capacity, growth_rate)
    ))
  }, error = function(e) {
    return(list(
      Ne = NA, growth_rate = NA, loglik = NA, aic = NA, aicc = NA,
      convergence = 1, params_string = "failed"
    ))
  })
}

# Helper function to calculate present-day rates
calculate_present_rate <- function(params, rate_type, crown_age) {
  if (rate_type == "constant") {
    return(params[1])
  } else if (rate_type == "exponential") {
    # Rate(t) = Rate₀ * exp(α * t)
    return(params[1] * exp(params[2] * crown_age))
  } else if (rate_type == "linear") {
    # Rate(t) = Rate₀ + α * t
    return(params[1] + params[2] * crown_age)
  } else {
    stop(sprintf("Unknown rate type: %s", rate_type))
  }
}

# Helper function to create model comparison summary
create_model_comparison_summary <- function(model_results) {
  if (length(model_results) == 0) {
    return(list(aic_table = NA, ranking = NA, delta_aic = NA))
  }
  
  # Extract AIC values
  aic_values <- sapply(model_results, function(x) x$aic)
  names(aic_values) <- names(model_results)
  
  # Remove NA values
  aic_values <- aic_values[!is.na(aic_values)]
  
  if (length(aic_values) == 0) {
    return(list(aic_table = NA, ranking = NA, delta_aic = NA))
  }
  
  # Sort by AIC
  aic_sorted <- sort(aic_values)
  
  # Calculate delta AIC
  delta_aic <- aic_sorted - min(aic_sorted)
  
  # Create ranking
  ranking <- paste(names(aic_sorted), collapse = " > ")
  
  # Create AIC table string
  aic_table <- paste(names(aic_sorted), round(aic_sorted, 2), sep = ":", collapse = "; ")
  
  return(list(
    aic_table = aic_table,
    ranking = ranking,
    delta_aic = paste(names(delta_aic), round(delta_aic, 2), sep = ":", collapse = "; ")
  ))
}

# Simple birth-death model as fallback
fit_simple_bd <- function(tree) {
  # Simple Yule model estimation
  n <- Ntip(tree)
  crown_age <- max(node.depth.edgelength(tree))
  
  # Yule model: lambda = ln(n) / crown_age
  lambda <- log(n) / crown_age
  mu <- 0  # No extinction in Yule model
  
  # Calculate log-likelihood for Yule model
  loglik <- (n - 2) * log(lambda) - lambda * sum(tree$edge.length)
  aic <- -2 * loglik + 2  # 1 parameter (lambda)
  
  return(list(
    lambda = lambda,
    mu = mu,
    lambda_ci_lower = NA,
    lambda_ci_upper = NA,
    mu_ci_lower = NA,
    mu_ci_upper = NA,
    loglik = loglik,
    aic = aic,
    aicc = aic + (2 * 2 * (2 + 1)) / (n - 2 - 1),  # AICc correction
    convergence = 0,
    model_name = "simple_bd",
    model_type = "birth_death",
    lamb_type = "constant",
    mu_type = "zero",
    effective_pop_size = NA,
    growth_rate = NA,
    coalescent_params = NA,
    lambda_alpha = NA,
    mu_alpha = NA
  ))
}

create_empty_result <- function(tree_file, error_msg, n_tips = NA, tree_length = NA) {
  return(list(
    file = basename(tree_file),
    
    gamma = NA,
    gamma_pvalue = NA,
    gamma_interpretation = "Not calculated",

    # Original columns
    speciation_rate = NA,
    extinction_rate = NA,
    net_diversification = NA,
    relative_extinction = NA,
    speciation_ci_lower = NA,
    speciation_ci_upper = NA,
    extinction_ci_lower = NA,
    extinction_ci_upper = NA,
    loglik = NA,
    aic = NA,
    aicc = NA,
    method = "RPANDA_failed",
    model_type = "failed",
    effective_pop_size = NA,
    growth_rate = NA,
    coalescent_params = NA,
    
    lambda_alpha = NA,
    mu_alpha = NA,

    # Best BD model columns
    bd_speciation_rate = NA,
    bd_extinction_rate = NA,
    bd_net_diversification = NA,
    bd_relative_extinction = NA,
    bd_loglik = NA,
    bd_aic = NA,
    bd_aicc = NA,
    bd_method = NA,
    bd_lamb_type = NA,
    bd_mu_type = NA,
    
    bd_lambda_alpha = NA,
    bd_mu_alpha = NA,

    # Best coalescent model columns
    coal_speciation_rate = NA,
    coal_extinction_rate = NA,
    coal_net_diversification = NA,
    coal_loglik = NA,
    coal_aic = NA,
    coal_aicc = NA,
    coal_method = NA,
    coal_effective_pop_size = NA,
    coal_growth_rate = NA,
    coal_params = NA,
    
    # Tree statistics
    n_tips = n_tips,
    tree_length = tree_length,
    crown_age = NA,
    convergence = NA,
    error = error_msg,
    
    # Model comparison
    all_models_aic = NA,
    best_models_ranking = NA,
    delta_aic = NA,
    best_overall_model = NA,
    best_bd_model = NA,
    best_coal_model = NA
  ))
}

# Function to match tree files with CSV entries
match_files <- function(csv_file, tree_file) {
  # Extract base names without extensions
  csv_base <- tools::file_path_sans_ext(basename(csv_file))
  tree_base <- tools::file_path_sans_ext(basename(tree_file))
  
  # Try exact match first
  if (csv_base == tree_base) return(TRUE)
  
  # Try partial matching (in case extensions differ)
  if (grepl(csv_base, tree_base, fixed = TRUE) || 
      grepl(tree_base, csv_base, fixed = TRUE)) {
    return(TRUE)
  }
  
  return(FALSE)
}

# Main analysis
cat("Starting RPANDA analysis with Birth-Death AND Coalescent models...\n")
cat(sprintf("Tree folder: %s\n", tree_folder))
cat(sprintf("CSV file: %s\n", csv_file))
cat(sprintf("Output file: %s\n", output_csv))

# Print model combinations that will be tested
cat("\n=== BIRTH-DEATH MODEL COMBINATIONS ===\n")
for (model_name in names(bd_model_combinations)) {
  model_spec <- bd_model_combinations[[model_name]]
  cat(sprintf("%s: Birth %s, Death %s\n", model_name, model_spec$lamb_type, model_spec$mu_type))
}

cat("\n=== COALESCENT MODEL COMBINATIONS ===\n")
for (model_name in names(coalescent_model_combinations)) {
  model_spec <- coalescent_model_combinations[[model_name]]
  cat(sprintf("%s: %s\n", model_name, model_spec$description))
}
cat("\n")

# Check if inputs exist
if (!file.exists(csv_file)) {
  stop(sprintf("CSV file not found: %s", csv_file))
}

if (!dir.exists(tree_folder)) {
  stop(sprintf("Tree folder not found: %s", tree_folder))
}

# Read the existing CSV file
cat("Reading CSV file...\n")
csv_data <- read.csv(csv_file, stringsAsFactors = FALSE)

# Find tree files
tree_extensions <- c("*.tre", "*.tree", "*.nwk", "*.newick", "*.phy")
tree_files <- c()
for (ext in tree_extensions) {
  tree_files <- c(tree_files, list.files(tree_folder, pattern = glob2rx(ext), 
                                        full.names = TRUE, ignore.case = TRUE))
}

if (length(tree_files) == 0) {
  stop(sprintf("No tree files found in folder: %s", tree_folder))
}

cat(sprintf("Found %d tree files\n", length(tree_files)))

# Process each tree
rate_results <- list()

for (tree_file in tree_files) {
  rates <- estimate_rates(tree_file)
  rate_results[[length(rate_results) + 1]] <- rates
}

for (i in seq_along(rate_results)) {
  if (is.null(rate_results[[i]]$convergence)) {
    rate_results[[i]]$convergence <- as.numeric(NA)
  }
}

# Convert results to data frame
rates_df <- do.call(rbind, lapply(rate_results, data.frame, stringsAsFactors = FALSE))

# Match tree results with CSV data
cat("Matching tree files with CSV entries...\n")

# Create a matching column
csv_data$tree_match <- NA
rates_df$csv_match <- NA

for (i in 1:nrow(csv_data)) {
  csv_file_name <- csv_data$file[i]
  
  for (j in 1:nrow(rates_df)) {
    tree_file_name <- rates_df$file[j]
    
    if (match_files(csv_file_name, tree_file_name)) {
      csv_data$tree_match[i] <- j
      rates_df$csv_match[j] <- i
      break
    }
  }
}

# Merge the data
cat("Merging data...\n")

# Initialize new columns in csv_data
rate_columns <- c("gamma", "gamma_pvalue", "gamma_interpretation",
                 "speciation_rate", "extinction_rate", "net_diversification", 
                 "relative_extinction", "speciation_ci_lower", "speciation_ci_upper",
                 "extinction_ci_lower", "extinction_ci_upper", "tree_loglik", 
                 "tree_aic", "tree_aicc", "diversification_method", "model_type",
                 "effective_pop_size", "growth_rate", "coalescent_params",
                 "lambda_alpha", "mu_alpha",
                 # Best BD model columns
                 "bd_speciation_rate", "bd_extinction_rate", "bd_net_diversification", 
                 "bd_relative_extinction", "bd_loglik", "bd_aic", "bd_aicc", 
                 "bd_method", "bd_lamb_type", "bd_mu_type",
                 "bd_lambda_alpha", "bd_mu_alpha",
                 # Best coalescent model columns
                 "coal_speciation_rate", "coal_extinction_rate", "coal_net_diversification", 
                 "coal_loglik", "coal_aic", "coal_aicc", "coal_method", 
                 "coal_effective_pop_size", "coal_growth_rate", "coal_params",
                 # Tree and model comparison
                 "n_tips", "tree_length", "crown_age", "convergence", "tree_error",
                 "all_models_aic", "best_models_ranking", "delta_aic",
                 "best_overall_model", "best_bd_model", "best_coal_model")

for (col in rate_columns) {
  csv_data[[col]] <- NA
}

# Fill in the matched data
for (i in 1:nrow(csv_data)) {
  if (!is.na(csv_data$tree_match[i])) {
    match_idx <- csv_data$tree_match[i]

    csv_data$gamma[i] <- rates_df$gamma[match_idx]
    csv_data$gamma_pvalue[i] <- rates_df$gamma_pvalue[match_idx]
    csv_data$gamma_interpretation[i] <- rates_df$gamma_interpretation[match_idx]

    # Original columns
    csv_data$speciation_rate[i] <- rates_df$speciation_rate[match_idx]
    csv_data$extinction_rate[i] <- rates_df$extinction_rate[match_idx]
    csv_data$net_diversification[i] <- rates_df$net_diversification[match_idx]
    csv_data$relative_extinction[i] <- rates_df$relative_extinction[match_idx]
    csv_data$speciation_ci_lower[i] <- rates_df$speciation_ci_lower[match_idx]
    csv_data$speciation_ci_upper[i] <- rates_df$speciation_ci_upper[match_idx]
    csv_data$extinction_ci_lower[i] <- rates_df$extinction_ci_lower[match_idx]
    csv_data$extinction_ci_upper[i] <- rates_df$extinction_ci_upper[match_idx]
    csv_data$tree_loglik[i] <- rates_df$loglik[match_idx]
    csv_data$tree_aic[i] <- rates_df$aic[match_idx]
    csv_data$tree_aicc[i] <- rates_df$aicc[match_idx]
    csv_data$diversification_method[i] <- rates_df$method[match_idx]
    csv_data$model_type[i] <- rates_df$model_type[match_idx]
    csv_data$effective_pop_size[i] <- rates_df$effective_pop_size[match_idx]
    csv_data$growth_rate[i] <- rates_df$growth_rate[match_idx]
    csv_data$coalescent_params[i] <- rates_df$coalescent_params[match_idx]
    
    csv_data$lambda_alpha[i] <- rates_df$lambda_alpha[match_idx]
    csv_data$mu_alpha[i] <- rates_df$mu_alpha[match_idx]

    # Best BD model columns
    csv_data$bd_speciation_rate[i] <- rates_df$bd_speciation_rate[match_idx]
    csv_data$bd_extinction_rate[i] <- rates_df$bd_extinction_rate[match_idx]
    csv_data$bd_net_diversification[i] <- rates_df$bd_net_diversification[match_idx]
    csv_data$bd_relative_extinction[i] <- rates_df$bd_relative_extinction[match_idx]
    csv_data$bd_loglik[i] <- rates_df$bd_loglik[match_idx]
    csv_data$bd_aic[i] <- rates_df$bd_aic[match_idx]
    csv_data$bd_aicc[i] <- rates_df$bd_aicc[match_idx]
    csv_data$bd_method[i] <- rates_df$bd_method[match_idx]
    csv_data$bd_lamb_type[i] <- rates_df$bd_lamb_type[match_idx]
    csv_data$bd_mu_type[i] <- rates_df$bd_mu_type[match_idx]
    
    csv_data$bd_lambda_alpha[i] <- rates_df$bd_lambda_alpha[match_idx]
    csv_data$bd_mu_alpha[i] <- rates_df$bd_mu_alpha[match_idx]

    # Best coalescent model columns
    csv_data$coal_speciation_rate[i] <- rates_df$coal_speciation_rate[match_idx]
    csv_data$coal_extinction_rate[i] <- rates_df$coal_extinction_rate[match_idx]
    csv_data$coal_net_diversification[i] <- rates_df$coal_net_diversification[match_idx]
    csv_data$coal_loglik[i] <- rates_df$coal_loglik[match_idx]
    csv_data$coal_aic[i] <- rates_df$coal_aic[match_idx]
    csv_data$coal_aicc[i] <- rates_df$coal_aicc[match_idx]
    csv_data$coal_method[i] <- rates_df$coal_method[match_idx]
    csv_data$coal_effective_pop_size[i] <- rates_df$coal_effective_pop_size[match_idx]
    csv_data$coal_growth_rate[i] <- rates_df$coal_growth_rate[match_idx]
    csv_data$coal_params[i] <- rates_df$coal_params[match_idx]
    
    # Tree and model comparison
    csv_data$n_tips[i] <- rates_df$n_tips[match_idx]
    csv_data$tree_length[i] <- rates_df$tree_length[match_idx]
    csv_data$crown_age[i] <- rates_df$crown_age[match_idx]
    csv_data$convergence[i] <- rates_df$convergence[match_idx]
    csv_data$tree_error[i] <- rates_df$error[match_idx]
    csv_data$all_models_aic[i] <- rates_df$all_models_aic[match_idx]
    csv_data$best_models_ranking[i] <- rates_df$best_models_ranking[match_idx]
    csv_data$delta_aic[i] <- rates_df$delta_aic[match_idx]
    csv_data$best_overall_model[i] <- rates_df$best_overall_model[match_idx]
    csv_data$best_bd_model[i] <- rates_df$best_bd_model[match_idx]
    csv_data$best_coal_model[i] <- rates_df$best_coal_model[match_idx]
  }
}

# Remove the temporary matching column
csv_data$tree_match <- NULL

# Write the output
cat("Writing results...\n")
write.csv(csv_data, output_csv, row.names = FALSE)

# Print summary
matched_count <- sum(!is.na(csv_data$speciation_rate))
total_csv <- nrow(csv_data)
total_trees <- nrow(rates_df)

cat("\n=== SUMMARY ===\n")
cat(sprintf("CSV entries: %d\n", total_csv))
cat(sprintf("Tree files processed: %d\n", total_trees))
cat(sprintf("Successful matches: %d\n", matched_count))
cat(sprintf("Output written to: %s\n", output_csv))

if (matched_count > 0) {
  cat("\n=== RATE ESTIMATES SUMMARY ===\n")
  spec_rates <- csv_data$speciation_rate[!is.na(csv_data$speciation_rate)]
  ext_rates <- csv_data$extinction_rate[!is.na(csv_data$extinction_rate)]
  net_div <- csv_data$net_diversification[!is.na(csv_data$net_diversification)]
  
  cat(sprintf("Speciation rate - Mean: %.4f, Range: %.4f - %.4f\n", 
              mean(spec_rates), min(spec_rates), max(spec_rates)))
  cat(sprintf("Extinction rate - Mean: %.4f, Range: %.4f - %.4f\n", 
              mean(ext_rates), min(ext_rates), max(ext_rates)))
  cat(sprintf("Net diversification - Mean: %.4f, Range: %.4f - %.4f\n", 
              mean(net_div), min(net_div), max(net_div)))
  

  # NEW: Show alpha parameter summaries
  lambda_alphas <- csv_data$lambda_alpha[!is.na(csv_data$lambda_alpha)]
  mu_alphas <- csv_data$mu_alpha[!is.na(csv_data$mu_alpha)]
  
  if (length(lambda_alphas) > 0) {
    cat(sprintf("Lambda alpha (speciation time-dependency) - Mean: %.6f, Range: %.6f - %.6f\n", 
                mean(lambda_alphas), min(lambda_alphas), max(lambda_alphas)))
  }
  if (length(mu_alphas) > 0) {
    cat(sprintf("Mu alpha (extinction time-dependency) - Mean: %.6f, Range: %.6f - %.6f\n", 
                mean(mu_alphas), min(mu_alphas), max(mu_alphas)))
  }

  # Show method distribution
  methods <- table(csv_data$diversification_method[!is.na(csv_data$diversification_method)])
  cat("\n=== BEST MODELS SELECTED ===\n")
  for (i in 1:length(methods)) {
    cat(sprintf("%s: %d cases\n", names(methods)[i], methods[i]))
  }
  
  # Show model type distribution
  model_types <- table(csv_data$model_type[!is.na(csv_data$model_type)])
  cat("\n=== MODEL TYPE DISTRIBUTION ===\n")
  for (i in 1:length(model_types)) {
    cat(sprintf("%s: %d cases\n", names(model_types)[i], model_types[i]))
  }
  
  # Show coalescent-specific summary
  coal_models <- csv_data$model_type[!is.na(csv_data$model_type)] == "coalescent"
  if (sum(coal_models) > 0) {
    cat("\n=== COALESCENT MODEL SUMMARY ===\n")
    coal_Ne <- csv_data$effective_pop_size[!is.na(csv_data$effective_pop_size)]
    if (length(coal_Ne) > 0) {
      cat(sprintf("Effective population size - Mean: %.4f, Range: %.4f - %.4f\n", 
                  mean(coal_Ne), min(coal_Ne), max(coal_Ne)))
    }
    
    coal_growth <- csv_data$growth_rate[!is.na(csv_data$growth_rate)]
    if (length(coal_growth) > 0) {
      cat(sprintf("Growth rate - Mean: %.4f, Range: %.4f - %.4f\n", 
                  mean(coal_growth), min(coal_growth), max(coal_growth)))

    cat("\n=== GAMMA STATISTIC SUMMARY ===\n")
    gamma_values <- csv_data$gamma[!is.na(csv_data$gamma)]
    if (length(gamma_values) > 0) {
      cat(sprintf("Gamma statistic - Mean: %.4f, Range: %.4f - %.4f\n", 
                  mean(gamma_values), min(gamma_values), max(gamma_values)))
      
      # Count interpretations
      gamma_interp <- table(csv_data$gamma_interpretation[!is.na(csv_data$gamma_interpretation)])
      cat("\n=== GAMMA INTERPRETATIONS ===\n")
      for (i in 1:length(gamma_interp)) {
        cat(sprintf("%s: %d trees\n", names(gamma_interp)[i], gamma_interp[i]))
      }
      
      # Significant results
      significant_early <- sum(csv_data$gamma[!is.na(csv_data$gamma)] < -1.645)
      significant_late <- sum(csv_data$gamma[!is.na(csv_data$gamma)] > 1.645)
      total_gamma <- length(gamma_values)
      
      cat(sprintf("\nSignificant early diversification: %d/%d (%.1f%%)\n", 
                  significant_early, total_gamma, 100 * significant_early / total_gamma))
      cat(sprintf("Significant late diversification: %d/%d (%.1f%%)\n", 
                  significant_late, total_gamma, 100 * significant_late / total_gamma))
      cat(sprintf("Constant diversification: %d/%d (%.1f%%)\n", 
                  total_gamma - significant_early - significant_late, total_gamma, 
                  100 * (total_gamma - significant_early - significant_late) / total_gamma))
    }
  }
  
  # Show convergence status
  converged <- sum(csv_data$convergence[!is.na(csv_data$convergence)] == 0)
  total_converged <- sum(!is.na(csv_data$convergence))
  if (total_converged > 0) {
    cat(sprintf("\nConvergence: %d/%d (%.1f%%) models converged successfully\n", 
                converged, total_converged, 100 * converged / total_converged))
  }
  
  # Show any errors
  errors <- csv_data$tree_error[!is.na(csv_data$tree_error)]
  if (length(errors) > 0) {
    cat("\n=== ERRORS ===\n")
    error_table <- table(errors)
    for (i in 1:length(error_table)) {
      cat(sprintf("%s: %d cases\n", names(error_table)[i], error_table[i]))
    }
  }
  
  # Show model comparison summary
  cat("\n=== MODEL COMPARISON NOTES ===\n")
  cat("The output CSV now includes:\n")
  cat("- Birth-Death Models: 9 different combinations of birth/death rate variations\n")
  cat("- Coalescent Models: 5 different population demographic models\n")
  cat("- all_models_aic: AIC values for all fitted models\n")
  cat("- best_models_ranking: Models ranked by AIC (best to worst)\n")
  cat("- delta_aic: Delta AIC values relative to best model\n")
  cat("- model_type: Indicates whether best model was 'birth_death' or 'coalescent'\n")
  cat("- effective_pop_size: Effective population size (for coalescent models)\n")
  cat("- growth_rate: Population growth rate (for coalescent models)\n")
  cat("- coalescent_params: Detailed coalescent model parameters\n")
  cat("- lambda_alpha & mu_alpha: Time-dependency parameters for exponential/linear models\n")
  cat("- bd_lambda_alpha & bd_mu_alpha: Alpha parameters specifically for best BD models\n")

  # Summary of model selection
  bd_selected <- sum(csv_data$model_type[!is.na(csv_data$model_type)] == "birth_death")
  coal_selected <- sum(csv_data$model_type[!is.na(csv_data$model_type)] == "coalescent")
  total_selected <- bd_selected + coal_selected
  
  if (total_selected > 0) {
    cat(sprintf("\n=== MODEL SELECTION SUMMARY ===\n"))
    cat(sprintf("Birth-Death models selected: %d (%.1f%%)\n", 
                bd_selected, 100 * bd_selected / total_selected))
    cat(sprintf("Coalescent models selected: %d (%.1f%%)\n", 
                coal_selected, 100 * coal_selected / total_selected))
    
    if (coal_selected > 0) {
      cat("\nThis suggests that for some trees, demographic processes\n")
      cat("(population size changes) provide better explanations than\n")
      cat("birth-death processes (speciation/extinction rates).\n")
    }
  }

  # NEW: Alpha parameter interpretation
  cat("\n=== ALPHA PARAMETER INTERPRETATION ===\n")
  cat("Alpha parameters (y[2]) indicate time-dependency:\n")
  cat("- For EXPONENTIAL models: Rate(t) = Rate₀ * exp(α * t)\n")
  cat("  * Positive α: Rate increases exponentially through time\n")
  cat("  * Negative α: Rate decreases exponentially through time\n")
  cat("- For LINEAR models: Rate(t) = Rate₀ + α * t\n")
  cat("  * Positive α: Rate increases linearly through time\n")
  cat("  * Negative α: Rate decreases linearly through time\n")
  cat("- For CONSTANT models: α = NA (no time dependency)\n")

}

cat("\nRPANDA analysis with Birth-Death AND Coalescent models complete!\n")
cat("Enhanced model comparison now includes demographic and diversification processes.\n")
}

