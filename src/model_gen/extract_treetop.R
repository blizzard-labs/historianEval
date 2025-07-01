#!/usr/bin/env Rscript

# RPANDA Analysis Script
# Estimates speciation and extinction rates from phylogenetic trees
# and adds them to existing parameter CSV files
# Enhanced to fit all combinations of birth and death models

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

# Define all model combinations
model_combinations <- list(
  "BCSTDCST" = list(lamb_type = "constant", mu_type = "constant"),     # Birth Constant, Death Constant
  "BEXPCDST" = list(lamb_type = "exponential", mu_type = "constant"),  # Birth Exponential, Death Constant
  "BLINCDST" = list(lamb_type = "linear", mu_type = "constant"),       # Birth Linear, Death Constant
  "BCSTDEXP" = list(lamb_type = "constant", mu_type = "exponential"),  # Birth Constant, Death Exponential
  "BEXPDEXP" = list(lamb_type = "exponential", mu_type = "exponential"), # Birth Exponential, Death Exponential
  "BLINDEXP" = list(lamb_type = "linear", mu_type = "exponential"),    # Birth Linear, Death Exponential
  "BCSTDLIN" = list(lamb_type = "constant", mu_type = "linear"),       # Birth Constant, Death Linear
  "BEXPDLIN" = list(lamb_type = "exponential", mu_type = "linear"),    # Birth Exponential, Death Linear
  "BLINDLIN" = list(lamb_type = "linear", mu_type = "linear")          # Birth Linear, Death Linear
)

# Function to estimate diversification rates using RPANDA
estimate_rates <- function(tree_file) {
  cat(sprintf("Processing tree: %s\n", basename(tree_file)))
  
  tryCatch({
    # Read the tree
    tree <- read.tree(tree_file)
    
    # Basic tree statistics
    n_tips <- Ntip(tree)
    tree_length <- sum(tree$edge.length)
    
    if (n_tips < 3) {
      warning(sprintf("Tree has too few tips: %s", tree_file))
      return(create_empty_result(tree_file, "Too few tips", n_tips, tree_length))
    }
    
    # RPANDA requires ultrametric trees
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
    
    # Get the crown age
    crown_age <- max(node.depth.edgelength(tree))
    
    if (crown_age <= 0) {
      warning(sprintf("Invalid crown age in tree: %s", tree_file))
      return(create_empty_result(tree_file, "Invalid crown age", n_tips, tree_length))
    }
    
    # Store results for different models
    model_results <- list()
    best_model <- NULL
    best_aic <- Inf
    
    # Try all model combinations
    for (model_name in names(model_combinations)) {
      model_spec <- model_combinations[[model_name]]
      cat(sprintf("  Fitting %s model...\n", model_name))
      
      model_result <- tryCatch({
        fit_model_combination(tree, model_spec$lamb_type, model_spec$mu_type, crown_age, model_name)
      }, error = function(e) {
        cat(sprintf("    %s model failed: %s\n", model_name, e$message))
        NULL
      })
      
      if (!is.null(model_result)) {
        model_results[[model_name]] <- model_result
        
        # Track best model by AIC
        if (!is.na(model_result$aic) && model_result$aic < best_aic) {
          best_aic <- model_result$aic
          best_model <- model_name
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
    
    # Extract results from best model
    if (!is.null(best_model) && best_model %in% names(model_results)) {
      result <- model_results[[best_model]]
      
      # Create comprehensive result with all model comparisons
      all_models_summary <- create_model_comparison_summary(model_results)
      
      return(list(
        file = basename(tree_file),
        speciation_rate = result$lambda,
        extinction_rate = result$mu,
        net_diversification = result$lambda - result$mu,
        relative_extinction = if (result$lambda > 0) result$mu / result$lambda else NA,
        speciation_ci_lower = result$lambda_ci_lower,
        speciation_ci_upper = result$lambda_ci_upper,
        extinction_ci_lower = result$mu_ci_lower,
        extinction_ci_upper = result$mu_ci_upper,
        loglik = result$loglik,
        aic = result$aic,
        aicc = result$aicc,
        method = result$model_name,
        n_tips = n_tips,
        tree_length = tree_length,
        crown_age = crown_age,
        convergence = result$convergence,
        error = NA,
        # Add model comparison results
        all_models_aic = all_models_summary$aic_table,
        best_models_ranking = all_models_summary$ranking,
        delta_aic = all_models_summary$delta_aic
      ))
    } else {
      return(create_empty_result(tree_file, "All models failed", n_tips, tree_length))
    }
    
  }, error = function(e) {
    warning(sprintf("Error processing tree %s: %s", tree_file, e$message))
    return(create_empty_result(tree_file, as.character(e$message)))
  })
}

# Helper function to fit model combinations
fit_model_combination <- function(tree, lamb_type, mu_type, crown_age, model_name) {
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
    model_name = paste0("RPANDA_", model_name),
    lamb_type = lamb_type,
    mu_type = mu_type
  ))
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
    lamb_type = "constant",
    mu_type = "zero"
  ))
}

create_empty_result <- function(tree_file, error_msg, n_tips = NA, tree_length = NA) {
  return(list(
    file = basename(tree_file),
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
    n_tips = n_tips,
    tree_length = tree_length,
    crown_age = NA,
    convergence = NA,
    error = error_msg,
    all_models_aic = NA,
    best_models_ranking = NA,
    delta_aic = NA
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
cat("Starting RPANDA analysis with all model combinations...\n")
cat(sprintf("Tree folder: %s\n", tree_folder))
cat(sprintf("CSV file: %s\n", csv_file))
cat(sprintf("Output file: %s\n", output_csv))

# Print model combinations that will be tested
cat("\n=== MODEL COMBINATIONS TO BE TESTED ===\n")
for (model_name in names(model_combinations)) {
  model_spec <- model_combinations[[model_name]]
  cat(sprintf("%s: Birth %s, Death %s\n", model_name, model_spec$lamb_type, model_spec$mu_type))
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
rate_columns <- c("speciation_rate", "extinction_rate", "net_diversification", 
                 "relative_extinction", "speciation_ci_lower", "speciation_ci_upper",
                 "extinction_ci_lower", "extinction_ci_upper", "tree_loglik", 
                 "tree_aic", "tree_aicc", "diversification_method", "n_tips", 
                 "tree_length", "crown_age", "convergence", "tree_error",
                 "all_models_aic", "best_models_ranking", "delta_aic")

for (col in rate_columns) {
  csv_data[[col]] <- NA
}

# Fill in the matched data
for (i in 1:nrow(csv_data)) {
  if (!is.na(csv_data$tree_match[i])) {
    match_idx <- csv_data$tree_match[i]
    
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
    csv_data$n_tips[i] <- rates_df$n_tips[match_idx]
    csv_data$tree_length[i] <- rates_df$tree_length[match_idx]
    csv_data$crown_age[i] <- rates_df$crown_age[match_idx]
    csv_data$convergence[i] <- rates_df$convergence[match_idx]
    csv_data$tree_error[i] <- rates_df$error[match_idx]
    csv_data$all_models_aic[i] <- rates_df$all_models_aic[match_idx]
    csv_data$best_models_ranking[i] <- rates_df$best_models_ranking[match_idx]
    csv_data$delta_aic[i] <- rates_df$delta_aic[match_idx]
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
  
  # Show method distribution
  methods <- table(csv_data$diversification_method[!is.na(csv_data$diversification_method)])
  cat("\n=== BEST MODELS SELECTED ===\n")
  for (i in 1:length(methods)) {
    cat(sprintf("%s: %d cases\n", names(methods)[i], methods[i]))
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
  cat("- all_models_aic: AIC values for all fitted models\n")
  cat("- best_models_ranking: Models ranked by AIC (best to worst)\n")
  cat("- delta_aic: Delta AIC values relative to best model\n")
}

cat("\nRPANDA analysis with all model combinations complete!\n")