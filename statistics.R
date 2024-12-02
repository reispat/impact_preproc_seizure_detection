# Statistics for "The impact of EEG preprocessing parameters on ultra-low-power seizure detection"
# load packages
if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman") }
library("pacman")
p_load("lmerTest", "emmeans", "effsize", update=FALSE)
library("lmerTest")
library("emmeans")
library("effsize")

# set working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
print(getwd())

# define filenames
data_path <- "/data/"
filenames <- c(
  "df_folds_avg_sampling_rates",
  "df_folds_avg_window_sizes_0_percent_overlap",
  "df_folds_avg_window_sizes_50_percent_overlap",
  "df_folds_avg_bit_width",
  "df_folds_avg_n_channels")

# define conditions and metrics
conditions <- c("sampling_rate", "window_length", "bit_width", "n_channels")
metrics <- c("event_sensitivity", "event_false_detections_per_hour", "event_average_detection_delay")

# define reference levels for each condition
ref_levels <- list(
  sampling_rate = c("256", "128", "64"), # reference: 256Hz
  window_length = c("1", "2", "4", "8"), # reference: 1 s
  bit_width = c("16", "14", "12", "10", "8"), # reference: 16 bits
  n_channels = c("4", "3", "2", "1") # reference: 4 channels
)

# function to load data & do LMM, posthoc-contrasts and effect size calculation
do_stats <- function(filename, condition, metrics, ref_levels) {
  
  # load data
  df_folds_avg <- read.csv(paste0(getwd(), data_path, filename, ".csv"))
  df_folds_avg$patient <- as.factor(df_folds_avg$patient)
  
  # skip if condition not in data
  if (!(condition %in% colnames(df_folds_avg))) {
    return()
  }
  
  # convert condition to factor and set reference levels
  df_folds_avg[[condition]] <- as.factor(df_folds_avg[[condition]])
  if (!is.null(ref_levels[[condition]])) {
    df_folds_avg[[condition]] <- factor(df_folds_avg[[condition]], levels = ref_levels[[condition]])
  }
  
  # calculations for each metric
  for (metric in metrics) {
    if (!(metric %in% colnames(df_folds_avg))) {
      message(paste("Metric", metric, "not found in", filename))
      next
    }
    
    # fit LMM
    model_formula <- as.formula(paste(metric, "~ 1 +", condition, "+ (1 | patient)"))
    model <- lmer(model_formula, data = df_folds_avg)
    model_summary <- summary(model)
    
    # post-hoc contrasts
    emmeans_results <- emmeans(model, specs = condition)
    pairwise_results <- pairs(emmeans_results, adjust = "fdr") # Benjamini & Hochberg procedure
    
    # effect sizes (Cohen's d)
    cohens_d <- eff_size(emmeans_results, sigma = sigma(model), edf = Inf)
    
    # print results
    cat("\n----------------------- Results for", filename, metric, "-----------------------\n")
    cat("\nLMM:\n")
    print(model_summary$coefficients)
    cat("\nPost-hoc-contrasts:\n")
    print(pairwise_results)
    cat("\nEffect sizes (Cohen's d):\n")
    print(cohens_d)
  }
}

# do statistics
for (filename in filenames) {
  for (condition in conditions) {
    do_stats(filename, condition, metrics, ref_levels)
  }
}


