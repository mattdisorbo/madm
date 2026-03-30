# ============================================================
# Study 3: Data exploration scripts
# Run interactively or source entire file
# ============================================================

library(tidyverse)

DATA_DIR <- "results/study3"

# ============================================================
# 1. Load all summary data into one tidy dataframe
# ============================================================

load_summaries <- function(tag_suffix, condition_type = "hint") {
  # tag_suffix: e.g. "Qwen3.5-9B" or "gpt-5-mini"
  # condition_type: "hint" (nothink) or "nohint" (nothink_nohint)

  datasets <- c("HotelBookings", "LendingClub", "WikipediaToxicity", "MovieLens", "MoralMachine")

  if (condition_type == "hint") {
    pattern <- paste0("_summary_nothink_", tag_suffix, ".csv")
  } else {
    pattern <- paste0("_summary_nothink_nohint_", tag_suffix, ".csv")
  }

  map_dfr(datasets, function(ds) {
    fpath <- file.path(DATA_DIR, paste0(ds, pattern))
    if (file.exists(fpath)) {
      df <- read_csv(fpath, show_col_types = FALSE)
      if (nrow(df) > 0) {
        df %>% mutate(dataset = ds, model = tag_suffix)
      }
    }
  })
}

# Model tags and display names
models <- tribble(
  ~display,              ~tag,
  "Qwen3.5-9B",         "Qwen3.5-9B",
  "Qwen3.5-397B",       "Qwen3.5-397B-A17B",
  "GPT-5-nano",         "gpt-5-nano",
  "GPT-5-mini",         "gpt-5-mini",
  "Llama4-Maverick",    "Llama-4-Maverick-17B-128E-Instruct-FP8",
  "Llama3.3-70B",       "Llama-3.3-70B-Instruct-Turbo",
  "Mixtral-8x7B",       "Mixtral-8x7B-Instruct-v0.1",
  "Mistral-Small-24B",  "Mistral-Small-24B-Instruct-2501"
)

# Load all hint data
hint_all <- map_dfr(seq_len(nrow(models)), function(i) {
  load_summaries(models$tag[i], "hint") %>%
    mutate(display = models$display[i])
})

# Load all nohint data
nohint_all <- map_dfr(seq_len(nrow(models)), function(i) {
  load_summaries(models$tag[i], "nohint") %>%
    mutate(display = models$display[i])
})

cat("Loaded", nrow(hint_all), "hint rows and", nrow(nohint_all), "nohint rows\n")

# ============================================================
# 2. Compute p* (implicit threshold) per model
# ============================================================

compute_pstar <- function(df) {
  # Fit linear regression: esc_rate ~ pred_acc
  # Solve for pred_acc where esc_rate = 0.5
  fit <- lm(esc_rate ~ pred_acc, data = df)
  slope <- coef(fit)["pred_acc"]
  intercept <- coef(fit)["(Intercept)"]
  pstar <- (0.5 - intercept) / slope
  tibble(pstar = as.numeric(pstar), slope = as.numeric(slope), intercept = as.numeric(intercept))
}

pstar_df <- hint_all %>%
  filter(dataset != "MoralMachine") %>%
  group_by(display, model) %>%
  group_modify(~compute_pstar(.x)) %>%
  ungroup()

cat("\n=== Implicit thresholds (p*) ===\n")
pstar_df %>% arrange(pstar) %>% print(n = Inf)

# ============================================================
# 3. Compute self-estimated accuracy (ahat) per model
# ============================================================

compute_ahat <- function(hint_df, nohint_df) {
  fit <- lm(esc_rate ~ pred_acc, data = hint_df)
  slope <- coef(fit)["pred_acc"]
  intercept <- coef(fit)["(Intercept)"]

  avg_nohint_esc <- mean(nohint_df$esc_rate, na.rm = TRUE)
  ahat <- pmin(pmax((avg_nohint_esc - intercept) / slope, 0), 1)
  actual_avg <- mean(hint_df$pred_acc, na.rm = TRUE)

  tibble(ahat = as.numeric(ahat), actual_avg = actual_avg, gap = as.numeric(ahat) - actual_avg)
}

ahat_df <- map_dfr(seq_len(nrow(models)), function(i) {
  h <- hint_all %>% filter(model == models$tag[i], dataset != "MoralMachine")
  n <- nohint_all %>% filter(model == models$tag[i], dataset != "MoralMachine")
  if (nrow(h) > 0 & nrow(n) > 0) {
    compute_ahat(h, n) %>% mutate(display = models$display[i])
  }
})

cat("\n=== Self-estimated accuracy (ahat) ===\n")
ahat_df %>% arrange(desc(gap)) %>% print(n = Inf)

# ============================================================
# 4. Compute per-condition overconfidence
# ============================================================

compute_overconfidence <- function(hint_df, nohint_df) {
  # Fit regression on all hint data for this model
  fit <- lm(esc_rate ~ pred_acc, data = hint_df)
  slope <- coef(fit)["pred_acc"]
  intercept <- coef(fit)["(Intercept)"]

  # For each nohint condition, compute self-estimated accuracy
  nohint_df %>%
    inner_join(
      hint_df %>% select(condition, dataset, actual_acc = pred_acc),
      by = c("condition", "dataset")
    ) %>%
    mutate(
      self_est_acc = pmin(pmax((esc_rate - intercept) / slope, 0), 1),
      overconfident = self_est_acc > actual_acc,
      gap = self_est_acc - actual_acc
    )
}

overconf_all <- map_dfr(seq_len(nrow(models)), function(i) {
  h <- hint_all %>% filter(model == models$tag[i], dataset != "MoralMachine")
  n <- nohint_all %>% filter(model == models$tag[i], dataset != "MoralMachine")
  if (nrow(h) > 0 & nrow(n) > 0) {
    compute_overconfidence(h, n) %>% mutate(display = models$display[i])
  }
})

cat("\n=== Overconfidence summary ===\n")
overconf_all %>%
  group_by(display) %>%
  summarise(
    n_conditions = n(),
    n_overconfident = sum(overconfident),
    pct_overconfident = mean(overconfident) * 100,
    mean_gap = mean(gap) * 100,
    .groups = "drop"
  ) %>%
  arrange(desc(pct_overconfident)) %>%
  print(n = Inf)

cat("\nTotal:", sum(overconf_all$overconfident), "/", nrow(overconf_all),
    "(", round(mean(overconf_all$overconfident) * 100), "%)\n")

# ============================================================
# 5. Quick plots (run interactively)
# ============================================================

# --- Figure 3 recreation: Escalation rate vs accuracy ---
plot_esc_vs_acc <- function() {
  hint_all %>%
    filter(dataset != "MoralMachine") %>%
    mutate(display = factor(display, levels = c(
      "Qwen3.5-9B", "GPT-5-nano", "Llama4-Maverick", "Mixtral-8x7B",
      "Qwen3.5-397B", "GPT-5-mini", "Llama3.3-70B", "Mistral-Small-24B"
    ))) %>%
    ggplot(aes(x = pred_acc, y = esc_rate, color = dataset)) +
    geom_point(size = 2, alpha = 0.7) +
    geom_errorbar(aes(
      ymin = esc_rate - sqrt(esc_rate * (1 - esc_rate) / n),
      ymax = esc_rate + sqrt(esc_rate * (1 - esc_rate) / n)
    ), width = 0, alpha = 0.5) +
    facet_wrap(~display, nrow = 2) +
    labs(x = "Predictive accuracy", y = "Escalation rate", color = "Dataset") +
    theme_minimal() +
    theme(legend.position = "bottom")
}

# --- Figure 4 recreation: Overconfidence scatter ---
plot_overconfidence <- function() {
  overconf_all %>%
    mutate(display = factor(display, levels = c(
      "Qwen3.5-9B", "GPT-5-nano", "Llama4-Maverick", "Mixtral-8x7B",
      "Qwen3.5-397B", "GPT-5-mini", "Llama3.3-70B", "Mistral-Small-24B"
    ))) %>%
    ggplot(aes(x = actual_acc, y = self_est_acc, color = dataset)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
    geom_point(size = 2, alpha = 0.7) +
    facet_wrap(~display, nrow = 2) +
    coord_cartesian(xlim = c(0.4, 1), ylim = c(0.4, 1)) +
    labs(x = "Actual accuracy", y = "Self-estimated accuracy", color = "Dataset") +
    theme_minimal() +
    theme(legend.position = "bottom")
}

# --- Escalation rate distribution by model ---
plot_esc_distribution <- function() {
  hint_all %>%
    filter(dataset != "MoralMachine") %>%
    mutate(display = fct_reorder(display, esc_rate, .fun = median)) %>%
    ggplot(aes(x = display, y = esc_rate, fill = dataset)) +
    geom_boxplot(alpha = 0.7, outlier.size = 1) +
    coord_flip() +
    labs(x = NULL, y = "Escalation rate", fill = "Dataset") +
    theme_minimal()
}

# --- Prediction accuracy by model and dataset ---
plot_pred_accuracy <- function() {
  hint_all %>%
    filter(dataset != "MoralMachine") %>%
    mutate(display = fct_reorder(display, pred_acc, .fun = median)) %>%
    ggplot(aes(x = display, y = pred_acc, fill = dataset)) +
    geom_boxplot(alpha = 0.7, outlier.size = 1) +
    coord_flip() +
    labs(x = NULL, y = "Predictive accuracy", fill = "Dataset") +
    theme_minimal()
}

# --- Within-family comparison ---
plot_family_comparison <- function() {
  families <- tribble(
    ~family,   ~small,          ~large,
    "Qwen",    "Qwen3.5-9B",   "Qwen3.5-397B",
    "GPT",     "GPT-5-nano",   "GPT-5-mini",
    "Llama",   "Llama4-Maverick", "Llama3.3-70B",
    "Mistral", "Mixtral-8x7B", "Mistral-Small-24B"
  )

  bind_rows(
    pstar_df %>% select(display, pstar) %>% mutate(metric = "p*", value = pstar),
    ahat_df %>% select(display, ahat, actual_avg) %>%
      pivot_longer(c(ahat, actual_avg), names_to = "metric", values_to = "value") %>%
      mutate(metric = ifelse(metric == "ahat", "Self-estimated acc", "Actual acc"))
  ) %>%
    left_join(
      families %>% pivot_longer(c(small, large), names_to = "size", values_to = "display"),
      by = "display"
    ) %>%
    filter(!is.na(family)) %>%
    ggplot(aes(x = size, y = value, group = family, color = family)) +
    geom_point(size = 3) +
    geom_line() +
    facet_wrap(~metric, scales = "free_y") +
    labs(x = "Model size", y = "Value", color = "Family") +
    theme_minimal()
}

cat("\n=== Ready! ===\n")
cat("Available dataframes: hint_all, nohint_all, pstar_df, ahat_df, overconf_all\n")
cat("Available plots: plot_esc_vs_acc(), plot_overconfidence(), plot_esc_distribution(),\n")
cat("                 plot_pred_accuracy(), plot_family_comparison()\n")
