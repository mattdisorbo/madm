# Figure 4: Overconfidence scatter -- actual vs self-estimated accuracy (2x4 grid)
# Same-family pairs vertically aligned, small on top, large on bottom

library(tidyverse)

DATA_DIR <- "results/study3"

# Model definitions
models <- tribble(
  ~display,             ~tag,
  "Qwen3.5-9B",        "Qwen3.5-9B",
  "GPT-5-nano",        "gpt-5-nano",
  "Llama4-Maverick",   "Llama-4-Maverick-17B-128E-Instruct-FP8",
  "Mixtral-8x7B",      "Mixtral-8x7B-Instruct-v0.1",
  "Qwen3.5-397B",      "Qwen3.5-397B-A17B",
  "GPT-5-mini",        "gpt-5-mini",
  "Llama3.3-70B",      "Llama-3.3-70B-Instruct-Turbo",
  "Mistral-Small-24B", "Mistral-Small-24B-Instruct-2501"
)

datasets <- c("HotelBookings", "LendingClub", "WikipediaToxicity", "MovieLens")
ds_colors <- c(
  "HotelBookings" = "#E69F00",
  "LendingClub" = "#0072B2",
  "WikipediaToxicity" = "#D55E00",
  "MovieLens" = "#8B4DAB"
)

# ============================================================
# Load hint and nohint summaries
# ============================================================

load_summary <- function(tag, type = "hint") {
  pattern <- if (type == "hint") {
    paste0("_summary_nothink_", tag, ".csv")
  } else {
    paste0("_summary_nothink_nohint_", tag, ".csv")
  }
  map_dfr(datasets, function(ds) {
    fpath <- file.path(DATA_DIR, paste0(ds, pattern))
    if (file.exists(fpath)) {
      read_csv(fpath, show_col_types = FALSE) %>%
        filter(n > 0) %>%
        mutate(dataset = ds)
    }
  })
}

# ============================================================
# Compute self-estimated accuracy per model per condition
# ============================================================

overconf_all <- map_dfr(seq_len(nrow(models)), function(i) {
  tag <- models$tag[i]
  display <- models$display[i]

  hint <- load_summary(tag, "hint")
  nohint <- load_summary(tag, "nohint")
  if (nrow(hint) == 0 || nrow(nohint) == 0) return(NULL)

  # Fit regression on hint data: esc_rate ~ pred_acc
  fit <- lm(esc_rate ~ pred_acc, data = hint)
  slope <- coef(fit)["pred_acc"]
  intercept <- coef(fit)["(Intercept)"]

  # For each nohint condition, invert to get self-estimated accuracy
  nohint %>%
    inner_join(
      hint %>% select(condition, dataset, actual_acc = pred_acc),
      by = c("condition", "dataset")
    ) %>%
    mutate(
      self_est_acc = pmin(pmax((esc_rate - intercept) / slope, 0), 1),
      overconfident = self_est_acc > actual_acc,
      gap = self_est_acc - actual_acc,
      display = display
    )
})

# ============================================================
# Inspect the data
# ============================================================

# Summary per model
overconf_all %>%
  group_by(display) %>%
  summarise(
    n_conditions = n(),
    n_overconfident = sum(overconfident),
    pct_overconfident = round(mean(overconfident) * 100, 1),
    mean_gap = round(mean(gap) * 100, 1),
    mean_actual = round(mean(actual_acc) * 100, 1),
    mean_self_est = round(mean(self_est_acc) * 100, 1),
    .groups = "drop"
  ) %>%
  arrange(desc(pct_overconfident)) %>%
  print(n = Inf)

cat("\nTotal:", sum(overconf_all$overconfident), "/", nrow(overconf_all),
    "(", round(mean(overconf_all$overconfident) * 100), "%)\n")

# Per-model per-dataset breakdown
overconf_all %>%
  group_by(display, dataset) %>%
  summarise(
    n = n(),
    pct_overconf = round(mean(overconfident) * 100),
    mean_gap = round(mean(gap) * 100, 1),
    .groups = "drop"
  ) %>%
  print(n = Inf)

# Browse full data:
# View(overconf_all)

# Worst overconfidence cases:
# overconf_all %>% arrange(desc(gap)) %>% head(20) %>% View()

# Worst underconfidence cases:
# overconf_all %>% arrange(gap) %>% head(20) %>% View()

# ============================================================
# Load individual sample-level data (prompts, predictions, escalation decisions)
# ============================================================

load_samples <- function(model_tag, dataset, condition, nohint = FALSE) {
  hint_tag <- if (nohint) "_nothink_nohint_" else "_nothink_"
  fpath <- file.path(DATA_DIR, paste0(dataset, "_", condition, hint_tag, model_tag, ".csv"))
  if (file.exists(fpath)) {
    read_csv(fpath, show_col_types = FALSE) %>%
      mutate(dataset = dataset, model = model_tag, condition_name = condition, nohint = nohint)
  } else {
    cat("File not found:", fpath, "\n")
    NULL
  }
}

load_all_samples <- function(model_tag, dataset, nohint = FALSE) {
  hint_tag <- if (nohint) "_nothink_nohint_" else "_nothink_"
  summary_path <- file.path(DATA_DIR, paste0(dataset, "_summary", hint_tag, model_tag, ".csv"))
  if (!file.exists(summary_path)) return(NULL)
  conditions <- read_csv(summary_path, show_col_types = FALSE)$condition
  map_dfr(conditions, ~load_samples(model_tag, dataset, .x, nohint = nohint))
}

# Example: load nohint samples (what Figure 4 uses)
# all_lc_nohint <- load_all_samples("gpt-5-mini", "LendingClub", nohint = TRUE)
# View(all_lc_nohint)

# Compare hint vs nohint for same condition:
# hint <- load_samples("gpt-5-mini", "LendingClub", "fico_over_700", nohint = FALSE)
# nohint <- load_samples("gpt-5-mini", "LendingClub", "fico_over_700", nohint = TRUE)
# View(hint)
# View(nohint)

# Find the most overconfident conditions and inspect their nohint prompts:
# worst <- overconf_all %>% arrange(desc(gap)) %>% head(5)
# samples <- load_samples(worst$model[1], worst$dataset[1], worst$condition[1], nohint = TRUE)
# View(samples)

# ============================================================
# Plot
# ============================================================

display_order <- c(
  "Qwen3.5-9B", "GPT-5-nano", "Llama4-Maverick", "Mixtral-8x7B",
  "Qwen3.5-397B", "GPT-5-mini", "Llama3.3-70B", "Mistral-Small-24B"
)
overconf_all$display <- factor(overconf_all$display, levels = display_order)

ggplot(overconf_all, aes(x = actual_acc, y = self_est_acc, color = dataset)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(size = 2, alpha = 0.7) +
  facet_wrap(~display, nrow = 2) +
  scale_color_manual(values = ds_colors) +
  coord_cartesian(xlim = c(0.4, 1), ylim = c(0.4, 1)) +
  labs(
    x = "Actual accuracy",
    y = "Self-estimated accuracy",
    color = "Dataset"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    strip.text = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

ggsave("paper/figures/fig4_R.png", width = 14, height = 7, dpi = 300)
cat("Saved to paper/figures/fig4_R.png\n")
