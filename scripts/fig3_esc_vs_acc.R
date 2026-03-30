# Figure 3: Escalation rate vs predictive accuracy (2x4 grid)
# Same-family pairs vertically aligned, small on top, large on bottom

library(tidyverse)

DATA_DIR <- "results/study3"

# Model definitions: display name, file tag, row position
models <- tribble(
  ~display,             ~tag,                                          ~row,
  "Qwen3.5-9B",        "Qwen3.5-9B",                                 1,
  "GPT-5-nano",        "gpt-5-nano",                                  1,
  "Llama4-Maverick",   "Llama-4-Maverick-17B-128E-Instruct-FP8",     1,
  "Mixtral-8x7B",      "Mixtral-8x7B-Instruct-v0.1",                 1,
  "Qwen3.5-397B",      "Qwen3.5-397B-A17B",                          2,
  "GPT-5-mini",        "gpt-5-mini",                                  2,
  "Llama3.3-70B",      "Llama-3.3-70B-Instruct-Turbo",               2,
  "Mistral-Small-24B", "Mistral-Small-24B-Instruct-2501",            2,
)

datasets <- c("HotelBookings", "LendingClub", "WikipediaToxicity", "MovieLens")
ds_colors <- c(
  "HotelBookings" = "#E69F00",
  "LendingClub" = "#0072B2",
  "WikipediaToxicity" = "#D55E00",
  "MovieLens" = "#8B4DAB"
)

# Load all hint summaries
df <- map_dfr(seq_len(nrow(models)), function(i) {
  map_dfr(datasets, function(ds) {
    fpath <- file.path(DATA_DIR, paste0(ds, "_summary_nothink_", models$tag[i], ".csv"))
    if (file.exists(fpath)) {
      read_csv(fpath, show_col_types = FALSE) %>%
        filter(n > 0) %>%
        mutate(
          dataset = ds,
          display = models$display[i],
          se = sqrt(esc_rate * (1 - esc_rate) / n)
        )
    }
  })
})

# Set factor levels for facet ordering (top row then bottom row)
display_order <- c(
  "Qwen3.5-9B", "GPT-5-nano", "Llama4-Maverick", "Mixtral-8x7B",
  "Qwen3.5-397B", "GPT-5-mini", "Llama3.3-70B", "Mistral-Small-24B"
)
df$display <- factor(df$display, levels = display_order)

# ============================================================
# Inspect the data
# ============================================================

# Quick summary per model
df %>%
  group_by(display, dataset) %>%
  summarise(
    n_conditions = n(),
    total_samples = sum(n),
    mean_pred_acc = mean(pred_acc),
    mean_esc_rate = mean(esc_rate),
    .groups = "drop"
  ) %>%
  print(n = Inf)

# Check for missing model-dataset combos
cat("\n=== Coverage matrix ===\n")
df %>%
  count(display, dataset) %>%
  pivot_wider(names_from = dataset, values_from = n, values_fill = 0) %>%
  print(n = Inf)

# Browse summary data:
# View(df)

# Filter to a single model:
# df %>% filter(display == "GPT-5-mini") %>% View()

# ============================================================
# Load individual sample-level data (prompts, predictions, escalation decisions)
# ============================================================

load_samples <- function(model_tag, dataset, condition) {
  # Loads the raw CSV for one model/dataset/condition
  fpath <- file.path(DATA_DIR, paste0(dataset, "_", condition, "_nothink_", model_tag, ".csv"))
  if (file.exists(fpath)) {
    read_csv(fpath, show_col_types = FALSE) %>%
      mutate(dataset = dataset, model = model_tag, condition_name = condition)
  } else {
    cat("File not found:", fpath, "\n")
    NULL
  }
}

# Example: load all samples for GPT-5-mini on LendingClub, fico_over_700 condition
# samples <- load_samples("gpt-5-mini", "LendingClub", "fico_over_700")
# View(samples)
# Columns: condition, ground_truth, prediction, correct, escalate, prompt, esc_prompt,
#           thought, esc_reasoning, thinking_predict, thinking_escalate, timestamp

# Load ALL samples for one model/dataset (all conditions):
load_all_samples <- function(model_tag, dataset) {
  summary_path <- file.path(DATA_DIR, paste0(dataset, "_summary_nothink_", model_tag, ".csv"))
  if (!file.exists(summary_path)) return(NULL)
  conditions <- read_csv(summary_path, show_col_types = FALSE)$condition
  map_dfr(conditions, ~load_samples(model_tag, dataset, .x))
}

# Example: all GPT-5-mini LendingClub samples
# all_lc <- load_all_samples("gpt-5-mini", "LendingClub")
# View(all_lc)
# all_lc %>% filter(correct == 0, escalate == 0) %>% select(prompt, thought, esc_reasoning) %>% View()  # wrong and didn't escalate

# ============================================================
# Plot
# ============================================================

ggplot(df, aes(x = pred_acc, y = esc_rate, color = dataset)) +
  geom_point(size = 2, alpha = 0.7) +
  geom_errorbar(
    aes(ymin = pmax(esc_rate - se, 0), ymax = pmin(esc_rate + se, 1)),
    width = 0, alpha = 0.5, linewidth = 0.4
  ) +
  facet_wrap(~display, nrow = 2) +
  scale_color_manual(values = ds_colors) +
  scale_x_continuous(limits = c(0.4, 1.0)) +
  scale_y_continuous(limits = c(0, 1.0)) +
  labs(
    x = "Predictive accuracy",
    y = "Escalation rate",
    color = "Dataset"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    strip.text = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

ggsave("paper/figures/fig3_R.png", width = 14, height = 7, dpi = 300)
cat("Saved to paper/figures/fig3_R.png\n")
