library(tidyverse)

data_dir <- "results/study3"

models <- c(
  "Qwen3.5-9B",
  "Qwen3-Next-80B-A3B-Instruct",
  "Llama-4-Maverick-17B-128E-Instruct-FP8"
)
model_labels <- c(
  "Qwen3.5-9B" = "Qwen3.5-9B",
  "Qwen3-Next-80B-A3B-Instruct" = "Qwen3-Next 3B",
  "Llama-4-Maverick-17B-128E-Instruct-FP8" = "Llama 4 Maverick"
)
datasets <- c("HotelBookings", "LendingClub", "MoralMachine", "WikipediaToxicity", "MovieLens")

# Load all hint-based, no-thinking, no-cost results
results <- map_dfr(models, function(m) {
  map_dfr(datasets, function(ds) {
    pattern <- sprintf("%s/%s_.*_nothink_%s\\.csv", data_dir, ds, m)
    files <- Sys.glob(pattern)
    files <- files[!grepl("summary|nohint|cost", files)]
    map_dfr(files, function(f) {
      df <- read_csv(f, show_col_types = FALSE)
      tibble(
        model = m,
        dataset = ds,
        pred_acc = mean(df$correct),
        pred_se = sqrt(mean(df$correct) * (1 - mean(df$correct)) / nrow(df)),
        esc_rate = mean(df$escalate),
        esc_se = sqrt(mean(df$escalate) * (1 - mean(df$escalate)) / nrow(df)),
        n = nrow(df)
      )
    })
  })
})

results <- results %>%
  mutate(model_label = model_labels[model])

# ── Chart 1: One facet per model, colored by dataset ──
ggplot(results, aes(x = pred_acc, y = esc_rate, color = dataset)) +
  geom_point(size = 2) +
  geom_line() +
  geom_errorbar(aes(ymin = esc_rate - esc_se, ymax = esc_rate + esc_se), width = 0.01) +
  geom_vline(xintercept = 0.75, linetype = "dotted", alpha = 0.5) +
  facet_wrap(~model_label) +
  labs(
    x = "Predictive Accuracy",
    y = "Escalation Rate",
    color = "Dataset",
    title = "Escalation Rate vs. Predictive Accuracy"
  ) +
  xlim(0.45, 1.02) +
  ylim(0, 1.02) +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave("paper/figures/esc_vs_acc_nothink_faceted.png", width = 14, height = 5, dpi = 150)

# ── Chart 2: One facet per dataset, colored by model ──
ggplot(results, aes(x = pred_acc, y = esc_rate, color = model_label)) +
  geom_point(size = 2) +
  geom_line() +
  geom_errorbar(aes(ymin = esc_rate - esc_se, ymax = esc_rate + esc_se), width = 0.01) +
  geom_vline(xintercept = 0.75, linetype = "dotted", alpha = 0.5) +
  facet_wrap(~dataset) +
  labs(
    x = "Predictive Accuracy",
    y = "Escalation Rate",
    color = "Model",
    title = "Escalation Rate vs. Predictive Accuracy"
  ) +
  xlim(0.45, 1.02) +
  ylim(0, 1.02) +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave("paper/figures/esc_vs_acc_nothink_by_dataset_faceted.png", width = 14, height = 8, dpi = 150)
