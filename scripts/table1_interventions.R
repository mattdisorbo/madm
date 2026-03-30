# Table 1: Escalation accuracy under different interventions
# Cost ratio R=4, optimal threshold τ*=75%
# Correct action: escalate if pred_acc < 75%, implement if >= 75%

library(tidyverse)

DATA_DIR <- "results/study3"
THRESHOLD <- 0.75

# ============================================================
# Load summary data for each intervention
# ============================================================

load_intervention <- function(tag, datasets) {
  map_dfr(datasets, function(ds) {
    fpath <- file.path(DATA_DIR, paste0(ds, "_summary_", tag, ".csv"))
    if (file.exists(fpath)) {
      df <- read_csv(fpath, show_col_types = FALSE)
      if (nrow(df) > 0) {
        df %>% mutate(dataset = ds, tag = tag)
      }
    }
  })
}

# Define interventions
datasets_4 <- c("HotelBookings", "LendingClub", "WikipediaToxicity", "MovieLens")
datasets_3 <- c("HotelBookings", "LendingClub", "WikipediaToxicity")

interventions <- tribble(
  ~label,                                ~tag,                                ~datasets,
  "Qwen 9B baseline",                   "nothink_Qwen3.5-9B",               list(datasets_4),
  "  + cost framing",                    "cost4_nothink_Qwen3.5-9B",         list(datasets_4),
  "  + thinking",                        "think_Qwen3.5-9B",                 list(datasets_4),
  "  + thinking + cost framing",         "cost4_think_Qwen3.5-9B",           list(datasets_4),
  "GPT-5-mini baseline (no reasoning)",  "nothink_noreason_gpt-5-mini",      list(datasets_3),
  "  + cost framing",                    "cost4_nothink_noreason_gpt-5-mini", list(datasets_3),
  "  + reasoning",                       "nothink_gpt-5-mini",               list(datasets_3),
  "  + reasoning + cost framing",        "cost4_nothink_gpt-5-mini",         list(datasets_3),
)

# ============================================================
# Compute escalation accuracy per intervention
# ============================================================

compute_esc_accuracy <- function(df) {
  df %>%
    mutate(
      should_escalate = pred_acc < THRESHOLD,
      correct_decisions = ifelse(should_escalate,
        round(esc_rate * n),        # escalated when should escalate
        round((1 - esc_rate) * n)   # implemented when should implement
      )
    ) %>%
    summarise(
      total_correct = sum(correct_decisions),
      total_n = sum(n),
      accuracy = total_correct / total_n,
      .groups = "drop"
    )
}

results <- map_dfr(seq_len(nrow(interventions)), function(i) {
  df <- load_intervention(interventions$tag[i], interventions$datasets[[i]][[1]])
  if (is.null(df) || nrow(df) == 0) return(NULL)
  acc <- compute_esc_accuracy(df)
  tibble(
    label = interventions$label[i],
    tag = interventions$tag[i],
    accuracy = acc$accuracy,
    n = acc$total_n
  )
})

# Add delta column
results <- results %>%
  mutate(
    group = cumsum(str_detect(label, "baseline")),
    baseline_acc = accuracy[match(group, group)],
    delta = accuracy - baseline_acc
  ) %>%
  select(-group, -baseline_acc)

cat("=== Table 1 ===\n")
results %>%
  mutate(
    accuracy = paste0(round(accuracy * 100, 1), "%"),
    delta = ifelse(delta == 0, "---", paste0(sprintf("%+.1f", delta * 100)))
  ) %>%
  select(label, accuracy, n, delta) %>%
  print(n = Inf)

# ============================================================
# Inspect the data per intervention
# ============================================================

# Load detailed per-condition data for any intervention
load_conditions <- function(tag, datasets) {
  df <- load_intervention(tag, datasets)
  if (is.null(df) || nrow(df) == 0) return(NULL)
  df %>%
    mutate(
      should_escalate = pred_acc < THRESHOLD,
      correct_decisions = ifelse(should_escalate,
        round(esc_rate * n),
        round((1 - esc_rate) * n)
      ),
      condition_accuracy = correct_decisions / n
    ) %>%
    arrange(pred_acc)
}

# Example: inspect Qwen baseline per condition
# qwen_base <- load_conditions("nothink_Qwen3.5-9B", datasets_4)
# View(qwen_base)

# Compare baseline vs thinking + cost for Qwen
# qwen_think_cost <- load_conditions("cost4_think_Qwen3.5-9B", datasets_4)
# View(qwen_think_cost)

# See which conditions flip from wrong to right with the intervention:
# compare <- qwen_base %>%
#   select(condition, dataset, base_acc = condition_accuracy, pred_acc, should_escalate) %>%
#   inner_join(
#     qwen_think_cost %>% select(condition, dataset, intervention_acc = condition_accuracy),
#     by = c("condition", "dataset")
#   ) %>%
#   mutate(improved = intervention_acc > base_acc)
# View(compare)

# ============================================================
# Load individual sample-level data
# ============================================================

load_samples <- function(model_tag, dataset, condition, prefix = "nothink") {
  fpath <- file.path(DATA_DIR, paste0(dataset, "_", condition, "_", prefix, "_", model_tag, ".csv"))
  if (file.exists(fpath)) {
    read_csv(fpath, show_col_types = FALSE) %>%
      mutate(dataset = dataset, model = model_tag, condition_name = condition, prefix = prefix)
  } else {
    cat("File not found:", fpath, "\n")
    NULL
  }
}

load_all_samples <- function(model_tag, dataset, prefix = "nothink") {
  summary_path <- file.path(DATA_DIR, paste0(dataset, "_summary_", prefix, "_", model_tag, ".csv"))
  if (!file.exists(summary_path)) return(NULL)
  conditions <- read_csv(summary_path, show_col_types = FALSE)$condition
  map_dfr(conditions, ~load_samples(model_tag, dataset, .x, prefix = prefix))
}

# Example: GPT-5-mini with reasoning vs without
# with_reason <- load_all_samples("gpt-5-mini", "LendingClub", prefix = "nothink")
# no_reason <- load_all_samples("gpt-5-mini", "LendingClub", prefix = "nothink_noreason")
# View(with_reason)
# View(no_reason)

# Example: Qwen thinking + cost
# qwen_tc <- load_all_samples("Qwen3.5-9B", "HotelBookings", prefix = "cost4_think")
# View(qwen_tc)

# Prefixes for each intervention:
#   "nothink_Qwen3.5-9B"               -> prefix = "nothink",             model = "Qwen3.5-9B"
#   "cost4_nothink_Qwen3.5-9B"         -> prefix = "cost4_nothink",       model = "Qwen3.5-9B"
#   "think_Qwen3.5-9B"                 -> prefix = "think",               model = "Qwen3.5-9B"
#   "cost4_think_Qwen3.5-9B"           -> prefix = "cost4_think",         model = "Qwen3.5-9B"
#   "nothink_noreason_gpt-5-mini"      -> prefix = "nothink_noreason",    model = "gpt-5-mini"
#   "cost4_nothink_noreason_gpt-5-mini"-> prefix = "cost4_nothink_noreason", model = "gpt-5-mini"
#   "nothink_gpt-5-mini"               -> prefix = "nothink",             model = "gpt-5-mini"
#   "cost4_nothink_gpt-5-mini"         -> prefix = "cost4_nothink",       model = "gpt-5-mini"
