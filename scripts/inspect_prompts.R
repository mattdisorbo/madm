# Inspect all prompts across every experimental variant
# Use this to verify hint text, cost framing, thinking mode, etc.

library(tidyverse)

DATA_DIR <- "results/study3"

# ============================================================
# List all CSV files and parse their variant info
# ============================================================

all_files <- list.files(DATA_DIR, pattern = "\\.csv$", full.names = TRUE)
all_files <- all_files[!grepl("summary", all_files)]

parse_filename <- function(fpath) {
  fname <- basename(fpath) %>% str_remove("\\.csv$")

  # Known model tags (longest first to avoid partial matches)
  model_tags <- c(
    "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "Llama-3.3-70B-Instruct-Turbo",
    "Mistral-Small-24B-Instruct-2501",
    "Mixtral-8x7B-Instruct-v0.1",
    "Qwen3.5-397B-A17B",
    "Qwen3.5-9B",
    "gpt-5-nano",
    "gpt-5-mini"
  )

  model <- NA_character_
  remainder <- fname
  for (tag in model_tags) {
    if (str_ends(fname, tag)) {
      model <- tag
      remainder <- str_remove(fname, paste0("_", tag, "$"))
      break
    }
  }
  if (is.na(model)) return(NULL)

  # Known datasets
  datasets <- c("HotelBookings", "LendingClub", "WikipediaToxicity", "MovieLens", "MoralMachine")
  dataset <- NA_character_
  for (ds in datasets) {
    if (str_starts(remainder, ds)) {
      dataset <- ds
      remainder <- str_remove(remainder, paste0("^", ds, "_"))
      break
    }
  }
  if (is.na(dataset)) return(NULL)

  # Parse modifiers
  has_think    <- str_detect(remainder, "_think($|_)") & !str_detect(remainder, "nothink")
  has_nothink  <- str_detect(remainder, "nothink")
  has_nohint   <- str_detect(remainder, "nohint")
  has_cost4    <- str_detect(remainder, "cost4")
  has_noreason <- str_detect(remainder, "noreason")
  has_isolated <- str_detect(remainder, "isolated")

  # Extract condition name by removing known modifiers
  condition <- remainder %>%
    str_remove_all("_?(nothink|think|nohint|cost4|noreason|isolated)") %>%
    str_trim() %>%
    str_remove("^_|_$")

  # Build variant label
  parts <- c()
  if (has_cost4) parts <- c(parts, "cost")
  if (has_think) parts <- c(parts, "think")
  if (has_nothink) parts <- c(parts, "nothink")
  if (has_nohint) parts <- c(parts, "nohint")
  if (has_noreason) parts <- c(parts, "noreason")
  if (has_isolated) parts <- c(parts, "isolated")
  variant <- paste(parts, collapse = "+")
  if (variant == "") variant <- "base"

  tibble(
    file = fpath,
    dataset = dataset,
    condition = condition,
    model = model,
    variant = variant,
    has_think = has_think,
    has_nothink = has_nothink,
    has_nohint = has_nohint,
    has_cost4 = has_cost4,
    has_noreason = has_noreason,
    has_isolated = has_isolated
  )
}

cat("Parsing all filenames...\n")
file_index <- map_dfr(all_files, parse_filename)
cat(sprintf("Found %d files across %d variants\n\n", nrow(file_index), n_distinct(file_index$variant)))

# ============================================================
# Summary: what variants exist per model?
# ============================================================

cat("=== Variants per model ===\n")
file_index %>%
  count(model, variant) %>%
  pivot_wider(names_from = variant, values_from = n, values_fill = 0) %>%
  print(n = Inf, width = Inf)

cat("\n=== Variants per dataset ===\n")
file_index %>%
  count(dataset, variant) %>%
  pivot_wider(names_from = variant, values_from = n, values_fill = 0) %>%
  print(n = Inf, width = Inf)

# ============================================================
# Load one sample from each variant to inspect prompts
# ============================================================

load_one_sample <- function(fpath) {
  df <- read_csv(fpath, show_col_types = FALSE)
  if (nrow(df) == 0) return(NULL)
  df %>% slice(1)
}

cat("\n=== Loading one sample per variant per model per dataset ===\n")

# Pick one file per (model, dataset, variant) combo
exemplars <- file_index %>%
  group_by(model, dataset, variant) %>%
  slice(1) %>%
  ungroup()

samples <- map_dfr(seq_len(nrow(exemplars)), function(i) {
  row <- exemplars[i, ]
  s <- load_one_sample(row$file)
  if (is.null(s)) return(NULL)
  s %>%
    mutate(
      model = row$model,
      dataset = row$dataset,
      condition = row$condition,
      variant = row$variant,
      source_file = basename(row$file)
    )
})

cat(sprintf("Loaded %d exemplar samples\n\n", nrow(samples)))

# ============================================================
# Show prompt structure for each variant
# ============================================================

show_prompt <- function(row) {
  cat(sprintf("\n{'='*70}\n"))
  cat(sprintf("MODEL: %s | DATASET: %s | VARIANT: %s\n", row$model, row$dataset, row$variant))
  cat(sprintf("CONDITION: %s | FILE: %s\n", row$condition, row$source_file))
  cat(sprintf("{'='*70}\n"))

  cat("\n--- PROMPT (Turn 1: User) ---\n")
  prompt_text <- as.character(row$prompt)
  # Show first 500 chars to keep output manageable
  if (nchar(prompt_text) > 500) {
    cat(substr(prompt_text, 1, 500), "...[truncated]\n")
  } else {
    cat(prompt_text, "\n")
  }

  if ("thought" %in% names(row) && !is.na(row$thought)) {
    cat("\n--- THOUGHT (Turn 1: Assistant) ---\n")
    thought_text <- as.character(row$thought)
    if (nchar(thought_text) > 500) {
      cat(substr(thought_text, 1, 500), "...[truncated]\n")
    } else {
      cat(thought_text, "\n")
    }
  }

  if ("esc_prompt" %in% names(row) && !is.na(row$esc_prompt)) {
    cat("\n--- ESC_PROMPT (Turn 2: User) ---\n")
    cat(as.character(row$esc_prompt), "\n")
  }

  if ("esc_thought" %in% names(row) && !is.na(row$esc_thought)) {
    cat("\n--- ESC_THOUGHT (Turn 2: Assistant) ---\n")
    esc_text <- as.character(row$esc_thought)
    if (nchar(esc_text) > 500) {
      cat(substr(esc_text, 1, 500), "...[truncated]\n")
    } else {
      cat(esc_text, "\n")
    }
  }

  if ("escalate" %in% names(row)) {
    cat(sprintf("\n--- DECISION: %s ---\n", ifelse(row$escalate == 1, "ESCALATE", "IMPLEMENT")))
  }
  if ("correct" %in% names(row)) {
    cat(sprintf("--- PREDICTION CORRECT: %s ---\n", ifelse(row$correct == 1, "YES", "NO")))
  }
  cat("\n")
}

# ============================================================
# Interactive exploration helpers
# ============================================================

# Show all unique variants
cat("=== All unique variants ===\n")
cat(paste(sort(unique(file_index$variant)), collapse = "\n"), "\n")

# To inspect prompts for a specific variant, run:
# samples %>% filter(variant == "nothink", model == "Qwen3.5-9B") %>% slice(1) %>% show_prompt()

# ============================================================
# Print one exemplar per variant for Qwen3.5-9B on HotelBookings
# (change model/dataset as needed)
# ============================================================

cat("\n\n========================================\n")
cat("EXEMPLAR PROMPTS: Qwen3.5-9B x HotelBookings\n")
cat("========================================\n")

exemplar_set <- samples %>%
  filter(model == "Qwen3.5-9B", dataset == "HotelBookings") %>%
  arrange(variant)

if (nrow(exemplar_set) > 0) {
  for (i in seq_len(nrow(exemplar_set))) {
    show_prompt(exemplar_set[i, ])
  }
} else {
  cat("No Qwen3.5-9B HotelBookings samples found. Try another model.\n")
}

# ============================================================
# Quick comparison: hint vs nohint
# ============================================================

cat("\n\n========================================\n")
cat("HINT vs NOHINT COMPARISON\n")
cat("========================================\n")

hint_samples <- samples %>% filter(variant == "nothink", dataset == "HotelBookings") %>% slice(1)
nohint_samples <- samples %>% filter(variant == "nothink+nohint", dataset == "HotelBookings") %>% slice(1)

if (nrow(hint_samples) > 0) {
  cat("\n--- WITH HINT ---\n")
  show_prompt(hint_samples[1, ])
}
if (nrow(nohint_samples) > 0) {
  cat("\n--- WITHOUT HINT ---\n")
  show_prompt(nohint_samples[1, ])
}

# ============================================================
# Quick comparison: cost framing
# ============================================================

cat("\n\n========================================\n")
cat("COST vs NO-COST COMPARISON\n")
cat("========================================\n")

nocost <- samples %>% filter(variant == "nothink", dataset == "HotelBookings") %>% slice(1)
cost <- samples %>% filter(variant == "cost+nothink", dataset == "HotelBookings") %>% slice(1)

if (nrow(nocost) > 0) {
  cat("\n--- WITHOUT COST ---\n")
  show_prompt(nocost[1, ])
}
if (nrow(cost) > 0) {
  cat("\n--- WITH COST ---\n")
  show_prompt(cost[1, ])
}

# ============================================================
# Usage: explore interactively
# ============================================================
#
# Browse the file index:
  # View(file_index)
#
# See all variants for a model:
  # file_index %>% filter(model == "gpt-5-mini") %>% count(variant)
#
# Load full data for a specific file:
#   df <- read_csv("results/study3/HotelBookings_has_special_requests_nothink_Qwen3.5-9B.csv")
#   View(df)
#
# Show prompt for any sample:
  # samples %>% filter(variant == "nothink+nohint", model == "Qwen3.5-9B") %>% slice(1) %>% show_prompt()
#
# Compare esc_prompt across variants:
#   samples %>% select(model, dataset, variant, esc_prompt) %>% View()
#
# Load ALL data for one variant:
#   variant_files <- file_index %>% filter(variant == "nothink", model == "Qwen3.5-9B")
#   all_data <- map_dfr(variant_files$file, read_csv, show_col_types = FALSE)
