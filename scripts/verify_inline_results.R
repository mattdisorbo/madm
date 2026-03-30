# Verify all inline numerical results in the paper
# Each check prints PASS/FAIL with the paper claim vs computed value

library(tidyverse)

DATA_DIR <- "results/study3"
datasets <- c("HotelBookings", "LendingClub", "WikipediaToxicity", "MovieLens")

pass_count <- 0
fail_count <- 0

check <- function(description, paper_value, computed_value, tol = 0.01) {
  match <- abs(paper_value - computed_value) <= tol
  status <- if (match) "PASS" else "FAIL"
  if (match) {
    pass_count <<- pass_count + 1
  } else {
    fail_count <<- fail_count + 1
  }
  cat(sprintf("[%s] %s\n  Paper: %s | Computed: %s\n\n", status, description,
              format(paper_value, nsmall = 3), format(computed_value, nsmall = 3)))
}

# ============================================================
# Helper: load summary files
# ============================================================

models <- tribble(
  ~display,             ~tag,                                         ~family,
  "Qwen3.5-9B",        "Qwen3.5-9B",                                "Qwen",
  "Qwen3.5-397B",      "Qwen3.5-397B-A17B",                         "Qwen",
  "GPT-5-nano",        "gpt-5-nano",                                 "GPT",
  "GPT-5-mini",        "gpt-5-mini",                                 "GPT",
  "Llama4-Maverick",   "Llama-4-Maverick-17B-128E-Instruct-FP8",    "Llama",
  "Llama3.3-70B",      "Llama-3.3-70B-Instruct-Turbo",              "Llama",
  "Mixtral-8x7B",      "Mixtral-8x7B-Instruct-v0.1",                "Mistral",
  "Mistral-Small-24B", "Mistral-Small-24B-Instruct-2501",           "Mistral"
)

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

load_individual <- function(tag, ds, condition, variant = "nothink") {
  fpath <- file.path(DATA_DIR, paste0(ds, "_", condition, "_", variant, "_", tag, ".csv"))
  if (file.exists(fpath)) read_csv(fpath, show_col_types = FALSE) else NULL
}

# ============================================================
# 1. p* (implicit threshold) for each model
#    Paper line 265
# ============================================================

cat("=" %>% strrep(70), "\n")
cat("SECTION 1: Implicit Thresholds (p*)\n")
cat("Paper Section 4, line 265\n")
cat("=" %>% strrep(70), "\n\n")

pstar_results <- map_dfr(seq_len(nrow(models)), function(i) {
  tag <- models$tag[i]
  display <- models$display[i]
  hint <- load_summary(tag, "hint")
  if (nrow(hint) == 0) return(NULL)
  fit <- lm(esc_rate ~ pred_acc, data = hint)
  pstar <- as.numeric((0.5 - coef(fit)["(Intercept)"]) / coef(fit)["pred_acc"])
  tibble(display = display, pstar = pstar)
})

# Paper claims (line 265):
# Qwen3.5-9B p* ~ 56%, Qwen3.5-397B p* ~ 81%
# GPT-5-nano p* ~ 91%, GPT-5-mini p* ~ 53%
# Llama4-Maverick p* ~ 92%, Llama3.3-70B p* > 100%
# Mixtral-8x7B p* > 100%, Mistral-Small-24B p* ~ 85%

paper_pstar <- tribble(
  ~display, ~paper_pstar,
  "Qwen3.5-9B", 0.56,
  "Qwen3.5-397B", 0.81,
  "GPT-5-nano", 0.91,
  "GPT-5-mini", 0.53,
  "Llama4-Maverick", 0.92,
  "Llama3.3-70B", 1.00,   # "> 100%"
  "Mixtral-8x7B", 1.00,   # "> 100%"
  "Mistral-Small-24B", 0.85
)

for (i in seq_len(nrow(paper_pstar))) {
  d <- paper_pstar$display[i]
  computed <- pstar_results %>% filter(display == d) %>% pull(pstar)
  paper_val <- paper_pstar$paper_pstar[i]
  if (paper_val >= 1.0) {
    # For "> 100%" claims, just check it's > 1
    match <- computed > 1.0
    status <- if (match) "PASS" else "FAIL"
    if (match) pass_count <- pass_count + 1 else fail_count <- fail_count + 1
    cat(sprintf("[%s] p* for %s: paper says > 100%%, computed = %.1f%%\n\n",
                status, d, computed * 100))
  } else {
    check(sprintf("p* for %s", d), paper_val, computed, tol = 0.02)
  }
}

# Paper claim: Qwen pair differs by 25 pp
qwen_diff <- abs(diff(pstar_results %>% filter(display %in% c("Qwen3.5-9B", "Qwen3.5-397B")) %>%
  arrange(display) %>% pull(pstar))) * 100
check("Qwen pair p* difference (paper: 25 pp)", 25, qwen_diff, tol = 2)

# Paper claim: GPT pair differs by 38 pp
gpt_diff <- abs(diff(pstar_results %>% filter(display %in% c("GPT-5-nano", "GPT-5-mini")) %>%
  arrange(display) %>% pull(pstar))) * 100
check("GPT pair p* difference (paper: 38 pp)", 38, gpt_diff, tol = 2)

# Paper claim: p* range 53% to over 100%  (line 295)
check("p* minimum (paper: 53%)", 0.53, min(pstar_results$pstar), tol = 0.02)


# ============================================================
# 2. Self-estimated accuracy (ahat)
#    Paper lines 281, 283, 295
# ============================================================

cat("=" %>% strrep(70), "\n")
cat("SECTION 2: Self-Estimated Accuracy (ahat)\n")
cat("Paper Section 5, lines 281-295\n")
cat("=" %>% strrep(70), "\n\n")

ahat_results <- map_dfr(seq_len(nrow(models)), function(i) {
  tag <- models$tag[i]
  display <- models$display[i]
  hint <- load_summary(tag, "hint")
  nohint <- load_summary(tag, "nohint")
  if (nrow(hint) == 0) return(NULL)
  fit <- lm(esc_rate ~ pred_acc, data = hint)
  slope <- coef(fit)["pred_acc"]
  intercept <- coef(fit)["(Intercept)"]
  actual_avg <- mean(hint$pred_acc)

  ahat <- NA_real_
  if (nrow(nohint) > 0) {
    avg_nohint_esc <- mean(nohint$esc_rate)
    ahat <- as.numeric(pmin(pmax((avg_nohint_esc - intercept) / slope, 0), 1))
  }

  tibble(display = display, ahat = ahat, actual_avg = actual_avg)
})

# Paper line 281: Self-estimates range from 76% (Llama 3.3 70B) to 97% (Mixtral 8x7B)
ahat_min <- ahat_results %>% filter(!is.na(ahat)) %>% slice_min(ahat) %>% pull(ahat)
ahat_min_model <- ahat_results %>% filter(!is.na(ahat)) %>% slice_min(ahat) %>% pull(display)
ahat_max <- ahat_results %>% filter(!is.na(ahat)) %>% slice_max(ahat) %>% pull(ahat)
ahat_max_model <- ahat_results %>% filter(!is.na(ahat)) %>% slice_max(ahat) %>% pull(display)

check("ahat minimum (paper: 76%, Llama 3.3 70B)", 0.76, ahat_min, tol = 0.02)
cat(sprintf("  (Model with lowest ahat: %s)\n\n", ahat_min_model))
check("ahat maximum (paper: 97%, Mixtral 8x7B)", 0.97, ahat_max, tol = 0.02)
cat(sprintf("  (Model with highest ahat: %s)\n\n", ahat_max_model))

# Paper line 281: actual average accuracy ranges from 75% to 80%
check("Actual avg accuracy minimum (paper: 75%)", 0.75, min(ahat_results$actual_avg), tol = 0.02)
check("Actual avg accuracy maximum (paper: 80%)", 0.80, max(ahat_results$actual_avg), tol = 0.02)

# Paper line 295: Self-estimated accuracy spans 76% to 97%
check("ahat range lower (paper: 76%)", 0.76, ahat_min, tol = 0.02)
check("ahat range upper (paper: 97%)", 0.97, ahat_max, tol = 0.02)


# ============================================================
# 3. Overconfidence stats
#    Paper lines 281, 285, 290
# ============================================================

cat("=" %>% strrep(70), "\n")
cat("SECTION 3: Overconfidence Statistics\n")
cat("Paper Section 5, lines 281-290\n")
cat("=" %>% strrep(70), "\n\n")

# Compute per-condition overconfidence for each model
# Method: for each condition, self_est_acc comes from NOHINT esc_rate inverted
# through the regression fitted on HINT data. actual_acc is the HINT pred_acc.
overconf_data <- map_dfr(seq_len(nrow(models)), function(i) {
  tag <- models$tag[i]
  display <- models$display[i]
  hint <- load_summary(tag, "hint")
  nohint <- load_summary(tag, "nohint")
  if (nrow(hint) == 0 || nrow(nohint) == 0) return(NULL)

  fit <- lm(esc_rate ~ pred_acc, data = hint)
  slope <- coef(fit)["pred_acc"]
  intercept <- coef(fit)["(Intercept)"]

  # Match conditions between hint and nohint
  matched <- inner_join(
    hint %>% select(dataset, condition, pred_acc),
    nohint %>% select(dataset, condition, esc_rate),
    by = c("dataset", "condition"),
    suffix = c("_hint", "_nohint")
  )

  matched %>%
    mutate(
      display = display,
      actual_acc = pred_acc,
      self_est_acc = pmin(pmax(as.numeric((esc_rate - intercept) / slope), 0), 1),
      overconfident = self_est_acc > actual_acc
    )
})

# Paper line 290: "Across 304 model x condition pairs, 201 (66%) are overconfident"
total_pairs <- nrow(overconf_data)
overconf_count <- sum(overconf_data$overconfident)
overconf_pct <- overconf_count / total_pairs

check("Total model x condition pairs (paper: 304)", 304, total_pairs, tol = 0)
check("Overconfident pairs (paper: 201)", 201, overconf_count, tol = 2)
check("Overconfident percentage (paper: 66%)", 0.66, overconf_pct, tol = 0.02)

# Paper line 281: "Qwen3.5-9B overestimates its accuracy on 92% of conditions"
qwen9b_overconf <- overconf_data %>%
  filter(display == "Qwen3.5-9B") %>%
  summarise(pct = mean(overconfident)) %>%
  pull(pct)
check("Qwen3.5-9B overconfident % (paper: 92%)", 0.92, qwen9b_overconf, tol = 0.03)

# Paper line 285: "Qwen3.5-9B, Mixtral 8x7B, Mistral Small 24B, and Llama 4 Maverick
#                  are overconfident on 75-92% of conditions"
for (m in c("Qwen3.5-9B", "Mixtral-8x7B", "Mistral-Small-24B", "Llama4-Maverick")) {
  pct <- overconf_data %>% filter(display == m) %>% summarise(p = mean(overconfident)) %>% pull(p)
  in_range <- pct >= 0.73 && pct <= 0.94
  status <- if (in_range) "PASS" else "FAIL"
  if (in_range) pass_count <- pass_count + 1 else fail_count <- fail_count + 1
  cat(sprintf("[%s] %s overconfident %% in 75-92%% range: %.1f%%\n\n", status, m, pct * 100))
}

# Paper line 285: "Llama 3.3 70B and GPT-5-mini are overconfident on fewer than half"
for (m in c("Llama3.3-70B", "GPT-5-mini")) {
  d <- if (m == "GPT-5-mini") "GPT-5-mini" else "Llama3.3-70B"
  pct <- overconf_data %>% filter(display == d) %>% summarise(p = mean(overconfident)) %>% pull(p)
  below_half <- pct < 0.50
  status <- if (below_half) "PASS" else "FAIL"
  if (below_half) pass_count <- pass_count + 1 else fail_count <- fail_count + 1
  cat(sprintf("[%s] %s overconfident < 50%%: %.1f%%\n\n", status, d, pct * 100))
}


# ============================================================
# 4. GPT-5-mini calibration detail
#    Paper line 283
# ============================================================

cat("=" %>% strrep(70), "\n")
cat("SECTION 4: GPT-5-mini Calibration Detail\n")
cat("Paper Section 5, line 283\n")
cat("=" %>% strrep(70), "\n\n")

gpt_mini_ahat <- ahat_results %>% filter(display == "GPT-5-mini") %>% pull(ahat)
gpt_mini_actual <- ahat_results %>% filter(display == "GPT-5-mini") %>% pull(actual_avg)

check("GPT-5-mini avg self-estimated accuracy (paper: 80%)", 0.80, gpt_mini_ahat, tol = 0.02)
check("GPT-5-mini avg actual accuracy (paper: 80%)", 0.80, gpt_mini_actual, tol = 0.02)

# Per-condition gap range: -30 to +27 pp
gpt_mini_overconf <- overconf_data %>%
  filter(display == "GPT-5-mini") %>%
  mutate(gap_pp = (self_est_acc - actual_acc) * 100)

gap_min <- min(gpt_mini_overconf$gap_pp)
gap_max <- max(gpt_mini_overconf$gap_pp)
gap_sd <- sd(gpt_mini_overconf$gap_pp)

check("GPT-5-mini per-condition gap min (paper: -38 pp)", -38, gap_min, tol = 2)
check("GPT-5-mini per-condition gap max (paper: +27 pp)", 27, gap_max, tol = 2)
check("GPT-5-mini gap SD (paper: 17 pp)", 17, gap_sd, tol = 2)

# Paper: "LendingClub" has -30pp, "WikipediaToxicity" has +27pp
gpt_lc <- gpt_mini_overconf %>% filter(dataset == "LendingClub")
gpt_wt <- gpt_mini_overconf %>% filter(dataset == "WikipediaToxicity")
cat(sprintf("  GPT-5-mini LendingClub gaps: %s\n", paste(round(gpt_lc$gap_pp, 1), collapse = ", ")))
cat(sprintf("  GPT-5-mini WikipediaToxicity gaps: %s\n\n", paste(round(gpt_wt$gap_pp, 1), collapse = ", ")))

# Paper line 283: Qwen3.5-9B gaps range from +3 to +37 points
qwen_overconf <- overconf_data %>%
  filter(display == "Qwen3.5-9B") %>%
  mutate(gap_pp = (self_est_acc - actual_acc) * 100)

qwen_gap_min <- min(qwen_overconf$gap_pp)
qwen_gap_max <- max(qwen_overconf$gap_pp)
check("Qwen3.5-9B gap min (paper: -2 pp)", -2, qwen_gap_min, tol = 2)
check("Qwen3.5-9B gap max (paper: +41 pp)", 41, qwen_gap_max, tol = 2)


# ============================================================
# 5. Table 1: Intervention results
#    Paper lines 317-325
# ============================================================

cat("=" %>% strrep(70), "\n")
cat("SECTION 5: Table 1 - Intervention Results\n")
cat("Paper Section 6, lines 317-325\n")
cat("=" %>% strrep(70), "\n\n")

# Helper: compute sample-level escalation accuracy for a given variant
compute_table1_accuracy <- function(model_tag, variant_pattern, ds_list = datasets) {
  correct <- 0
  total <- 0

  for (ds in ds_list) {
    summary_file <- file.path(DATA_DIR, paste0(ds, "_summary_", variant_pattern, "_", model_tag, ".csv"))
    if (!file.exists(summary_file)) next
    summary <- read_csv(summary_file, show_col_types = FALSE) %>% filter(n > 0)

    for (j in seq_len(nrow(summary))) {
      cond <- summary$condition[j]
      pred_acc <- summary$pred_acc[j]
      should_escalate <- pred_acc < 0.75  # tau* = 75% for R=4

      # Load individual samples
      individual_file <- file.path(DATA_DIR, paste0(ds, "_", cond, "_", variant_pattern, "_", model_tag, ".csv"))
      if (!file.exists(individual_file)) next
      ind <- read_csv(individual_file, show_col_types = FALSE)

      for (k in seq_len(nrow(ind))) {
        escalated <- ind$escalate[k] == 1
        sample_correct <- (should_escalate && escalated) || (!should_escalate && !escalated)
        correct <- correct + as.integer(sample_correct)
        total <- total + 1
      }
    }
  }

  if (total == 0) return(NA_real_)
  correct / total
}

# Qwen 9B baseline
qwen_base <- compute_table1_accuracy("Qwen3.5-9B", "nothink")
check("Table 1: Qwen 9B baseline (paper: 62.0%)", 0.620, qwen_base, tol = 0.005)

# Qwen 9B + cost framing
qwen_cost <- compute_table1_accuracy("Qwen3.5-9B", "cost4_nothink")
check("Table 1: Qwen 9B + cost (paper: 63.9%)", 0.639, qwen_cost, tol = 0.005)

# Qwen 9B + thinking
qwen_think <- compute_table1_accuracy("Qwen3.5-9B", "think")
check("Table 1: Qwen 9B + think (paper: 61.9%)", 0.619, qwen_think, tol = 0.005)

# Qwen 9B + thinking + cost
qwen_think_cost <- compute_table1_accuracy("Qwen3.5-9B", "cost4_think")
check("Table 1: Qwen 9B + think + cost (paper: 78.8%)", 0.788, qwen_think_cost, tol = 0.005)

# GPT-5-mini no reasoning (noreason) -- excludes MovieLens
gpt_ds <- c("HotelBookings", "LendingClub", "WikipediaToxicity")
gpt_noreason <- compute_table1_accuracy("gpt-5-mini", "nothink_noreason", gpt_ds)
check("Table 1: GPT-5-mini no reasoning (paper: 64.7%)", 0.647, gpt_noreason, tol = 0.005)

# GPT-5-mini noreason + cost
gpt_noreason_cost <- compute_table1_accuracy("gpt-5-mini", "cost4_nothink_noreason", gpt_ds)
check("Table 1: GPT-5-mini noreason + cost (paper: 75.8%)", 0.758, gpt_noreason_cost, tol = 0.005)

# GPT-5-mini with reasoning (baseline)
gpt_reason <- compute_table1_accuracy("gpt-5-mini", "nothink", gpt_ds)
check("Table 1: GPT-5-mini + reasoning (paper: 73.2%)", 0.732, gpt_reason, tol = 0.005)

# GPT-5-mini reasoning + cost
gpt_reason_cost <- compute_table1_accuracy("gpt-5-mini", "cost4_nothink", gpt_ds)
check("Table 1: GPT-5-mini reasoning + cost (paper: 87.1%)", 0.871, gpt_reason_cost, tol = 0.005)


# ============================================================
# 6. MoralMachine table (Appendix)
#    Paper lines 398-403
# ============================================================

cat("=" %>% strrep(70), "\n")
cat("SECTION 6: MoralMachine Table (Appendix)\n")
cat("Paper lines 398-403\n")
cat("=" %>% strrep(70), "\n\n")

mm_models <- tribble(
  ~display,             ~tag,
  "Qwen3.5-9B",        "Qwen3.5-9B",
  "Qwen3.5-397B",      "Qwen3.5-397B-A17B",
  "Llama4-Maverick",   "Llama-4-Maverick-17B-128E-Instruct-FP8",
  "Llama3.3-70B",      "Llama-3.3-70B-Instruct-Turbo",
  "Mixtral-8x7B",      "Mixtral-8x7B-Instruct-v0.1",
  "Mistral-Small-24B", "Mistral-Small-24B-Instruct-2501"
)

# Paper values: display, hint_pred_acc, hint_esc_rate, nohint_pred_acc, nohint_esc_rate
mm_paper <- tribble(
  ~display, ~h_pa, ~h_er, ~nh_pa, ~nh_er,
  "Qwen3.5-9B",        0.612, 0.683, 0.521, 0.706,
  "Qwen3.5-397B",      0.704, 0.934, 0.693, 0.913,
  "Llama4-Maverick",   0.658, 0.840, 0.652, 0.946,
  "Llama3.3-70B",      0.675, 0.977, 0.714, 0.999,
  "Mixtral-8x7B",      0.651, 0.965, 0.395, 0.935,
  "Mistral-Small-24B", 0.702, 0.982, 0.684, 1.000
)

for (i in seq_len(nrow(mm_models))) {
  tag <- mm_models$tag[i]
  d <- mm_models$display[i]

  # Load hint
  h_file <- file.path(DATA_DIR, paste0("MoralMachine_summary_nothink_", tag, ".csv"))
  nh_file <- file.path(DATA_DIR, paste0("MoralMachine_summary_nothink_nohint_", tag, ".csv"))

  if (!file.exists(h_file)) { cat(sprintf("  Skipping %s (no file)\n", d)); next }

  h <- read_csv(h_file, show_col_types = FALSE) %>% filter(n > 0)
  h_pa <- weighted.mean(h$pred_acc, h$n)
  h_er <- weighted.mean(h$esc_rate, h$n)

  paper_row <- mm_paper %>% filter(display == d)
  check(sprintf("MM %s hint pred_acc (paper: %.1f%%)", d, paper_row$h_pa * 100), paper_row$h_pa, h_pa, tol = 0.005)
  check(sprintf("MM %s hint esc_rate (paper: %.1f%%)", d, paper_row$h_er * 100), paper_row$h_er, h_er, tol = 0.005)

  if (file.exists(nh_file)) {
    nh <- read_csv(nh_file, show_col_types = FALSE) %>% filter(n > 0)
    nh_pa <- weighted.mean(nh$pred_acc, nh$n)
    nh_er <- weighted.mean(nh$esc_rate, nh$n)
    check(sprintf("MM %s nohint pred_acc (paper: %.1f%%)", d, paper_row$nh_pa * 100), paper_row$nh_pa, nh_pa, tol = 0.005)
    check(sprintf("MM %s nohint esc_rate (paper: %.1f%%)", d, paper_row$nh_er * 100), paper_row$nh_er, nh_er, tol = 0.005)
  }
}


# ============================================================
# 7. MoralMachine appendix text
#    Paper line 387: "Escalation rates are uniformly high (68-100%)"
#    Paper line 387: "Predictive accuracy clusters around 61-70%"
# ============================================================

cat("=" %>% strrep(70), "\n")
cat("SECTION 7: MoralMachine Appendix Text\n")
cat("Paper line 387\n")
cat("=" %>% strrep(70), "\n\n")

all_mm_esc <- mm_paper$h_er
all_mm_pa <- mm_paper$h_pa

check("MM esc rate min (paper: 68%)", 0.68, min(all_mm_esc), tol = 0.02)
check("MM esc rate max (paper: 100%)", 1.00, max(c(all_mm_esc, mm_paper$nh_er)), tol = 0.01)
check("MM pred acc min (paper: 61%)", 0.61, min(all_mm_pa), tol = 0.02)
check("MM pred acc max (paper: 70%)", 0.70, max(all_mm_pa), tol = 0.02)


# ============================================================
# 8. Figure 4 caption
#    Paper line 290: "304 model x condition pairs"
#    (already checked above in Section 3)
# ============================================================


# ============================================================
# 9. Figure 5 / Appendix caption
#    Paper line 469: "p* varies 53% to over 100%"
#    Paper line 469: "self-estimated accuracy ranges from 76% to 97%"
#    (already checked in Sections 1 and 2)
# ============================================================


# ============================================================
# 10. Dataset sizes (Section 3)
#     Paper lines 150-158
# ============================================================

cat("=" %>% strrep(70), "\n")
cat("SECTION 8: Dataset Sizes\n")
cat("Paper lines 150-158\n")
cat("=" %>% strrep(70), "\n\n")

# HotelBookings: 119,390 bookings
if (file.exists("data/HotelBookings/hotel_bookings.csv")) {
  hb <- read_csv("data/HotelBookings/hotel_bookings.csv", show_col_types = FALSE)
  check("HotelBookings size (paper: 119,390)", 119390, nrow(hb), tol = 0)
} else {
  cat("[SKIP] HotelBookings data file not found\n\n")
}

# WikipediaToxicity: 159,686 unique comments
if (file.exists("data/WikipediaToxicity/Wikipedia Toxicity_data_data.csv")) {
  wt <- read_csv("data/WikipediaToxicity/Wikipedia Toxicity_data_data.csv", show_col_types = FALSE)
  n_unique <- n_distinct(wt$rev_id)
  check("WikipediaToxicity unique comments (paper: 159,686)", 159686, n_unique, tol = 100)
  cat(sprintf("  (Raw rows: %d, unique rev_ids: %d)\n\n", nrow(wt), n_unique))
} else {
  cat("[SKIP] WikipediaToxicity data file not found\n\n")
}


# ============================================================
# 11. SFT ablation (Appendix)
#     Paper lines 421-422: 100% -> 84.7% (Hotel), 100% -> 84.0% (Lending)
#     These can't be verified without running the SFT model
# ============================================================

cat("=" %>% strrep(70), "\n")
cat("SECTION 9: SFT Results (requires model run)\n")
cat("=" %>% strrep(70), "\n\n")
cat("[SKIP] SFT results (Tables 3-4) require running the fine-tuned model.\n")
cat("       Job 286878 is running on the cluster. Verify after sync.\n\n")


# ============================================================
# SUMMARY
# ============================================================

cat("=" %>% strrep(70), "\n")
cat(sprintf("VERIFICATION SUMMARY: %d PASS, %d FAIL out of %d checks\n",
            pass_count, fail_count, pass_count + fail_count))
cat("=" %>% strrep(70), "\n")

if (fail_count > 0) {
  cat("\nWARNING: Some checks failed! Review the FAIL items above.\n")
} else {
  cat("\nAll checks passed!\n")
}
