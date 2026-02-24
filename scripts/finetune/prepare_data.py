#!/usr/bin/env python3
"""
Prepare fine-tuning JSONL data for each dataset.

For each dataset, samples 100 examples from the training split (or a fixed
random subset for datasets without an explicit split), builds the base prompt
using the same logic as the corresponding run_*.py script, and writes
one JSONL file per dataset to finetune/data/.

Also saves the used indices to finetune/data/{dataset}_ft_indices.json so
they can be excluded from evaluation runs if needed.
"""

import os, json, random, ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "../../data")
OUT_DIR    = os.path.join(SCRIPT_DIR, "../../finetune/data")
os.makedirs(OUT_DIR, exist_ok=True)

N_FT = 100
random.seed(42)
np.random.seed(42)


def write_jsonl(path, examples):
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Wrote {len(examples)} examples -> {path}")


def save_indices(path, indices):
    with open(path, "w") as f:
        json.dump([int(i) for i in indices], f)


def ft_example(prompt, label):
    return {"messages": [
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": str(label)},
    ]}


# ============================================================
# LendingClub
# ============================================================
print("\n=== LendingClub ===")
acc = pd.read_csv(os.path.join(DATA_DIR, "accepted_10k.csv"))
rej = pd.read_csv(os.path.join(DATA_DIR, "rejected_10k.csv"))

acc_norm = pd.DataFrame({
    "loan_amnt":  acc["loan_amnt"],
    "purpose":    acc["purpose"],
    "emp_length": acc["emp_length"],
    "dti":        acc["dti"],
    "fico":       acc["fico_range_low"],
    "accepted":   1,
})
rej_dti = rej["Debt-To-Income Ratio"].astype(str).str.replace("%", "", regex=False)
rej_norm = pd.DataFrame({
    "loan_amnt":  rej["Amount Requested"],
    "purpose":    rej["Loan Title"],
    "emp_length": rej["Employment Length"],
    "dti":        pd.to_numeric(rej_dti, errors="coerce"),
    "fico":       pd.to_numeric(rej["Risk_Score"], errors="coerce"),
    "accepted":   0,
})
lc_data = pd.concat([acc_norm, rej_norm], ignore_index=True)
lc_data = lc_data.dropna(subset=["loan_amnt", "dti", "fico"]).reset_index(drop=True)

train_idx, _ = train_test_split(lc_data.index, test_size=0.2, random_state=42)
ft_idx = random.sample(list(train_idx), N_FT)

examples = []
for idx in ft_idx:
    row = lc_data.loc[idx]
    prompt = (
        f"You are helping Person 1, a loan officer, decide whether to approve a loan application. "
        f"The applicant is requesting ${row['loan_amnt']:,.0f} for {row['purpose']}. "
        f"They have been employed for {row['emp_length']}, "
        f"a debt-to-income ratio of {row['dti']:.1f}%, "
        f"and a credit score of {row['fico']:.0f}."
    )
    examples.append(ft_example(prompt, int(row["accepted"])))

write_jsonl(os.path.join(OUT_DIR, "LendingClub.jsonl"), examples)
save_indices(os.path.join(OUT_DIR, "LendingClub_ft_indices.json"), ft_idx)


# ============================================================
# HotelBookings
# ============================================================
print("\n=== HotelBookings ===")
hb_data = pd.read_csv(os.path.join(DATA_DIR, "hotel_bookings.csv"))
hb_data = hb_data.dropna(subset=[
    "is_canceled", "arrival_date_year", "arrival_date_month",
    "arrival_date_week_number", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "is_repeated_guest", "previous_cancellations",
    "required_car_parking_spaces", "total_of_special_requests",
])
hb_data["children"] = hb_data["children"].fillna(0).astype(int)
hb_data["babies"]   = hb_data["babies"].fillna(0).astype(int)
hb_data = hb_data.reset_index(drop=True)

train_idx, _ = train_test_split(hb_data.index, test_size=0.2, random_state=42)
ft_idx = random.sample(list(train_idx), N_FT)

examples = []
for idx in ft_idx:
    row = hb_data.loc[idx]
    party = f"{int(row['adults'])} adult(s)"
    if int(row["children"]) > 0:
        party += f", {int(row['children'])} child(ren)"
    if int(row["babies"]) > 0:
        party += f", {int(row['babies'])} baby/babies"
    repeated     = "a repeated guest" if row["is_repeated_guest"] == 1 else "not a repeated guest"
    prev_cancel  = int(row["previous_cancellations"])
    parking      = int(row["required_car_parking_spaces"])
    special      = int(row["total_of_special_requests"])
    prompt = (
        f"You are helping predict Person 1's hotel booking decision. "
        f"Person 1 has booked a hotel stay arriving on {row['arrival_date_month']} {int(row['arrival_date_day_of_month'])}, "
        f"{int(row['arrival_date_year'])} (week {int(row['arrival_date_week_number'])}), "
        f"with {int(row['stays_in_weekend_nights'])} weekend night(s) and {int(row['stays_in_week_nights'])} weekday night(s). "
        f"The party consists of {party}. "
        f"Person 1 is {repeated} and has {prev_cancel} previous cancellation(s). "
        f"They have requested {parking} car parking space(s) and made {special} special request(s)."
    )
    examples.append(ft_example(prompt, int(row["is_canceled"])))

write_jsonl(os.path.join(OUT_DIR, "HotelBookings.jsonl"), examples)
save_indices(os.path.join(OUT_DIR, "HotelBookings_ft_indices.json"), ft_idx)


# ============================================================
# MoralMachine
# ============================================================
print("\n=== MoralMachine ===")
mm_df = pd.read_csv(os.path.join(DATA_DIR, "MoralMachine/SharedResponsesSurveyUSA1M.csv"))
if "Unnamed: 0" in mm_df.columns:
    mm_df = mm_df.drop(columns=["Unnamed: 0"])
mm_df["row_num"] = mm_df.groupby("ResponseID").cumcount()
mm_wide = mm_df.pivot(index="ResponseID", columns="row_num")
mm_wide.columns = [f"{col}_{num}" for col, num in mm_wide.columns]
mm_wide = mm_wide.reset_index()

features = [
    "Intervention_0", "Intervention_1", "PedPed_0", "PedPed_1",
    "Barrier_0", "Barrier_1", "CrossingSignal_0", "CrossingSignal_1",
    "AttributeLevel_0", "AttributeLevel_1", "NumberOfCharacters_0",
    "NumberOfCharacters_1", "UserCountry3_0", "Review_age_0",
    "Review_education_0", "Review_gender_0", "Review_income_0",
    "Review_political_0", "Review_religious_0",
]
target = "Saved_0"
mm_clean = mm_wide[features + [target]].copy()
mm_clean["Review_age_0"]    = pd.to_numeric(mm_clean["Review_age_0"], errors="coerce")
mm_clean["Review_income_0"] = pd.to_numeric(mm_clean["Review_income_0"], errors="coerce")
mm_clean["Review_income_0"] = mm_clean["Review_income_0"].clip(lower=0, upper=1e6)
mm_clean = mm_clean[(mm_clean["Review_age_0"] >= 0) & (mm_clean["Review_age_0"] <= 120)]
mm_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
mm_clean.dropna(inplace=True)

train_idx, _ = train_test_split(mm_clean.index, test_size=0.2, random_state=42)
ft_idx = random.sample(list(train_idx), N_FT)

education_map = {
    "underHigh": "Less Than a High School Diploma", "high": "High School Diploma",
    "vocational": "Vocational Training", "college": "Attended College",
    "bachelor": "Bachelor Degree", "graduate": "Graduate Degree", "other": "Other",
}
income_map = {
    "under5000": "Under $5,000", "5000": "$5,000–$10,000", "10000": "$10,001–$15,000",
    "15000": "$15,001–$25,000", "25000": "$25,001–$35,000", "35000": "$35,001–$50,000",
    "50000": "$50,001–$80,000", "80000": "$80,001–$100,000", "above100000": "Over $100,000",
}

def mm_prompt(row):
    def action(val):       return "doesn't swerve" if val == 0 else "swerves"
    def group(val):        return "passengers" if val == 1 else "pedestrians"
    def attr_text(attr):
        if attr == "Hoomans": attr = "Human"
        if attr == "Pets":    attr = "pets (dog/cat)"
        attr = attr.lower()
        return "" if attr in ["more", "less", "rand"] else f"{attr} "
    def crossing_signal_text(val):
        if val == 1:   return "legally crossing the street"
        elif val == 2: return "illegally crossing the street"
        else:          return ""

    base = (
        f"An autonomous vehicle is about to get in an accident. "
        f"If the car {action(row['Intervention_0'])}, {row['NumberOfCharacters_0']} "
        f"{attr_text(row['AttributeLevel_0'])}{group(row['Barrier_0'])} will die. "
        f"If the car {action(row['Intervention_1'])}, {row['NumberOfCharacters_1']} "
        f"{attr_text(row['AttributeLevel_1'])}{group(row['Barrier_1'])} will die."
    )
    ped0, ped1   = row["PedPed_0"], row["PedPed_1"]
    signal0      = crossing_signal_text(row["CrossingSignal_0"])
    signal1      = crossing_signal_text(row["CrossingSignal_1"])
    if ped0 == 1 and ped1 == 0 and signal0:
        base += f" The pedestrians are {signal0}."
    elif ped0 == 0 and ped1 == 1 and signal1:
        base += f" The pedestrians are {signal1}."
    elif ped0 == 1 and ped1 == 1:
        parts = []
        if signal0: parts.append(f"The first group of pedestrians are {signal0}.")
        if signal1: parts.append(f"The second group of pedestrians are {signal1}.")
        if parts: base += " " + " ".join(parts)

    education_str = education_map.get(row["Review_education_0"], "No Answer")
    income_str    = income_map.get(row["Review_income_0"], "No Answer")
    return (
        base +
        " Person 1, with the following characteristics, is in the driver's seat: "
        f"Age: {row['Review_age_0']}. "
        f"Education: {education_str}. "
        f"Gender: {row['Review_gender_0']}. "
        f"Income: {income_str}. "
        f"Political (0 is Conservative, 1 is Progressive): {row['Review_political_0']}. "
        f"Religious (0 is Not Religious, 1 is Religious): {row['Review_religious_0']}. "
    )

examples = []
for idx in ft_idx:
    row = mm_clean.loc[idx]
    examples.append(ft_example(mm_prompt(row), int(row[target])))

write_jsonl(os.path.join(OUT_DIR, "MoralMachine.jsonl"), examples)
save_indices(os.path.join(OUT_DIR, "MoralMachine_ft_indices.json"), ft_idx)


# ============================================================
# Uber
# ============================================================
print("\n=== Uber ===")
ub_data = pd.read_csv(os.path.join(DATA_DIR, "Bookings.csv"))
ub_data = ub_data[ub_data["Booking_Status"].isin(["Success", "Canceled by Driver"])].copy()
ub_data["cancelled"] = (ub_data["Booking_Status"] == "Canceled by Driver").astype(int)
ub_data = ub_data.dropna(subset=["Pickup_Location", "Drop_Location"])
ub_data = ub_data.reset_index(drop=True)

train_idx, _ = train_test_split(ub_data.index, test_size=0.2, random_state=42)
ft_idx = random.sample(list(train_idx), N_FT)

examples = []
for idx in ft_idx:
    row = ub_data.loc[idx]
    prompt = (
        f"You are helping Person 1 with a ride decision. "
        f"Person 1 has received a ride request going from {row['Pickup_Location']} to {row['Drop_Location']}."
    )
    examples.append(ft_example(prompt, int(row["cancelled"])))

write_jsonl(os.path.join(OUT_DIR, "Uber.jsonl"), examples)
save_indices(os.path.join(OUT_DIR, "Uber_ft_indices.json"), ft_idx)


# ============================================================
# MovieLens
# ============================================================
print("\n=== MovieLens ===")
ml_df = pd.read_csv(os.path.join(DATA_DIR, "MovieLens/movies_and_ratings_last1000000.csv"))
_, test_df = train_test_split(ml_df, test_size=0.2, random_state=42)
# Use only train-split users so no overlap with evaluation data
train_user_ids = set(ml_df.index.difference(test_df.index).map(lambda i: ml_df.iloc[i]["userId"]))
# Actually use train rows directly
train_ml = ml_df.drop(index=test_df.index).reset_index(drop=True)

examples = []
attempts = 0
rng = random.Random(42)
train_user_list = list(train_ml["userId"].unique())
rng.shuffle(train_user_list)

for user_id in train_user_list:
    if len(examples) >= N_FT:
        break
    user_data = train_ml[train_ml["userId"] == user_id].copy()
    if len(user_data) < 7:
        continue
    shuffled = user_data.sample(frac=1, random_state=attempts)
    attempts += 1
    movie_1 = shuffled.iloc[0]
    movie_2 = None
    used_indices = [0]
    for i in range(1, len(shuffled)):
        if shuffled.iloc[i]["rating"] != movie_1["rating"]:
            movie_2 = shuffled.iloc[i]
            used_indices.append(i)
            break
    if movie_2 is None:
        continue
    history = shuffled.drop(shuffled.index[used_indices]).head(5)
    prompt = "Person 1 has reviewed the following movies:\n\n"
    for _, r in history.iterrows():
        prompt += f"- {r['title']} ({r['genres']}): Rated {r['rating']}/5\n"
    prompt += "\nConsider these two movies they have not seen:\n\n"
    test_pair = [movie_1, movie_2]
    rng.shuffle(test_pair)
    for movie in test_pair:
        prompt += f"- {movie['title']} ({movie['genres']})\n"
    label = "1" if test_pair[0]["rating"] > test_pair[1]["rating"] else "2"
    examples.append(ft_example(prompt, label))

write_jsonl(os.path.join(OUT_DIR, "MovieLens.jsonl"), examples)
# No index tracking needed; restricted to train users
save_indices(os.path.join(OUT_DIR, "MovieLens_ft_indices.json"), [])


# ============================================================
# AIME
# ============================================================
print("\n=== AIME ===")
aime_df = pd.read_csv(os.path.join(DATA_DIR, "AIME_Dataset_1983_2024.csv"))
aime_df = aime_df.reset_index(drop=True)

ft_idx = random.sample(list(aime_df.index), min(N_FT, len(aime_df)))

examples = []
for idx in ft_idx:
    row = aime_df.iloc[idx]
    prompt = f"You are helping Person 1 solve the following math problem: {row['Question']}."
    examples.append(ft_example(prompt, str(int(row["Answer"]))))

write_jsonl(os.path.join(OUT_DIR, "AIME.jsonl"), examples)
save_indices(os.path.join(OUT_DIR, "AIME_ft_indices.json"), ft_idx)


# ============================================================
# JFLEG
# ============================================================
print("\n=== JFLEG ===")
jfleg_data = pd.read_csv(os.path.join(DATA_DIR, "JFLEG/JFLEG.csv"))
jfleg_data = jfleg_data.reset_index(drop=True)

ft_idx = random.sample(list(jfleg_data.index), min(N_FT, len(jfleg_data)))

examples = []
for idx in ft_idx:
    row = jfleg_data.iloc[idx]
    prompt = (
        f"You are predicting how Person 1 would correct a grammatically incorrect statement. "
        f"This statement needs to be checked: '{row['sentence']}'"
    )
    # corrections is stored as a stringified list; take the first correction
    corrections_raw = row["corrections"]
    try:
        corrections = ast.literal_eval(corrections_raw)
        label = corrections[0] if isinstance(corrections, list) and corrections else str(corrections_raw)
    except Exception:
        label = str(corrections_raw)
    examples.append(ft_example(prompt, label))

write_jsonl(os.path.join(OUT_DIR, "JFLEG.jsonl"), examples)
save_indices(os.path.join(OUT_DIR, "JFLEG_ft_indices.json"), ft_idx)


# ============================================================
# FEVEROUS
# ============================================================
print("\n=== FEVEROUS ===")
fev_data = pd.read_json(os.path.join(DATA_DIR, "FEVEROUS/feverous_train_challenges.jsonl"), lines=True)
fev_data = fev_data.replace("", pd.NA).dropna(how="all")
fev_data = fev_data[fev_data["label"] != "NOT ENOUGH INFO"]
fev_data["supports"] = fev_data["label"].map({"SUPPORTS": 1, "REFUTES": 0})
fev_data = fev_data.reset_index(drop=True)

ft_idx = random.sample(list(fev_data.index), min(N_FT, len(fev_data)))

examples = []
for idx in ft_idx:
    row = fev_data.iloc[idx]
    prompt = (
        f"You are helping Person 1 with fact-checking. "
        f"This statement needs to be checked: {row['claim']}."
    )
    examples.append(ft_example(prompt, int(row["supports"])))

write_jsonl(os.path.join(OUT_DIR, "FEVEROUS.jsonl"), examples)
save_indices(os.path.join(OUT_DIR, "FEVEROUS_ft_indices.json"), ft_idx)


# ============================================================
# WikipediaToxicity
# ============================================================
print("\n=== WikipediaToxicity ===")
wt_data = pd.read_csv(os.path.join(DATA_DIR, "WikipediaToxicity/Wikipedia Toxicity_data_data.csv"))
wt_data["comment"] = (
    wt_data["comment"]
    .str.replace("NEWLINE_TOKEN", " \n ", regex=False)
    .str.replace("TAB_TOKEN", " \t ", regex=False)
)
wt_agg = (
    wt_data.groupby("rev_id", as_index=False)
    .agg(
        toxicity_score_mean=("toxicity_score", "mean"),
        comment=("comment", "first"),
    )
)
wt_agg["toxicity"] = np.where(wt_agg["toxicity_score_mean"] < 0, 1, 0)
wt_agg = wt_agg.reset_index(drop=True)

ft_idx = random.sample(list(wt_agg.index), min(N_FT, len(wt_agg)))

examples = []
for idx in ft_idx:
    row = wt_agg.iloc[idx]
    prompt = (
        f"You are helping a group of crowd-workers label Wikipedia discussion comments as toxic or not. "
        f"This comment needs to be checked: '{row['comment']}'."
    )
    examples.append(ft_example(prompt, int(row["toxicity"])))

write_jsonl(os.path.join(OUT_DIR, "WikipediaToxicity.jsonl"), examples)
save_indices(os.path.join(OUT_DIR, "WikipediaToxicity_ft_indices.json"), ft_idx)


print("\nDone. All datasets written to finetune/data/")
