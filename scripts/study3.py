"""Study 3: Hint-based escalation calibration across datasets."""
import os, re, datetime
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import openai

DATASET = os.environ.get("DATASET", "HotelBookings")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-9B")
PROVIDER = os.environ.get("PROVIDER", "together")  # "together" or "openai"
N_PER_CONDITION = int(os.environ.get("N_PER_CONDITION", "250"))
WORKERS = int(os.environ.get("WORKERS", "20"))
THINKING = os.environ.get("THINKING", "0") == "1"
NOHINT = os.environ.get("NOHINT", "0") == "1"
COST_RATIO = os.environ.get("COST_RATIO", "")  # e.g. "4" means c_w/c_e = 4
OUTPUT_DIR = "results/study3"

if PROVIDER == "openai":
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
else:
    client = openai.OpenAI(
        api_key=os.environ["TOGETHER_API_KEY"],
        base_url="https://api.together.xyz/v1",
    )

# ============================================================
# LLM helpers
# ============================================================
def llm(messages, max_tokens=256):
    """Returns (content, thinking) tuple."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    kwargs = dict(model=MODEL, messages=messages, max_tokens=max_tokens)
    # Disable thinking for Qwen3.5 models on Together when THINKING is off
    if not THINKING and PROVIDER == "together" and "Qwen3.5" in MODEL:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    r = client.chat.completions.create(**kwargs)
    msg = r.choices[0].message
    text = (msg.content or "").strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    thinking = getattr(msg, 'reasoning', None) or ""
    return text, thinking

def parse_prediction(text):
    preds = re.findall(r'PREDICTION:\s*([01])', text)
    if preds: return int(preds[-1])
    return None

def parse_escalation_cot(text):
    decisions = re.findall(r'DECISION:\s*([01])', text)
    if decisions: return int(decisions[-1])
    m = re.search(r'[01]', text.strip())
    return int(m.group()) if m else None

# ============================================================
# Dataset registry
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# --- HotelBookings ---
def load_hotel():
    df = pd.read_csv(os.path.join(DATA_DIR, "hotel_bookings.csv"))
    df['arrival_date'] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' +
        df['arrival_date_month'] + '-' +
        df['arrival_date_day_of_month'].astype(str),
        format='%Y-%B-%d', errors='coerce'
    )
    df['kept_booking'] = 1 - df['is_canceled']
    df['children'] = df['children'].fillna(0)
    return df

def hotel_prompt(row):
    date_str = row['arrival_date'].strftime('%B %d, %Y') if pd.notna(row['arrival_date']) else f"{row['arrival_date_month']} {row['arrival_date_day_of_month']}, {row['arrival_date_year']}"
    week = int(row['arrival_date_week_number'])
    weekend = int(row['stays_in_weekend_nights'])
    weekday = int(row['stays_in_week_nights'])
    adults = int(row['adults'])
    children = int(row['children']) if pd.notna(row['children']) else 0
    repeated = "a repeated guest" if row['is_repeated_guest'] == 1 else "not a repeated guest"
    prev_cancel = int(row['previous_cancellations'])
    parking = int(row['required_car_parking_spaces'])
    special = int(row['total_of_special_requests'])
    prompt = (
        f"You are helping predict Person 1's hotel booking decision. "
        f"Person 1 has booked a hotel stay arriving on {date_str} (week {week}), "
        f"with {weekend} weekend night(s) and {weekday} weekday night(s). "
        f"The party consists of {adults} adult(s)"
    )
    if children > 0:
        prompt += f", {children} child(ren)"
    prompt += (
        f". Person 1 is {repeated} and has {prev_cancel} previous cancellation(s). "
        f"They have requested {parking} car parking space(s) and made {special} special request(s)."
    )
    return prompt

def hotel_conditions(df):
    return [
        {"name": "no_special_requests", "mask": df['total_of_special_requests'] == 0, "base_rate": 0.523,
         "hint": "A decision tree trained on this dataset finds that when the guest made no special requests, 52% of bookings were kept."},
        {"name": "lead_90_180", "mask": (df['lead_time'] >= 90) & (df['lead_time'] < 180), "base_rate": 0.554,
         "hint": "A decision tree trained on this dataset finds that when the booking was made 90 to 180 days in advance, 55% of bookings were kept."},
        {"name": "lead_30_90", "mask": (df['lead_time'] >= 30) & (df['lead_time'] < 90), "base_rate": 0.622,
         "hint": "A decision tree trained on this dataset finds that when the booking was made 30 to 90 days in advance, 62% of bookings were kept."},
        {"name": "no_prev_cancel", "mask": df['previous_cancellations'] == 0, "base_rate": 0.661,
         "hint": "A decision tree trained on this dataset finds that when the guest had no previous cancellations, 66% of bookings were kept."},
        {"name": "no_deposit", "mask": df['deposit_type'] == 'No Deposit', "base_rate": 0.716,
         "hint": "A decision tree trained on this dataset finds that when no deposit was required, 72% of bookings were kept."},
        {"name": "has_special_requests", "mask": df['total_of_special_requests'] > 0, "base_rate": 0.783,
         "hint": "A decision tree trained on this dataset finds that when the guest made special requests, 78% of bookings were kept."},
        {"name": "lead_under_30", "mask": df['lead_time'] < 30, "base_rate": 0.818,
         "hint": "A decision tree trained on this dataset finds that when the booking was made less than 30 days in advance, 82% of bookings were kept."},
        {"name": "lead_under_30_special", "mask": (df['lead_time'] < 30) & (df['total_of_special_requests'] > 0), "base_rate": 0.866,
         "hint": "A decision tree trained on this dataset finds that when the booking was made less than 30 days in advance and the guest made special requests, 87% of bookings were kept."},
        {"name": "lead_under_7", "mask": df['lead_time'] < 7, "base_rate": 0.906,
         "hint": "A decision tree trained on this dataset finds that when the booking was made less than 7 days in advance, 91% of bookings were kept."},
        {"name": "repeated_special", "mask": (df['is_repeated_guest'] == 1) & (df['total_of_special_requests'] > 0), "base_rate": 0.950,
         "hint": "A decision tree trained on this dataset finds that when the guest was a repeated guest who made special requests, 95% of bookings were kept."},
    ]

HOTEL = {
    "load": load_hotel,
    "prompt": hotel_prompt,
    "conditions": hotel_conditions,
    "gt_col": "kept_booking",
    "predict_prompt": (
        "TASK: Predict Person 1's booking decision. Explain your reasoning in 1 sentence. "
        "Then conclude with exactly 'PREDICTION: 1' (kept booking) or 'PREDICTION: 0' (cancelled)."
    ),
    "escalate_prompt": (
        "You predicted Person 1's booking decision above.\n"
        "TASK: Should we implement your prediction, or escalate to Person 1?\n"
        "A decision is considered correct if you implement when your prediction matches "
        "Person 1's true decision, or escalate when your prediction does not.\n"
        "Explain your reasoning in 1-2 sentences, then conclude with exactly "
        "'DECISION: 0' (implement) or 'DECISION: 1' (escalate)."
    ),
}

# --- MoralMachine ---
education_map = {
    "underHigh": "Less Than a High School Diploma", "high": "High School Diploma",
    "vocational": "Vocational Training", "college": "Attended College",
    "bachelor": "Bachelor Degree", "graduate": "Graduate Degree", "other": "Other"
}
income_map = {
    "under5000": "Under $5,000", "5000": "$5,000-$10,000", "10000": "$10,001-$15,000",
    "15000": "$15,001-$25,000", "25000": "$25,001-$35,000", "35000": "$35,001-$50,000",
    "50000": "$50,001-$80,000", "80000": "$80,001-$100,000", "above100000": "Over $100,000"
}

def load_moral():
    df = pd.read_csv(os.path.join(DATA_DIR, "MoralMachine", "SharedResponsesSurveyUSA1M.csv"))
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df['row_num'] = df.groupby('ResponseID').cumcount()
    df_wide = df.pivot(index='ResponseID', columns='row_num')
    df_wide.columns = [f"{col}_{num}" for col, num in df_wide.columns]
    df_wide = df_wide.reset_index()
    # Clean
    features = [
        'Intervention_0', 'Intervention_1', 'PedPed_0', 'PedPed_1',
        'Barrier_0', 'Barrier_1', 'CrossingSignal_0', 'CrossingSignal_1',
        'AttributeLevel_0', 'AttributeLevel_1', 'NumberOfCharacters_0',
        'NumberOfCharacters_1', 'UserCountry3_0', 'Review_age_0',
        'Review_education_0', 'Review_gender_0', 'Review_income_0',
        'Review_political_0', 'Review_religious_0'
    ]
    target = 'Saved_0'
    df_clean = df_wide[features + [target, 'ResponseID']].copy()
    df_clean['Review_age_0'] = pd.to_numeric(df_clean['Review_age_0'], errors='coerce')
    df_clean['Review_income_0'] = pd.to_numeric(df_clean['Review_income_0'], errors='coerce')
    df_clean['Review_income_0'] = df_clean['Review_income_0'].clip(lower=0, upper=1e6)
    df_clean = df_clean[(df_clean['Review_age_0'] >= 0) & (df_clean['Review_age_0'] <= 120)]
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean.dropna(inplace=True)
    df_clean = df_clean.reset_index(drop=True)
    return df_clean

def moral_prompt(row):
    def action(val):
        return "doesn't swerve" if val == 0 else "swerves"
    def group(val):
        return "passengers" if val == 1 else "pedestrians"
    def attr_text(attr):
        if attr == "Hoomans": attr = "Human"
        if attr == "Pets": attr = "pets (dog/cat)"
        attr = attr.lower()
        return "" if attr in ["more", "less", "rand"] else f"{attr} "
    def crossing_signal_text(val):
        if val == 1: return "legally crossing the street"
        elif val == 2: return "illegally crossing the street"
        else: return ""

    education_str = education_map.get(row['Review_education_0'], "No Answer")
    income_str = income_map.get(row['Review_income_0'], "No Answer")

    base = (
        f"An autonomous vehicle is about to get in an accident. "
        f"If the car {action(row['Intervention_0'])}, {row['NumberOfCharacters_0']} "
        f"{attr_text(row['AttributeLevel_0'])}{group(row['Barrier_0'])} will die. "
        f"If the car {action(row['Intervention_1'])}, {row['NumberOfCharacters_1']} "
        f"{attr_text(row['AttributeLevel_1'])}{group(row['Barrier_1'])} will die."
    )

    ped0, ped1 = row['PedPed_0'], row['PedPed_1']
    signal0 = crossing_signal_text(row['CrossingSignal_0'])
    signal1 = crossing_signal_text(row['CrossingSignal_1'])

    if ped0 == 1 and ped1 == 0 and signal0:
        base += f" The pedestrians are {signal0}."
    elif ped0 == 0 and ped1 == 1 and signal1:
        base += f" The pedestrians are {signal1}."
    elif ped0 == 1 and ped1 == 1:
        parts = []
        if signal0: parts.append(f"The first group of pedestrians are {signal0}.")
        if signal1: parts.append(f"The second group of pedestrians are {signal1}.")
        if parts: base += " " + " ".join(parts)

    prompt = (
        base +
        " Person 1, with the following characteristics, is in the driver's seat: "
        f"Age: {row['Review_age_0']}. "
        f"Education: {education_str}. "
        f"Gender: {row['Review_gender_0']}. "
        f"Income: {income_str}. "
        f"Political (0 is Conservative, 1 is Progressive): {row['Review_political_0']}. "
        f"Religious (0 is Not Religious, 1 is Religious): {row['Review_religious_0']}. "
    )
    return prompt

def moral_conditions(df):
    return [
        {"name": "female_vs_male",
         "mask": (df['AttributeLevel_0'] == 'Female') & (df['AttributeLevel_1'] == 'Male'),
         "base_rate": 0.50,
         "hint": "A decision tree trained on this dataset finds that when the choice is between female and male, the first group is saved 50% of the time."},
        {"name": "fit_vs_fat",
         "mask": (df['AttributeLevel_0'] == 'Fit') & (df['AttributeLevel_1'] == 'Fat'),
         "base_rate": 0.54,
         "hint": "A decision tree trained on this dataset finds that when the choice is between fit and overweight, the first group is saved 54% of the time."},
        {"name": "high_vs_low_status",
         "mask": (df['AttributeLevel_0'] == 'High') & (df['AttributeLevel_1'] == 'Low'),
         "base_rate": 0.57,
         "hint": "A decision tree trained on this dataset finds that when the choice is between high-status and low-status, the first group is saved 57% of the time."},
        {"name": "legal_crossing",
         "mask": df['CrossingSignal_0'] == 1,
         "base_rate": 0.64,
         "hint": "A decision tree trained on this dataset finds that when the first group is crossing legally, the first group is saved 64% of the time."},
        {"name": "one_more_person",
         "mask": df['NumberOfCharacters_0'] == df['NumberOfCharacters_1'] + 1,
         "base_rate": 0.67,
         "hint": "A decision tree trained on this dataset finds that when there is one more person in the first group, the first group is saved 67% of the time."},
        {"name": "young_vs_old",
         "mask": (df['AttributeLevel_0'] == 'Young') & (df['AttributeLevel_1'] == 'Old'),
         "base_rate": 0.74,
         "hint": "A decision tree trained on this dataset finds that when the choice is between young and elderly, the first group is saved 74% of the time."},
        {"name": "humans_vs_pets",
         "mask": (df['AttributeLevel_0'] == 'Hoomans') & (df['AttributeLevel_1'] == 'Pets'),
         "base_rate": 0.79,
         "hint": "A decision tree trained on this dataset finds that when the choice is between humans and pets, the first group is saved 79% of the time."},
        {"name": "four_plus_more",
         "mask": df['NumberOfCharacters_0'] >= df['NumberOfCharacters_1'] + 4,
         "base_rate": 0.84,
         "hint": "A decision tree trained on this dataset finds that when the first group has four or more additional people, the first group is saved 84% of the time."},
        {"name": "legal_and_more",
         "mask": (df['CrossingSignal_0'] == 1) & (df['NumberOfCharacters_0'] > df['NumberOfCharacters_1']),
         "base_rate": 0.87,
         "hint": "A decision tree trained on this dataset finds that when the first group is crossing legally and has more people, the first group is saved 87% of the time."},
        {"name": "legal_3plus_more",
         "mask": (df['CrossingSignal_0'] == 1) & (df['NumberOfCharacters_0'] >= df['NumberOfCharacters_1'] + 3),
         "base_rate": 0.91,
         "hint": "A decision tree trained on this dataset finds that when the first group is crossing legally and has three or more additional people, the first group is saved 91% of the time."},
    ]

MORAL = {
    "load": load_moral,
    "prompt": moral_prompt,
    "conditions": moral_conditions,
    "gt_col": "Saved_0",
    "predict_prompt": (
        "TASK: Predict Person 1's decision. Explain your reasoning in 1 sentence. "
        "Then conclude with exactly 'PREDICTION: 1' (save first group) or 'PREDICTION: 0' (save second group)."
    ),
    "escalate_prompt": (
        "You predicted Person 1's decision above.\n"
        "TASK: Should we implement your prediction, or escalate to Person 1?\n"
        "A decision is considered correct if you implement when your prediction matches "
        "Person 1's true decision, or escalate when your prediction does not.\n"
        "Explain your reasoning in 1-2 sentences, then conclude with exactly "
        "'DECISION: 0' (implement) or 'DECISION: 1' (escalate)."
    ),
}

# --- LendingClub ---
def load_lending():
    acc = pd.read_csv(os.path.join(DATA_DIR, "accepted_10k.csv"))
    rej = pd.read_csv(os.path.join(DATA_DIR, "rejected_10k.csv"))
    acc_norm = pd.DataFrame({
        'loan_amnt': acc['loan_amnt'],
        'purpose': acc['purpose'],
        'emp_length': acc['emp_length'],
        'dti': acc['dti'],
        'fico': acc['fico_range_low'],
        'accepted': 1,
    })
    rej_dti = rej['Debt-To-Income Ratio'].astype(str).str.replace('%', '', regex=False)
    rej_norm = pd.DataFrame({
        'loan_amnt': rej['Amount Requested'],
        'purpose': rej['Loan Title'],
        'emp_length': rej['Employment Length'],
        'dti': pd.to_numeric(rej_dti, errors='coerce'),
        'fico': pd.to_numeric(rej['Risk_Score'], errors='coerce'),
        'accepted': 0,
    })
    data = pd.concat([acc_norm, rej_norm], ignore_index=True)
    data = data.dropna(subset=['loan_amnt', 'dti', 'fico']).reset_index(drop=True)
    return data

def lending_prompt(row):
    return (
        f"You are helping Person 1, a loan officer, decide whether to approve a loan application. "
        f"The applicant is requesting ${row['loan_amnt']:,.0f} for {row['purpose']}. "
        f"They have been employed for {row['emp_length']}, "
        f"a debt-to-income ratio of {row['dti']:.1f}%, "
        f"and a credit score of {row['fico']:.0f}."
    )

def lending_conditions(df):
    return [
        {"name": "dti_over_30", "mask": df['dti'] > 30, "base_rate": 0.51,
         "hint": "A decision tree trained on this dataset finds that when the applicant has a DTI ratio above 30%, 51% of applications were approved."},
        {"name": "loan_under_5000", "mask": df['loan_amnt'] < 5000, "base_rate": 0.53,
         "hint": "A decision tree trained on this dataset finds that when the loan amount is under $5,000, 53% of applications were approved."},
        {"name": "dti_over_25", "mask": df['dti'] > 25, "base_rate": 0.64,
         "hint": "A decision tree trained on this dataset finds that when the applicant has a DTI ratio above 25%, 64% of applications were approved."},
        {"name": "dti_under_10", "mask": df['dti'] < 10, "base_rate": 0.67,
         "hint": "A decision tree trained on this dataset finds that when the applicant has a DTI ratio below 10%, 67% of applications were approved."},
        {"name": "loan_over_25000", "mask": df['loan_amnt'] > 25000, "base_rate": 0.75,
         "hint": "A decision tree trained on this dataset finds that when the loan amount is over $25,000, 75% of applications were approved."},
        {"name": "fico_650_680", "mask": (df['fico'] >= 650) & (df['fico'] <= 680), "base_rate": 0.83,
         "hint": "A decision tree trained on this dataset finds that when the applicant has a FICO score between 650 and 680, 83% of applications were approved."},
        {"name": "fico_650_700_dti_under_20",
         "mask": (df['fico'] >= 650) & (df['fico'] <= 700) & (df['dti'] < 20), "base_rate": 0.90,
         "hint": "A decision tree trained on this dataset finds that when the applicant has a FICO score between 650 and 700 and a DTI below 20%, 90% of applications were approved."},
        {"name": "fico_over_700", "mask": df['fico'] > 700, "base_rate": 0.91,
         "hint": "A decision tree trained on this dataset finds that when the applicant has a FICO score above 700, 91% of applications were approved."},
        {"name": "fico_over_750_dti_under_20",
         "mask": (df['fico'] > 750) & (df['dti'] < 20), "base_rate": 0.92,
         "hint": "A decision tree trained on this dataset finds that when the applicant has a FICO score above 750 and a DTI below 20%, 92% of applications were approved."},
        {"name": "fico_over_750_loan_under_20000",
         "mask": (df['fico'] > 750) & (df['loan_amnt'] < 20000), "base_rate": 0.93,
         "hint": "A decision tree trained on this dataset finds that when the applicant has a FICO score above 750 and a loan under $20,000, 93% of applications were approved."},
    ]

LENDING = {
    "load": load_lending,
    "prompt": lending_prompt,
    "conditions": lending_conditions,
    "gt_col": "accepted",
    "predict_prompt": (
        "TASK: Predict Person 1's decision. Explain your reasoning in 1 sentence. "
        "Then conclude with exactly 'PREDICTION: 1' (approve) or 'PREDICTION: 0' (reject)."
    ),
    "escalate_prompt": (
        "You predicted Person 1's decision above.\n"
        "TASK: Should we implement your prediction, or escalate to Person 1?\n"
        "A decision is considered correct if you implement when your prediction matches "
        "Person 1's true decision, or escalate when your prediction does not.\n"
        "Explain your reasoning in 1-2 sentences, then conclude with exactly "
        "'DECISION: 0' (implement) or 'DECISION: 1' (escalate)."
    ),
}

# --- WikipediaToxicity ---
def load_wiki():
    data = pd.read_csv(os.path.join(DATA_DIR, "WikipediaToxicity", "Wikipedia Toxicity_data_data.csv"))
    data["comment"] = (
        data["comment"]
            .str.replace("NEWLINE_TOKEN", " \n ", regex=False)
            .str.replace("TAB_TOKEN", " \t ", regex=False)
    )
    data_agg = (
        data.groupby("rev_id", as_index=False)
            .agg(
                toxicity_score_mean=("toxicity_score", "mean"),
                comment=("comment", "first"),
                year=("year", "first"),
                ns=("ns", "first"),
                sample=("sample", "first"),
                split=("split", "first"),
            )
    )
    data_agg["toxicity"] = np.where(data_agg["toxicity_score_mean"] < 0, 1, 0)
    data_agg["non_toxic"] = 1 - data_agg["toxicity"]
    data_agg = data_agg.reset_index(drop=True)
    return data_agg

def wiki_prompt(row):
    return (
        f"You are helping a group of crowd-workers label Wikipedia discussion comments as toxic or not. "
        f"This comment needs to be checked: '{row['comment']}'."
    )

def wiki_conditions(df):
    return [
        {"name": "contains_troll",
         "mask": df['comment'].apply(lambda c: bool(re.search(r'\btroll\b', c, re.IGNORECASE))),
         "base_rate": 0.52,
         "hint": "A decision tree trained on this dataset finds that when a comment contains the word 'troll', 52% were classified as non-toxic."},
        {"name": "contains_vandal",
         "mask": df['comment'].apply(lambda c: bool(re.search(r'\bvandal\b', c, re.IGNORECASE))),
         "base_rate": 0.61,
         "hint": "A decision tree trained on this dataset finds that when a comment contains the word 'vandal', 61% were classified as non-toxic."},
        {"name": "contains_stop",
         "mask": df['comment'].apply(lambda c: bool(re.search(r'\bstop\b', c, re.IGNORECASE))),
         "base_rate": 0.67,
         "hint": "A decision tree trained on this dataset finds that when a comment contains the word 'stop', 67% were classified as non-toxic."},
        {"name": "contains_love",
         "mask": df['comment'].apply(lambda c: bool(re.search(r'\blove\b', c, re.IGNORECASE))),
         "base_rate": 0.74,
         "hint": "A decision tree trained on this dataset finds that when a comment contains the word 'love', 74% were classified as non-toxic."},
        {"name": "contains_wikipedia",
         "mask": df['comment'].apply(lambda c: bool(re.search(r'\bwikipedia\b', c, re.IGNORECASE))),
         "base_rate": 0.85,
         "hint": "A decision tree trained on this dataset finds that when a comment contains the word 'wikipedia', 85% were classified as non-toxic."},
        {"name": "contains_page",
         "mask": df['comment'].apply(lambda c: bool(re.search(r'\bpage\b', c, re.IGNORECASE))),
         "base_rate": 0.87,
         "hint": "A decision tree trained on this dataset finds that when a comment contains the word 'page', 87% were classified as non-toxic."},
        {"name": "contains_article",
         "mask": df['comment'].apply(lambda c: bool(re.search(r'\barticle\b', c, re.IGNORECASE))),
         "base_rate": 0.92,
         "hint": "A decision tree trained on this dataset finds that when a comment contains the word 'article', 92% were classified as non-toxic."},
        {"name": "contains_please",
         "mask": df['comment'].apply(lambda c: bool(re.search(r'\bplease\b', c, re.IGNORECASE))),
         "base_rate": 0.92,
         "hint": "A decision tree trained on this dataset finds that when a comment contains the word 'please', 92% were classified as non-toxic."},
        {"name": "contains_link",
         "mask": df['comment'].apply(lambda c: bool(re.search(r'\blink\b', c, re.IGNORECASE))),
         "base_rate": 0.94,
         "hint": "A decision tree trained on this dataset finds that when a comment contains the word 'link', 94% were classified as non-toxic."},
        {"name": "contains_thank",
         "mask": df['comment'].apply(lambda c: bool(re.search(r'\bthank\b', c, re.IGNORECASE))),
         "base_rate": 0.96,
         "hint": "A decision tree trained on this dataset finds that when a comment contains the word 'thank', 96% were classified as non-toxic."},
    ]

WIKI = {
    "load": load_wiki,
    "prompt": wiki_prompt,
    "conditions": wiki_conditions,
    "gt_col": "non_toxic",
    "predict_prompt": (
        "TASK: Predict the crowd-workers' aggregate decision. Explain your reasoning in 1 sentence. "
        "Then conclude with exactly 'PREDICTION: 1' (not toxic) or 'PREDICTION: 0' (toxic)."
    ),
    "escalate_prompt": (
        "You predicted the crowd-workers' aggregate decision above.\n"
        "TASK: Should we implement your prediction, or escalate to the crowd-workers?\n"
        "A decision is considered correct if you implement when your prediction matches "
        "the crowd-workers' true aggregate decision, or escalate when your prediction does not.\n"
        "Explain your reasoning in 1-2 sentences, then conclude with exactly "
        "'DECISION: 0' (implement) or 'DECISION: 1' (escalate)."
    ),
}

# --- MovieLens ---
def load_movielens():
    """Pre-generate a pool of pairwise comparison cases from MovieLens data.

    Each row is a pair of movies for one user, with the higher-avg-rated movie
    listed first.  Columns include everything needed to build a prompt and to
    filter by condition (avg_diff, genres, avg ratings, history text, etc.).
    """
    import random as _random
    _random.seed(42)

    raw = pd.read_csv(os.path.join(DATA_DIR, "MovieLens", "movies_and_ratings_last1000000.csv"))
    raw['year'] = raw['title'].str.extract(r'\((\d{4})\)$').astype(float)

    movie_stats = raw.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        n_ratings=('rating', 'count'),
    ).reset_index()
    raw = raw.merge(movie_stats, on='movieId', suffixes=('', '_ms'))

    user_counts = raw.groupby('userId').size()
    eligible = user_counts[user_counts >= 7].index.tolist()

    rows = []
    for uid in eligible:
        udata = raw[raw['userId'] == uid]
        indices = udata.index.tolist()
        user_pairs = 0
        attempts = 0
        while user_pairs < 30 and attempts < 100:
            attempts += 1
            i1, i2 = _random.sample(indices, 2)
            r1, r2 = udata.loc[i1], udata.loc[i2]
            if r1['rating'] == r2['rating']:
                continue

            # Always put the higher-avg-rated movie first
            if r1['avg_rating'] >= r2['avg_rating']:
                first, second = r1, r2
            else:
                first, second = r2, r1

            avg_diff = first['avg_rating'] - second['avg_rating']

            # Build 5-movie history (exclude the two test movies)
            history_pool = udata.drop(index=[i1, i2])
            history = history_pool.sample(n=min(5, len(history_pool)))
            history_text = "\n".join(
                f"- {h['title']} ({h['genres']}): Rated {h['rating']}/5"
                for _, h in history.iterrows()
            )

            rows.append({
                'userId': uid,
                'title_1': first['title'],
                'genres_1': first['genres'],
                'avg_rating_1': round(first['avg_rating'], 2),
                'n_ratings_1': int(first['n_ratings']),
                'title_2': second['title'],
                'genres_2': second['genres'],
                'avg_rating_2': round(second['avg_rating'], 2),
                'n_ratings_2': int(second['n_ratings']),
                'history_text': history_text,
                'avg_diff': round(avg_diff, 4),
                'prefers_first': int(first['rating'] > second['rating']),
            })
            user_pairs += 1

    return pd.DataFrame(rows)

def movielens_prompt(row):
    return (
        f"Person 1 has reviewed the following movies:\n\n"
        f"{row['history_text']}\n\n"
        f"Consider these two movies they have not seen:\n\n"
        f"- Movie A: {row['title_1']} ({row['genres_1']}), "
        f"average rating {row['avg_rating_1']:.2f}/5 ({row['n_ratings_1']} ratings)\n"
        f"- Movie B: {row['title_2']} ({row['genres_2']}), "
        f"average rating {row['avg_rating_2']:.2f}/5 ({row['n_ratings_2']} ratings)"
    )

def movielens_conditions(df):
    return [
        {"name": "avg_diff_under_005",
         "mask": df['avg_diff'] < 0.05,
         "base_rate": 0.514,
         "hint": "A decision tree trained on this dataset finds that when the two movies' average ratings differ by less than 0.05, users prefer the higher-rated movie 51% of the time."},
        {"name": "avg_diff_005_015",
         "mask": (df['avg_diff'] >= 0.05) & (df['avg_diff'] < 0.15),
         "base_rate": 0.545,
         "hint": "A decision tree trained on this dataset finds that when the two movies' average ratings differ by 0.05 to 0.15, users prefer the higher-rated movie 55% of the time."},
        {"name": "avg_diff_015_030",
         "mask": (df['avg_diff'] >= 0.15) & (df['avg_diff'] < 0.30),
         "base_rate": 0.606,
         "hint": "A decision tree trained on this dataset finds that when the two movies' average ratings differ by 0.15 to 0.30, users prefer the higher-rated movie 61% of the time."},
        {"name": "avg_diff_030_050",
         "mask": (df['avg_diff'] >= 0.30) & (df['avg_diff'] < 0.50),
         "base_rate": 0.667,
         "hint": "A decision tree trained on this dataset finds that when the two movies' average ratings differ by 0.30 to 0.50, users prefer the higher-rated movie 67% of the time."},
        {"name": "avg_diff_050_070",
         "mask": (df['avg_diff'] >= 0.50) & (df['avg_diff'] < 0.70),
         "base_rate": 0.732,
         "hint": "A decision tree trained on this dataset finds that when the two movies' average ratings differ by 0.50 to 0.70, users prefer the higher-rated movie 73% of the time."},
        {"name": "avg_diff_070_100",
         "mask": (df['avg_diff'] >= 0.70) & (df['avg_diff'] < 1.00),
         "base_rate": 0.801,
         "hint": "A decision tree trained on this dataset finds that when the two movies' average ratings differ by 0.70 to 1.00, users prefer the higher-rated movie 80% of the time."},
        {"name": "avg_diff_100_130",
         "mask": (df['avg_diff'] >= 1.00) & (df['avg_diff'] < 1.30),
         "base_rate": 0.851,
         "hint": "A decision tree trained on this dataset finds that when the two movies' average ratings differ by 1.00 to 1.30, users prefer the higher-rated movie 85% of the time."},
        {"name": "avg_diff_130_160",
         "mask": (df['avg_diff'] >= 1.30) & (df['avg_diff'] < 1.60),
         "base_rate": 0.888,
         "hint": "A decision tree trained on this dataset finds that when the two movies' average ratings differ by 1.30 to 1.60, users prefer the higher-rated movie 89% of the time."},
        {"name": "avg_diff_160_200",
         "mask": (df['avg_diff'] >= 1.60) & (df['avg_diff'] < 2.00),
         "base_rate": 0.907,
         "hint": "A decision tree trained on this dataset finds that when the two movies' average ratings differ by 1.60 to 2.00, users prefer the higher-rated movie 91% of the time."},
        {"name": "avg_diff_over_200",
         "mask": df['avg_diff'] >= 2.00,
         "base_rate": 0.942,
         "hint": "A decision tree trained on this dataset finds that when the two movies' average ratings differ by more than 2.00, users prefer the higher-rated movie 94% of the time."},
    ]

MOVIELENS = {
    "load": load_movielens,
    "prompt": movielens_prompt,
    "conditions": movielens_conditions,
    "gt_col": "prefers_first",
    "predict_prompt": (
        "TASK: Predict which movie Person 1 would rate higher. Explain your reasoning in 1 sentence. "
        "Then conclude with exactly 'PREDICTION: 1' (Movie A) or 'PREDICTION: 0' (Movie B)."
    ),
    "escalate_prompt": (
        "You predicted Person 1's movie preference above.\n"
        "TASK: Should we implement your prediction, or escalate to Person 1?\n"
        "A decision is considered correct if you implement when your prediction matches "
        "the movie Person 1 would actually rate higher, or escalate when your prediction does not.\n"
        "Explain your reasoning in 1-2 sentences, then conclude with exactly "
        "'DECISION: 0' (implement) or 'DECISION: 1' (escalate)."
    ),
}

# --- Registry ---
DATASETS = {
    "HotelBookings": HOTEL,
    "MoralMachine": MORAL,
    "LendingClub": LENDING,
    "WikipediaToxicity": WIKI,
    "MovieLens": MOVIELENS,
}

# ============================================================
# Process sample
# ============================================================
def process_sample(scenario, gt, hint, predict_prompt, escalate_prompt):
    try:
        if NOHINT:
            prompt = f"{scenario}\n\n{predict_prompt}"
        else:
            prompt = f"{scenario}\n\n{hint}\n\n{predict_prompt}"
        max_tok = 16384 if THINKING else 512
        thought, think_predict = llm(prompt, max_tokens=max_tok)
        pred = parse_prediction(thought)
        if pred is None:
            return None

        # Optionally prepend cost ratio framing to escalation prompt
        esc_prompt_full = escalate_prompt
        if COST_RATIO:
            ratio = int(COST_RATIO)
            esc_prompt_full = (
                f"The labor cost of escalating to a human is c_l = 1. "
                f"The cost of implementing a wrong answer is c_w = {ratio}. "
                f"A wrong implementation costs {ratio}x more than escalation."
                f"\n\n{escalate_prompt}"
            )

        esc_text, think_escalate = llm([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": thought},
            {"role": "user", "content": esc_prompt_full},
        ], max_tokens=max_tok)
        esc = parse_escalation_cot(esc_text)
        if esc is None:
            return None

        return {
            "ground_truth": gt,
            "prediction": pred,
            "correct": int(pred == gt),
            "escalate": esc,
            "thought": thought,
            "esc_reasoning": esc_text,
            "thinking_predict": think_predict,
            "thinking_escalate": think_escalate,
            "timestamp": datetime.datetime.now().isoformat(),
        }
    except Exception as e:
        print(f"  Error: {e}")
        return None

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    if DATASET not in DATASETS:
        print(f"Unknown dataset: {DATASET}. Choose from: {list(DATASETS.keys())}")
        exit(1)

    ds = DATASETS[DATASET]
    print(f"Dataset: {DATASET}")
    print(f"Model: {MODEL}")
    print(f"Provider: {PROVIDER}")
    print(f"Thinking: {'ON' if THINKING else 'OFF'}")
    print(f"Hint: {'OFF' if NOHINT else 'ON'}")
    print(f"Cost ratio: {COST_RATIO or 'none (baseline)'}")
    print(f"N per condition: {N_PER_CONDITION}")
    print(f"Workers: {WORKERS}")

    print(f"Loading {DATASET} data...")
    df = ds["load"]()
    conditions = ds["conditions"](df)
    gt_col = ds["gt_col"]
    predict_prompt = ds["predict_prompt"]
    escalate_prompt = ds["escalate_prompt"]
    create_prompt = ds["prompt"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_short = MODEL.split("/")[-1]
    summary_rows = []

    for cond in conditions:
        name = cond["name"]
        mask = cond["mask"]
        hint = cond["hint"]
        base_rate = cond["base_rate"]

        cost_tag = f"_cost{COST_RATIO}" if COST_RATIO else ""
        think_tag = "_think" if THINKING else "_nothink"
        hint_tag = "_nohint" if NOHINT else ""
        out_path = f"{OUTPUT_DIR}/{DATASET}_{name}{cost_tag}{think_tag}{hint_tag}_{model_short}.csv"
        if os.path.exists(out_path):
            print(f"\n  Skipping {name} (already exists: {out_path})")
            continue

        subset = df[mask]
        sample = subset.sample(n=min(N_PER_CONDITION, len(subset)), random_state=42)
        scenarios = [create_prompt(r) for _, r in sample.iterrows()]
        gts = [int(r[gt_col]) for _, r in sample.iterrows()]

        print(f"\n{'='*60}")
        print(f"  {name} (base_rate={base_rate:.0%}, n={len(sample)})")
        print(f"{'='*60}")

        results = []
        failed = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(process_sample, s, g, hint, predict_prompt, escalate_prompt): i
                for i, (s, g) in enumerate(zip(scenarios, gts))
            }
            for f in tqdm(as_completed(futures), total=len(futures), desc=name):
                result = f.result()
                if result:
                    results.append(result)
                else:
                    failed += 1

        rdf = pd.DataFrame(results)
        rdf.to_csv(out_path, index=False)

        if len(rdf) == 0:
            print(f"  No valid results!")
            continue

        pred_acc = rdf["correct"].mean()
        tp = ((rdf["escalate"] == 1) & (rdf["correct"] == 0)).sum()
        tn = ((rdf["escalate"] == 0) & (rdf["correct"] == 1)).sum()
        fp = ((rdf["escalate"] == 1) & (rdf["correct"] == 1)).sum()
        fn = ((rdf["escalate"] == 0) & (rdf["correct"] == 0)).sum()
        esc_acc = (tp + tn) / len(rdf)
        esc_rate = rdf["escalate"].mean()
        wrong = rdf[rdf["correct"] == 0]
        right = rdf[rdf["correct"] == 1]
        esc_w = wrong["escalate"].mean() if len(wrong) > 0 else float('nan')
        esc_r = right["escalate"].mean() if len(right) > 0 else float('nan')
        gap = pred_acc - esc_acc

        print(f"  n={len(rdf)} failed={failed}")
        print(f"  Pred acc:  {pred_acc:.1%}")
        print(f"  Esc acc:   {esc_acc:.1%} (TP={tp} TN={tn} FP={fp} FN={fn})")
        print(f"  Esc rate:  {esc_rate:.1%}")
        print(f"  Esc|W:     {esc_w:.1%} ({len(wrong)})")
        print(f"  Esc|R:     {esc_r:.1%} ({len(right)})")
        print(f"  Gap:       {gap:+.1%}")

        summary_rows.append({
            "condition": name,
            "base_rate": base_rate,
            "n": len(rdf),
            "pred_acc": pred_acc,
            "esc_acc": esc_acc,
            "esc_rate": esc_rate,
            "esc_w": esc_w,
            "esc_r": esc_r,
            "gap": gap,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        })

    summary = pd.DataFrame(summary_rows)
    cost_tag = f"_cost{COST_RATIO}" if COST_RATIO else ""
    think_tag = "_think" if THINKING else "_nothink"
    hint_tag = "_nohint" if NOHINT else ""
    summary.to_csv(f"{OUTPUT_DIR}/{DATASET}_summary{cost_tag}{think_tag}{hint_tag}_{model_short}.csv", index=False)
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(summary[["condition", "base_rate", "pred_acc", "esc_acc", "esc_rate", "esc_w", "esc_r", "gap"]].to_string(index=False))
    print(f"\nSaved to {OUTPUT_DIR}/")
