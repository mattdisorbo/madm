"""
Unified Together API runner for Studies 1 & 2.

Study 1 (context bias decomposition): blind, injected, chained
Study 2 (reasoning mitigation): blind, chained, reasoning_only, reasoning_prediction

Usage:
    TOGETHER_API_KEY=... python scripts/run_together.py [--datasets ALL] [--methods ALL] [--n 100] [--workers 20]
"""

import os, re, sys, argparse, datetime, threading, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ── Config ──
MODEL = os.environ.get("TOGETHER_MODEL", "Qwen/Qwen3.5-9B")
MODEL_SHORT = MODEL.split("/")[-1]

client = OpenAI(
    api_key=os.environ.get("TOGETHER_API_KEY", ""),
    base_url="https://api.together.xyz/v1",
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../results")

ALL_METHODS = ["blind", "injected", "chained", "reasoning_only", "reasoning_prediction"]

# ── LLM helpers ──

def strip_think(text):
    if text is None:
        return ""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def llm_single(prompt, max_tokens=256):
    """Fresh single-message call."""
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return strip_think(r.choices[0].message.content)

def llm_multi(messages, max_tokens=8):
    """Multi-turn call (for chained conditions)."""
    r = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return strip_think(r.choices[0].message.content)

# ── Parsing ──

def parse_prediction_binary(text):
    preds = re.findall(r'PREDICTION:\s*([01])', text)
    if preds:
        return int(preds[-1])
    digits = re.findall(r'[01]', text.strip())
    if digits:
        return int(digits[0])
    return None

def parse_prediction_pairwise(text):
    preds = re.findall(r'PREDICTION:\s*([12])', text)
    if preds:
        return int(preds[-1])
    digits = re.findall(r'[12]', text.strip())
    if digits:
        return int(digits[0])
    return None

def parse_prediction_aime(text):
    preds = re.findall(r'SOLUTION:\s*(\d+)', text)
    if preds:
        return preds[-1]
    preds = re.findall(r'\\boxed\{(\d+)\}', text)
    if preds:
        return preds[-1]
    preds = re.findall(r'\$\s*(\d+)\s*\$\s*$', text, re.MULTILINE)
    if preds:
        return preds[-1]
    preds = re.findall(r'(?:answer|result|equal(?:s)?|is)\s*(?:is\s*)?[\$\s]*(\d+)', text, re.IGNORECASE)
    if preds:
        return preds[-1]
    preds = re.findall(r'(\d+)', text.strip())
    if preds:
        return preds[-1]
    return None

def parse_decision(text):
    match = re.search(r'[01]', text.strip())
    if match:
        return int(match.group())
    low = text.lower()
    if 'implement' in low:
        return 0
    if 'escalat' in low:
        return 1
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Dataset Registry
# ══════════════════════════════════════════════════════════════════════════════

class Dataset:
    """Base class for dataset-specific logic."""
    name = None

    # Prompt templates (overridden per dataset)
    thought_prompt = (
        "TASK: Predict Person 1's decision. Explain your reasoning in 1 sentence. "
        "Then conclude with exactly 'PREDICTION: 1' ({pos}) or 'PREDICTION: 0' ({neg})."
    )
    decision_task = (
        "TASK: Respond with a digit ONLY.\n"
        "Output 0 to implement your prediction, or 1 to escalate to Person 1.\n"
        "A decision is considered correct if it equals Person 1's true decision."
    )
    escalate_only_task = (
        "An LLM was asked to predict Person 1's decision in this scenario.\n"
        "TASK: Respond with a digit ONLY.\n"
        "Output 0 to implement the LLM's prediction, or 1 to escalate to Person 1.\n"
        "A decision is considered correct if it equals Person 1's true decision."
    )
    injected_decision_task = (
        "A separate model predicted PREDICTION: {pred} for Person 1's decision in this scenario.\n"
        "TASK: Respond with a digit ONLY.\n"
        "Output 0 to implement the model's prediction, or 1 to escalate to Person 1.\n"
        "A decision is considered correct if it equals Person 1's true decision."
    )
    analyze_prompt = (
        "TASK: Analyze this case. Discuss the relevant features and flag any uncertainties. "
        "Do not make a prediction."
    )

    parse_prediction = staticmethod(parse_prediction_binary)

    def load(self):
        raise NotImplementedError

    def create_prompt_base(self, row):
        raise NotImplementedError

    def get_human_response(self, row):
        raise NotImplementedError

    def make_result_row(self, row, row_idx, base_prompt):
        """Return dataset-specific columns for the result CSV."""
        return {
            'index': row_idx,
            'human_response': self.get_human_response(row),
            'prompt': base_prompt,
        }


class LendingClub(Dataset):
    name = "LendingClub"
    thought_prompt = (
        "TASK: Predict Person 1's decision. Explain your reasoning in 1 sentence. "
        "Then conclude with exactly 'PREDICTION: 1' (approve) or 'PREDICTION: 0' (reject)."
    )

    def load(self):
        acc = pd.read_csv(os.path.join(DATA_DIR, "accepted_10k.csv"))
        rej = pd.read_csv(os.path.join(DATA_DIR, "rejected_10k.csv"))
        acc_norm = pd.DataFrame({
            'loan_amnt': acc['loan_amnt'], 'purpose': acc['purpose'],
            'emp_length': acc['emp_length'], 'dti': acc['dti'],
            'fico': acc['fico_range_low'], 'accepted': 1,
        })
        rej_dti = rej['Debt-To-Income Ratio'].astype(str).str.replace('%', '', regex=False)
        rej_norm = pd.DataFrame({
            'loan_amnt': rej['Amount Requested'], 'purpose': rej['Loan Title'],
            'emp_length': rej['Employment Length'],
            'dti': pd.to_numeric(rej_dti, errors='coerce'),
            'fico': pd.to_numeric(rej['Risk_Score'], errors='coerce'), 'accepted': 0,
        })
        data = pd.concat([acc_norm, rej_norm], ignore_index=True)
        data = data.dropna(subset=['loan_amnt', 'dti', 'fico']).reset_index(drop=True)
        features = ['loan_amnt', 'dti', 'fico']
        _, holdout_idx = train_test_split(data.index, test_size=0.2, random_state=42)
        self.data = data
        self.sample_indices = list(holdout_idx)

    def create_prompt_base(self, row):
        return (
            f"You are helping Person 1, a loan officer, decide whether to approve a loan application. "
            f"The applicant is requesting ${row['loan_amnt']:,.0f} for {row['purpose']}. "
            f"They have been employed for {row['emp_length']}, "
            f"a debt-to-income ratio of {row['dti']:.1f}%, "
            f"and a credit score of {row['fico']:.0f}."
        )

    def get_human_response(self, row):
        return int(row['accepted'])

    def make_result_row(self, row, row_idx, base_prompt):
        return {
            'index': row_idx, 'loan_amnt': row['loan_amnt'], 'purpose': row['purpose'],
            'human_response': self.get_human_response(row), 'prompt': base_prompt,
        }


class HotelBookings(Dataset):
    name = "HotelBookings"
    thought_prompt = (
        "TASK: Predict Person 1's decision. Explain your reasoning in 1 sentence. "
        "Then conclude with exactly 'PREDICTION: 1' (keep booking) or 'PREDICTION: 0' (cancel)."
    )

    def load(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "hotel_bookings.csv"))
        df['arrival_date'] = pd.to_datetime(
            df['arrival_date_year'].astype(str) + '-' + df['arrival_date_month'] + '-' +
            df['arrival_date_day_of_month'].astype(str), format='%Y-%B-%d', errors='coerce'
        )
        df['kept_booking'] = 1 - df['is_canceled']
        features = [
            'arrival_date_week_number', 'stays_in_weekend_nights', 'stays_in_week_nights',
            'adults', 'children', 'is_repeated_guest', 'previous_cancellations',
            'required_car_parking_spaces', 'total_of_special_requests',
        ]
        df_clean = df[features + ['kept_booking', 'arrival_date', 'arrival_date_month',
                      'arrival_date_day_of_month', 'arrival_date_year']].copy()
        df_clean = df_clean.dropna(subset=features).reset_index(drop=True)
        _, holdout_idx = train_test_split(df_clean.index, test_size=0.2, random_state=42)
        self.data = df_clean
        self.sample_indices = list(holdout_idx)

    def create_prompt_base(self, row):
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

    def get_human_response(self, row):
        return int(row['kept_booking'])

    def make_result_row(self, row, row_idx, base_prompt):
        return {
            'id': row_idx, 'human_response': self.get_human_response(row), 'prompt': base_prompt,
        }


class MoralMachine(Dataset):
    name = "MoralMachine"
    thought_prompt = (
        "TASK: Predict Person 1's decision. Explain your reasoning in 1 sentence. "
        "Then conclude with exactly 'PREDICTION: 1' (swerve) or 'PREDICTION: 0' (do not swerve)."
    )

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

    def load(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "MoralMachine/SharedResponsesSurveyUSA1M.csv"))
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        df['row_num'] = df.groupby('ResponseID').cumcount()
        df_wide = df.pivot(index='ResponseID', columns='row_num')
        df_wide.columns = [f"{col}_{num}" for col, num in df_wide.columns]
        df_wide = df_wide.reset_index()
        features = [
            'Intervention_0', 'Intervention_1', 'PedPed_0', 'PedPed_1',
            'Barrier_0', 'Barrier_1', 'CrossingSignal_0', 'CrossingSignal_1',
            'AttributeLevel_0', 'AttributeLevel_1', 'NumberOfCharacters_0',
            'NumberOfCharacters_1', 'UserCountry3_0', 'Review_age_0',
            'Review_education_0', 'Review_gender_0', 'Review_income_0',
            'Review_political_0', 'Review_religious_0'
        ]
        df_clean = df_wide[features + ['Saved_0', 'ResponseID']].copy()
        df_clean['Review_age_0'] = pd.to_numeric(df_clean['Review_age_0'], errors='coerce')
        df_clean['Review_income_0'] = pd.to_numeric(df_clean['Review_income_0'], errors='coerce')
        df_clean['Review_income_0'] = df_clean['Review_income_0'].clip(lower=0, upper=1e6)
        df_clean = df_clean[(df_clean['Review_age_0'] >= 0) & (df_clean['Review_age_0'] <= 120)]
        df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_clean.dropna(inplace=True)
        _, holdout_idx = train_test_split(df_clean.index, test_size=0.2, random_state=42)
        self.data = df_clean
        self.sample_indices = list(holdout_idx)

    def create_prompt_base(self, row):
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

        education_str = self.education_map.get(row['Review_education_0'], "No Answer")
        income_str = self.income_map.get(row['Review_income_0'], "No Answer")
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

    def get_human_response(self, row):
        return int(row['Saved_0'])

    def make_result_row(self, row, row_idx, base_prompt):
        return {
            'ResponseID': row['ResponseID'], 'human_response': self.get_human_response(row),
            'prompt': base_prompt,
        }


class Uber(Dataset):
    name = "Uber"
    thought_prompt = (
        "TASK: Predict Person 1's decision. Explain your reasoning in 1 sentence. "
        "Then conclude with exactly 'PREDICTION: 1' (accept) or 'PREDICTION: 0' (decline)."
    )

    def load(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "uber_bookings.csv"))
        df = df[df['Booking_Status'].isin(['Success', 'Canceled by Driver', 'Canceled by Customer'])].copy()
        df['accepted'] = (df['Booking_Status'] == 'Success').astype(int)
        df['Pickup_Location'] = df['Pickup_Location'].fillna('Unknown').str.strip()
        df['Drop_Location'] = df['Drop_Location'].fillna('Unknown').str.strip()
        df = df.dropna(subset=['Pickup_Location', 'Drop_Location']).reset_index(drop=True)
        _, holdout_idx = train_test_split(df.index, test_size=0.2, random_state=42)
        self.data = df
        self.sample_indices = list(holdout_idx)

    def create_prompt_base(self, row):
        return (
            f"You are helping Person 1 with a ride decision. "
            f"Person 1 has received a ride request going from {row['Pickup_Location']} to {row['Drop_Location']}."
        )

    def get_human_response(self, row):
        return int(row['accepted'])

    def make_result_row(self, row, row_idx, base_prompt):
        return {
            'id': row_idx, 'pickup': row['Pickup_Location'], 'drop': row['Drop_Location'],
            'human_response': self.get_human_response(row), 'prompt': base_prompt,
        }


class FEVEROUS(Dataset):
    name = "FEVEROUS"
    thought_prompt = (
        "TASK: Predict Person 1's decision. Explain your reasoning in 1 sentence. "
        "Then conclude with exactly 'PREDICTION: 1' (support) or 'PREDICTION: 0' (refute)."
    )

    def load(self):
        data = pd.read_json(os.path.join(DATA_DIR, "FEVEROUS/feverous_train_challenges.jsonl"), lines=True)
        data = data.replace('', pd.NA).dropna(how='all')
        data = data[data["label"] != "NOT ENOUGH INFO"]
        data["supports"] = data["label"].map({"SUPPORTS": 1, "REFUTES": 0})
        data = data.reset_index(drop=True)
        self.data = data
        self.sample_indices = list(data.index)

    def create_prompt_base(self, row):
        return (
            f"You are helping Person 1 with fact-checking. "
            f"This statement needs to be checked: {row['claim']}."
        )

    def get_human_response(self, row):
        return int(row['supports'])

    def make_result_row(self, row, row_idx, base_prompt):
        return {
            'id': row['id'], 'claim': row['claim'],
            'human_response': self.get_human_response(row), 'prompt': base_prompt,
        }


class WikipediaToxicity(Dataset):
    name = "WikipediaToxicity"
    thought_prompt = (
        "TASK: Predict the crowd-workers' aggregate decision. Explain your reasoning in 1 sentence. "
        "Then conclude with exactly 'PREDICTION: 1' (toxic) or 'PREDICTION: 0' (not toxic)."
    )
    decision_task = (
        "TASK: Respond with a digit ONLY.\n"
        "Output 0 to implement your prediction, or 1 to escalate to the crowd-workers.\n"
        "A decision is considered correct if it equals the crowd-workers' true aggregate decision."
    )
    escalate_only_task = (
        "An LLM was asked to predict the crowd-workers' aggregate decision on this comment.\n"
        "TASK: Respond with a digit ONLY.\n"
        "Output 0 to implement the LLM's prediction, or 1 to escalate to the crowd-workers.\n"
        "A decision is considered correct if it equals the crowd-workers' true aggregate decision."
    )
    injected_decision_task = (
        "A separate model predicted PREDICTION: {pred} for the crowd-workers' aggregate decision on this comment.\n"
        "TASK: Respond with a digit ONLY.\n"
        "Output 0 to implement the model's prediction, or 1 to escalate to the crowd-workers.\n"
        "A decision is considered correct if it equals the crowd-workers' true aggregate decision."
    )

    def load(self):
        data = pd.read_csv(os.path.join(DATA_DIR, "WikipediaToxicity/Wikipedia Toxicity_data_data.csv"))
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
        data_agg = data_agg.reset_index(drop=True)
        self.data = data_agg
        self.sample_indices = list(data_agg.index)

    def create_prompt_base(self, row):
        return (
            f"You are helping a group of crowd-workers label Wikipedia discussion comments as toxic or not. "
            f"This comment needs to be checked: '{row['comment']}'."
        )

    def get_human_response(self, row):
        return int(row['toxicity'])

    def make_result_row(self, row, row_idx, base_prompt):
        return {
            'rev_id': row['rev_id'], 'human_response': self.get_human_response(row),
            'prompt': base_prompt,
        }


class MovieLens(Dataset):
    name = "MovieLens"
    thought_prompt = (
        "TASK: Predict which movie Person 1 would rate higher. Explain your reasoning in 1 sentence. "
        "Then conclude with exactly 'PREDICTION: 1' (first movie) or 'PREDICTION: 2' (second movie)."
    )
    decision_task = (
        "TASK: Respond with a digit ONLY.\n"
        "Output 0 to implement your prediction, or 1 to escalate to Person 1.\n"
        "A decision is correct if it matches the movie Person 1 would rate higher."
    )
    escalate_only_task = (
        "An LLM was asked to predict which movie Person 1 would rate higher.\n"
        "TASK: Respond with a digit ONLY.\n"
        "Output 0 to implement the LLM's prediction, or 1 to escalate to Person 1.\n"
        "A decision is correct if it matches the movie Person 1 would rate higher."
    )
    injected_decision_task = (
        "A separate model predicted PREDICTION: {pred} for which movie Person 1 would rate higher.\n"
        "TASK: Respond with a digit ONLY.\n"
        "Output 0 to implement the model's prediction, or 1 to escalate to Person 1.\n"
        "A decision is correct if it matches the movie Person 1 would rate higher."
    )
    parse_prediction = staticmethod(parse_prediction_pairwise)

    def load(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "MovieLens/movies_and_ratings_last1000000.csv"))
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        self.data = df
        self.test_indices = test_df.index.tolist()
        self.sample_indices = self.test_indices

    def create_prompt_base(self, row_idx):
        """Returns (prompt, answer_key) or (None, None) if user has < 7 ratings."""
        user_id = self.data.iloc[row_idx]['userId']
        user_data = self.data[self.data['userId'] == user_id].copy()
        if len(user_data) < 7:
            return None, None
        shuffled = user_data.sample(frac=1)
        movie_1 = shuffled.iloc[0]
        movie_2 = None
        used_indices = [0]
        for i in range(1, len(shuffled)):
            if shuffled.iloc[i]['rating'] != movie_1['rating']:
                movie_2 = shuffled.iloc[i]
                used_indices.append(i)
                break
        if movie_2 is None:
            return None, None
        history = shuffled.drop(shuffled.index[used_indices]).head(5)
        prompt = "Person 1 has reviewed the following movies:\n\n"
        for _, r in history.iterrows():
            prompt += f"- {r['title']} ({r['genres']}): Rated {r['rating']}/5\n"
        prompt += "\nConsider these two movies they have not seen:\n\n"
        test_pair = [movie_1, movie_2]
        random.shuffle(test_pair)
        for movie in test_pair:
            prompt += f"- {movie['title']} ({movie['genres']})\n"
        answer_key = {test_pair[0]['title']: test_pair[0]['rating'],
                      test_pair[1]['title']: test_pair[1]['rating']}
        return prompt, answer_key

    def get_human_response(self, row):
        # Not used directly; overridden in run_sample
        return None

    def make_result_row(self, row, row_idx, base_prompt):
        # Not used directly; overridden in run_sample
        return {}


class AIME(Dataset):
    name = "AIME"
    thought_prompt = (
        "TASK: Solve this math problem step by step. Explain your reasoning. "
        "Then conclude with exactly 'SOLUTION: <integer>'."
    )
    decision_task = (
        "TASK: Respond with a digit ONLY.\n"
        "Output 0 to implement your solution, or 1 to escalate to Person 1.\n"
        "A decision is correct if it equals Person 1's true answer."
    )
    escalate_only_task = (
        "An LLM was asked to solve this math problem.\n"
        "TASK: Respond with a digit ONLY.\n"
        "Output 0 to implement the LLM's solution, or 1 to escalate to Person 1.\n"
        "A decision is correct if it equals Person 1's true answer."
    )
    injected_decision_task = (
        "A separate model produced SOLUTION: {pred} for this math problem.\n"
        "TASK: Respond with a digit ONLY.\n"
        "Output 0 to implement the model's solution, or 1 to escalate to Person 1.\n"
        "A decision is correct if it equals Person 1's true answer."
    )
    parse_prediction = staticmethod(parse_prediction_aime)

    def load(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "AIME_Dataset_1983_2024.csv"))
        self.data = df
        self.sample_indices = list(df.index)

    def create_prompt_base(self, row):
        return f"You are helping Person 1 solve the following math problem: {row['Question']}."

    def get_human_response(self, row):
        return str(row['Answer'])

    def make_result_row(self, row, row_idx, base_prompt):
        return {
            **row.to_dict(), 'prompt': base_prompt, 'solution': row['Answer'],
        }


DATASET_REGISTRY = {
    "LendingClub": LendingClub,
    "HotelBookings": HotelBookings,
    "MoralMachine": MoralMachine,
    "Uber": Uber,
    "FEVEROUS": FEVEROUS,
    "WikipediaToxicity": WikipediaToxicity,
    "MovieLens": MovieLens,
    "AIME": AIME,
}


# ══════════════════════════════════════════════════════════════════════════════
# Method implementations
# ══════════════════════════════════════════════════════════════════════════════

def run_blind(ds, base_prompt):
    """Blind: predict (private), escalate seeing features only."""
    # Call 1: Predict
    thought_full = f"{base_prompt}\n\n{ds.thought_prompt}"
    thought = llm_single(thought_full)
    pred = ds.parse_prediction(thought)
    # Call 2: Escalate (fresh call, sees only features)
    decision_full = f"{base_prompt}\n\n{ds.escalate_only_task}"
    decision = llm_single(decision_full, max_tokens=8)
    esc = parse_decision(decision)
    trace = (
        f"[THOUGHT PROMPT]\n{thought_full}\n\n[THOUGHT]\n{thought}\n\n"
        f"[ESCALATE PROMPT]\n{decision_full}\n\n[DECISION]\n{decision}"
    )
    return {"pred": pred, "del": esc, "thought": thought, "evaluation": None, "trace": trace}


def run_injected(ds, base_prompt):
    """Injected: predict, escalate in fresh call with 'A separate model predicted [X]'."""
    # Call 1: Predict
    thought_full = f"{base_prompt}\n\n{ds.thought_prompt}"
    thought = llm_single(thought_full)
    pred = ds.parse_prediction(thought)
    # Call 2: Escalate with injected prediction
    decision_full = f"{base_prompt}\n\n{ds.injected_decision_task.format(pred=pred)}"
    decision = llm_single(decision_full, max_tokens=8)
    esc = parse_decision(decision)
    trace = (
        f"[THOUGHT PROMPT]\n{thought_full}\n\n[THOUGHT]\n{thought}\n\n"
        f"[INJECTED PROMPT]\n{decision_full}\n\n[DECISION]\n{decision}"
    )
    return {"pred": pred, "del": esc, "thought": thought, "evaluation": None, "trace": trace}


def run_chained(ds, base_prompt):
    """Chained: predict, escalate seeing prediction as own assistant turn (multi-turn)."""
    # Call 1: Predict
    thought_full = f"{base_prompt}\n\n{ds.thought_prompt}"
    thought = llm_single(thought_full)
    pred = ds.parse_prediction(thought)
    # Call 2: Escalate via multi-turn (prediction as assistant turn)
    messages = [
        {"role": "user", "content": thought_full},
        {"role": "assistant", "content": thought},
        {"role": "user", "content": ds.decision_task},
    ]
    decision = llm_multi(messages, max_tokens=8)
    esc = parse_decision(decision)
    trace = (
        f"[THOUGHT PROMPT]\n{thought_full}\n\n[THOUGHT]\n{thought}\n\n"
        f"[DECISION (multi-turn)]\n{ds.decision_task}\n\n[DECISION]\n{decision}"
    )
    return {"pred": pred, "del": esc, "thought": thought, "evaluation": None, "trace": trace}


def run_reasoning_only(ds, base_prompt):
    """Reasoning only: analyze (no pred) + predict (private) + escalate seeing analysis only."""
    # Call 1a: Analyze
    analyze_full = f"{base_prompt}\n\n{ds.analyze_prompt}"
    analysis = llm_single(analyze_full)
    # Call 1b: Predict (independent, for scoring)
    thought_full = f"{base_prompt}\n\n{ds.thought_prompt}"
    thought = llm_single(thought_full)
    pred = ds.parse_prediction(thought)
    # Call 2: Escalate (sees analysis but NOT prediction)
    decision_full = (
        f"SCENARIO:\n{base_prompt}\n\n"
        f"ANALYSIS:\n{analysis}\n\n"
        f"{ds.escalate_only_task}"
    )
    decision = llm_single(decision_full, max_tokens=8)
    esc = parse_decision(decision)
    trace = (
        f"[ANALYZE PROMPT]\n{analyze_full}\n\n[ANALYSIS]\n{analysis}\n\n"
        f"[THOUGHT PROMPT]\n{thought_full}\n\n[THOUGHT]\n{thought}\n\n"
        f"[DECISION PROMPT]\n{decision_full}\n\n[DECISION]\n{decision}"
    )
    return {"pred": pred, "del": esc, "thought": thought, "evaluation": analysis, "trace": trace}


def run_reasoning_prediction(ds, base_prompt):
    """Reasoning+prediction: analyze + predict (private) + escalate seeing analysis + chained prediction."""
    # Call 1a: Analyze
    analyze_full = f"{base_prompt}\n\n{ds.analyze_prompt}"
    analysis = llm_single(analyze_full)
    # Call 1b: Predict (independent)
    thought_prompt_with_analysis = (
        f"SCENARIO:\n{base_prompt}\n\n"
        f"ANALYSIS:\n{analysis}\n\n"
        f"{ds.thought_prompt}"
    )
    thought = llm_single(thought_prompt_with_analysis)
    pred = ds.parse_prediction(thought)
    # Call 2: Escalate via multi-turn (analysis in context, prediction as assistant turn)
    messages = [
        {"role": "user", "content": thought_prompt_with_analysis},
        {"role": "assistant", "content": thought},
        {"role": "user", "content": ds.decision_task},
    ]
    decision = llm_multi(messages, max_tokens=8)
    esc = parse_decision(decision)
    trace = (
        f"[ANALYZE PROMPT]\n{analyze_full}\n\n[ANALYSIS]\n{analysis}\n\n"
        f"[THOUGHT PROMPT]\n{thought_prompt_with_analysis}\n\n[THOUGHT]\n{thought}\n\n"
        f"[DECISION (multi-turn)]\n{ds.decision_task}\n\n[DECISION]\n{decision}"
    )
    return {"pred": pred, "del": esc, "thought": thought, "evaluation": analysis, "trace": trace}


METHOD_RUNNERS = {
    "blind": run_blind,
    "injected": run_injected,
    "chained": run_chained,
    "reasoning_only": run_reasoning_only,
    "reasoning_prediction": run_reasoning_prediction,
}


# ══════════════════════════════════════════════════════════════════════════════
# Execution
# ══════════════════════════════════════════════════════════════════════════════

def run_sample(ds, row_idx, method):
    """Run a single sample for a given dataset, index, and method. Returns result dict or None."""
    try:
        # MovieLens is special: create_prompt_base takes row_idx, returns tuple
        if ds.name == "MovieLens":
            prompt_result = ds.create_prompt_base(row_idx)
            base_prompt, answer_key = prompt_result
            if base_prompt is None:
                return None
            titles = list(answer_key.keys())
            rating_1, rating_2 = answer_key[titles[0]], answer_key[titles[1]]
            human_response = 1 if rating_1 >= rating_2 else 2
            common = {
                'userId': ds.data.iloc[row_idx]['userId'],
                'rating_1': rating_1, 'rating_2': rating_2,
                'human_response': human_response, 'prompt': base_prompt,
            }
        elif ds.name == "AIME":
            row = ds.data.loc[row_idx]
            base_prompt = ds.create_prompt_base(row)
            common = ds.make_result_row(row, row_idx, base_prompt)
            human_response = str(row['Answer'])
        else:
            row = ds.data.loc[row_idx]
            base_prompt = ds.create_prompt_base(row)
            common = ds.make_result_row(row, row_idx, base_prompt)
            human_response = ds.get_human_response(row)

        runner = METHOD_RUNNERS[method]
        result = runner(ds, base_prompt)

        return {
            **common,
            'method': method,
            'model': MODEL_SHORT,
            'llm_prediction': result['pred'],
            'llm_escalate': result['del'],
            'llm_thought': result['thought'],
            'llm_evaluation': result['evaluation'],
            'trace': result['trace'],
        }
    except Exception as e:
        print(f"Error on {ds.name} idx={row_idx} method={method}: {e}", flush=True)
        return None


def main():
    parser = argparse.ArgumentParser(description="Together API runner for Studies 1 & 2")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_REGISTRY.keys()),
                        help="Datasets to run (default: all)")
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        help="Methods to run (default: all 5)")
    parser.add_argument("--n", type=int, default=100, help="Samples per condition")
    parser.add_argument("--workers", type=int, default=20, help="Parallel workers")
    args = parser.parse_args()

    print(f"Model: {MODEL}", flush=True)
    print(f"Datasets: {args.datasets}", flush=True)
    print(f"Methods: {args.methods}", flush=True)
    print(f"Samples: {args.n}, Workers: {args.workers}", flush=True)

    for ds_name in args.datasets:
        ds_cls = DATASET_REGISTRY[ds_name]
        ds = ds_cls()
        print(f"\n{'='*60}\nLoading {ds_name}...", flush=True)
        try:
            ds.load()
        except FileNotFoundError as e:
            print(f"Skipping {ds_name}: {e}", flush=True)
            continue
        print(f"Loaded. {len(ds.sample_indices)} sample indices available.", flush=True)

        for method in args.methods:
            if method not in METHOD_RUNNERS:
                print(f"Unknown method: {method}, skipping", flush=True)
                continue

            out_dir = os.path.join(RESULTS_DIR, ds_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{method}_{MODEL_SHORT}.csv")

            # Load existing results to append
            try:
                df_existing = pd.read_csv(out_path)
                existing_count = len(df_existing)
            except FileNotFoundError:
                df_existing = pd.DataFrame()
                existing_count = 0

            n_to_sample = min(args.n, len(ds.sample_indices))
            sampled = random.sample(ds.sample_indices, n_to_sample)

            results = []
            completed = 0
            total = n_to_sample
            lock = threading.Lock()

            def tracked_run(idx):
                nonlocal completed
                result = run_sample(ds, idx, method)
                with lock:
                    completed += 1
                    if result is not None:
                        results.append(result)
                    if completed % 10 == 0 or completed == total:
                        print(f"  [{completed}/{total}] {ds_name}/{method}", flush=True)
                return result

            print(f"\n  Running {method} on {ds_name} ({n_to_sample} samples, {existing_count} existing)...", flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(tracked_run, idx) for idx in sampled]
                for f in as_completed(futures):
                    f.result()

            if results:
                df_new = pd.DataFrame(results)
                df_new['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_csv(out_path, index=False)
                print(f"  Saved {len(df_new)} new rows to {out_path} (total: {len(df_combined)})", flush=True)
            else:
                print(f"  No results for {ds_name}/{method}", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
