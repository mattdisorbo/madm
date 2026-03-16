"""Prepare GRPO dataset for MoralMachine. Runs predictions via Together API."""
import os, re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import openai

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

def create_prompt_base(row):
    def action(val): return "doesn't swerve" if val == 0 else "swerves"
    def group(val): return "passengers" if val == 1 else "pedestrians"
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
    if ped0 == 1 and ped1 == 0 and signal0: base += f" The pedestrians are {signal0}."
    elif ped0 == 0 and ped1 == 1 and signal1: base += f" The pedestrians are {signal1}."
    elif ped0 == 1 and ped1 == 1:
        parts = []
        if signal0: parts.append(f"The first group of pedestrians are {signal0}.")
        if signal1: parts.append(f"The second group of pedestrians are {signal1}.")
        if parts: base += " " + " ".join(parts)
    return (base +
        " Person 1, with the following characteristics, is in the driver's seat: "
        f"Age: {row['Review_age_0']}. Education: {education_str}. "
        f"Gender: {row['Review_gender_0']}. Income: {income_str}. "
        f"Political (0 is Conservative, 1 is Progressive): {row['Review_political_0']}. "
        f"Religious (0 is Not Religious, 1 is Religious): {row['Review_religious_0']}. "
    )

MODEL = "Qwen/Qwen3.5-9B"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "../data/grpo_dataset_mm")

client = openai.OpenAI(
    api_key=os.environ["TOGETHER_API_KEY"],
    base_url="https://api.together.xyz/v1",
)

THOUGHT_PROMPT = (
    "TASK: Predict Person 1's decision. Explain your reasoning in 1 sentence. "
    "Then conclude with exactly 'PREDICTION: 1' or 'PREDICTION: 0'."
)
ESCALATE_TASK = (
    "An LLM was asked to predict Person 1's decision in this scenario.\n"
    "TASK: Respond with a digit ONLY.\n"
    "Output 0 to implement the LLM's prediction, or 1 to escalate to Person 1.\n"
    "A decision is considered correct if it equals Person 1's true decision."
)

def llm(prompt):
    r = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    text = r.choices[0].message.content.strip()
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def parse_prediction(text):
    preds = re.findall(r'PREDICTION:\s*([01])', text)
    if preds: return int(preds[-1])
    digits = re.findall(r'[01]', text.strip())
    if digits: return int(digits[0])
    return None

def process_sample(row):
    scenario = create_prompt_base(row)
    gt = int(row['Saved_0'])
    try:
        response = llm(f"{scenario}\n\n{THOUGHT_PROMPT}")
        pred = parse_prediction(response)
    except Exception:
        return None
    if pred is None: return None
    return {"prompt": f"{scenario}\n\n{ESCALATE_TASK}", "ground_truth": str(gt), "prediction": str(pred)}

if __name__ == "__main__":
    print("Loading MoralMachine data...")
    df = pd.read_csv(os.path.join(BASE_DIR, "../data/MoralMachine/SharedResponsesSurveyUSA1M.csv"))
    if "Unnamed: 0" in df.columns: df = df.drop(columns=["Unnamed: 0"])
    df['row_num'] = df.groupby('ResponseID').cumcount()
    mm = df.pivot(index='ResponseID', columns='row_num')
    mm.columns = [f"{col}_{num}" for col, num in mm.columns]
    mm = mm.reset_index()
    mm['Review_age_0'] = pd.to_numeric(mm['Review_age_0'], errors='coerce')
    mm = mm[(mm['Review_age_0'] >= 0) & (mm['Review_age_0'] <= 120)]
    # Sample 2000 balanced
    n_per = 1000
    s0 = mm[mm['Saved_0'] == 0].sample(n=n_per, random_state=42)
    s1 = mm[mm['Saved_0'] == 1].sample(n=n_per, random_state=42)
    train_data = pd.concat([s0, s1], ignore_index=True)
    print(f"Samples: {len(train_data)}")

    examples = []
    failed = 0
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_sample, train_data.iloc[i]): i for i in range(len(train_data))}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Predictions"):
            result = f.result()
            if result: examples.append(result)
            else: failed += 1

    print(f"Done: {len(examples)} ok, {failed} failed")
    from datasets import Dataset
    ds = Dataset.from_list(examples)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ds.save_to_disk(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR} ({len(ds)} examples)")
    df = pd.DataFrame(examples)
    print(f"Pred accuracy: {(df['prediction']==df['ground_truth']).mean():.3f}")
