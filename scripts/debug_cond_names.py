import glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ["TOGETHER_API_KEY"] = "unused"
from study3 import lending_conditions, load_lending, hotel_conditions, load_hotel

for ds_name, load_fn, cond_fn in [("HotelBookings", load_hotel, hotel_conditions), ("LendingClub", load_lending, lending_conditions)]:
    print(f"\n=== {ds_name} ===")
    df = load_fn()
    conds = cond_fn(df)
    cond_names = {c["name"] for c in conds}

    csvs = glob.glob(f"results/study3/{ds_name}_*_nothink_Qwen3.5-9B.csv")
    csvs = [f for f in csvs if "cost" not in f and "nohint" not in f and "summary" not in f and "isolated" not in f]
    for f in sorted(csvs):
        bn = os.path.basename(f)
        cond = bn.replace(f"{ds_name}_", "").replace("_nothink_Qwen3.5-9B.csv", "")
        match = cond in cond_names
        if not match:
            br = [c for c in conds if c["name"] in cond or cond in c["name"]]
            print(f"  {cond}: NO MATCH (defaults to 0.5!) similar: {[c['name'] for c in br]}")
        else:
            real_br = [c["base_rate"] for c in conds if c["name"] == cond][0]
            print(f"  {cond}: MATCH (base_rate={real_br:.2f})")
