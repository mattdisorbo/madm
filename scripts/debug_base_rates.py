import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ["TOGETHER_API_KEY"] = "unused"
from study3 import (load_hotel, hotel_conditions, load_lending, lending_conditions,
                     load_moral, moral_conditions, load_wiki, wiki_conditions,
                     load_movielens, movielens_conditions)

for name, load_fn, cond_fn in [
    ("HotelBookings", load_hotel, hotel_conditions),
    ("LendingClub", load_lending, lending_conditions),
    ("MoralMachine", load_moral, moral_conditions),
    ("WikipediaToxicity", load_wiki, wiki_conditions),
    ("MovieLens", load_movielens, movielens_conditions),
]:
    df = load_fn()
    conds = cond_fn(df)
    brs = [c["base_rate"] for c in conds]
    near = sum(1 for b in brs if 0.65 < b < 0.85)
    print(f"{name}: {[round(b,2) for b in sorted(brs)]}, near_75={near}/{len(brs)}")
