import subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor, as_completed

scripts_dir = os.path.dirname(os.path.abspath(__file__))
scripts = [
    "run_aime.py",
    "run_LendingClub.py",
    "run_HotelBookings.py",
    "run_Uber.py",
    "run_JFLEG.py",
    "run_FEVEROUS.py",
    "run_WikipediaToxicity.py",
    "run_MovieLens.py",
    "run_MoralMachine.py",
]

def run_script(script):
    path = os.path.join(scripts_dir, script)
    print(f"[START] {script}", flush=True)
    result = subprocess.run([sys.executable, path])
    print(f"[DONE] {script}", flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed with exit code {result.returncode}")
    return script

with ThreadPoolExecutor(max_workers=1) as executor:
    futures = {executor.submit(run_script, s): s for s in scripts}
    for f in as_completed(futures):
        f.result()

print("\nAll scripts complete.", flush=True)
