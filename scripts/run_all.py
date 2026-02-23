import subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor, as_completed

scripts_dir = os.path.dirname(os.path.abspath(__file__))
scripts = [
    "run_MoralMachine.py",
    "run_MovieLens.py",
    "run_WikipediaToxicity.py",
    "run_FEVEROUS.py",
    "run_JFLEG.py",
    "run_LendingClub.py",
    "run_aime.py",
]

def run_script(script):
    path = os.path.join(scripts_dir, script)
    print(f"[START] {script}", flush=True)
    result = subprocess.run([sys.executable, path], capture_output=True, text=True)
    print(f"[DONE] {script}\n{result.stdout}", flush=True)
    if result.returncode != 0:
        print(f"[ERROR] {script}\n{result.stderr}", flush=True)
        raise RuntimeError(f"{script} failed with exit code {result.returncode}")
    return script

with ThreadPoolExecutor(max_workers=1) as executor:
    futures = {executor.submit(run_script, s): s for s in scripts}
    for f in as_completed(futures):
        f.result()

print("\nAll scripts complete.", flush=True)
