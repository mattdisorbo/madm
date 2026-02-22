import subprocess, sys, os

scripts_dir = os.path.dirname(os.path.abspath(__file__))
scripts = [
    "run_MoralMachine.py",
    "run_MovieLens.py",
    "run_WikipediaToxicity.py",
    "run_FEVEROUS.py",
    "run_JFLEG.py",
    "run_LendingClub.py",
]

for script in scripts:
    path = os.path.join(scripts_dir, script)
    print(f"\n{'='*60}\nRunning {script}\n{'='*60}\n", flush=True)
    subprocess.run([sys.executable, path], check=True)

print("\nAll scripts complete.", flush=True)
