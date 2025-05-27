import os
from pymatgen.io.vasp.outputs import Vasprun
import glob

def check_convergence(paths_to_check):
    failed_jobs = []
    successful_jobs = []

    for path in paths_to_check:
        vasprun_path = os.path.join(path, "vasprun.xml")
        if os.path.isfile(vasprun_path):
            try:
                vasprun = Vasprun(vasprun_path, parse_dos=False, parse_eigen=False)
                if vasprun.converged:
                    successful_jobs.append(path)
                else:
                    failed_jobs.append(path)
            except Exception as e:
                print(f"Error parsing {vasprun_path}: {e}")
                failed_jobs.append(path)
        else:
            print(f"vasprun.xml not found in: {path}")
            failed_jobs.append(path)

    print(f"Converged jobs: {len(successful_jobs)}")
    print(f"Failed jobs: {len(failed_jobs)}")

    with open("converged_jobs.txt", "w") as f:
        f.writelines([p + "\n" for p in successful_jobs])

    with open("failed_jobs.txt", "w") as f:
        f.writelines([p + "\n" for p in failed_jobs])
