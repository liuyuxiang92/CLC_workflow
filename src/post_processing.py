import dpdata
import os
import numpy as np

def extract_dft_energy_dpdata(vasp_outcar_path):
    """
    Extract the total energy from a VASP OUTCAR file using dpdata.LabeledSystem.

    Args:
        outcar_path (str): Path to the OUTCAR file.

    Returns:
        float: Final total energy (eV), or None if extraction fails.
    """
    try:
        labeled_sys = dpdata.LabeledSystem(vasp_outcar_path, fmt='vasp/outcar')
        labeled_sys["energies"].shape == (1,)
        energy = labeled_sys['energies'][-1]  # Final energy in eV
        return energy
    except Exception as e:
        print(f"Failed to read energy from {vasp_outcar_path}: {e}")
        return None
def extract_magmom_from_outcar_all_frames(outcar_path):
    if not os.path.exists(outcar_path):
        raise FileNotFoundError(f"OUTCAR not found at: {outcar_path}")

    with open(outcar_path, 'r') as f:
        lines = f.readlines()

    last_block = []
    for i in reversed(range(len(lines))):
        if "magnetization (x)" in lines[i]:
            # Start collecting from here forward
            j = i + 4
            while j < len(lines) and lines[j].strip() and "---" not in lines[j]:
                parts = lines[j].split()
                last_block.append([float(parts[4])])
                j += 1
            break  # stop after finding the last block

    if not last_block:
        raise ValueError("No magmom data found in OUTCAR.")

    magmom_array = np.array([last_block])  # shape: (1, n_atoms, 1)
    return magmom_array

