import dpdata

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

