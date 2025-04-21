import pandas as pd
from ionic_data import OXIDATION_STATES

def filter_charge_neutral_compositions(
    df,
    delta_range=(0.0, 0.5),
    debug=False,
    save_csv=True,
    neutral_csv_path="neutral_entries.csv",
    non_neutral_csv_path="non_neutral_entries.csv"
):
    """
    Screen dataframe of compositions for charge neutrality using oxidation state range overlap.

    Args:
        df: DataFrame with 'x', 'y', 'A_dopant', 'B_dopant', 'A_base', 'B_base' columns.
        delta_range: tuple indicating oxygen deficiency range, e.g., (0.0, 0.5)
        debug: whether to print info on non-neutral compositions
        save_csv: whether to save neutral and non-neutral entries to CSV (default: True)
        neutral_csv_path: output file path for neutral entries
        non_neutral_csv_path: output file path for non-neutral entries

    Returns:
        Filtered DataFrame with only neutral entries
    """
    is_neutral_flags = []
    debug_info = []

    for _, row in df.iterrows():
        x = float(row['x'])
        y = float(row['y'])
        A = row['A_dopant']
        B = row['B_dopant']
        A_base = row['A_base']
        B_base = row['B_base']
        

        # Oxidation states
        A_base_states = OXIDATION_STATES.get(A_base, [])
        A_dop_states = OXIDATION_STATES.get(A, []) if A else A_base_states
        B_states = OXIDATION_STATES.get(B, []) if B else OXIDATION_STATES.get(B_base, [])
        Fe_states = OXIDATION_STATES.get("Fe", [])

        if not A_base_states or not A_dop_states or not B_states or not Fe_states:
            is_neutral_flags.append(False)
            if debug:
                print(f"Missing oxidation states: A_base={A_base}, A_dop={A}, B_base={B_base}, B_dop={B}")
            continue

        # A-site oxidation state range
        A_min = min(n_dop * x + n_base * (1 - x) for n_dop in A_dop_states for n_base in A_base_states)
        A_max = max(n_dop * x + n_base * (1 - x) for n_dop in A_dop_states for n_base in A_base_states)

        # Required total cation charge to balance O^{2-}
        total_anion_charge_range = [-(3 - delta) * 2 for delta in delta_range]
        total_cation_range = [abs(q) for q in total_anion_charge_range]

        target_charge_min = min(total_cation_range) - A_max
        target_charge_max = max(total_cation_range) - A_min

        # Fe/B oxidation state range
        Fe_min, Fe_max = min(Fe_states), max(Fe_states)
        B_min, B_max = min(B_states), max(B_states)

        min_FB_charge = (1 - y) * Fe_min + y * B_min
        max_FB_charge = (1 - y) * Fe_max + y * B_max

        overlap = not (max_FB_charge < target_charge_min or min_FB_charge > target_charge_max)
        is_neutral_flags.append(overlap)

        if debug and not overlap:
            debug_info.append({
                'x': x, 'y': y, 'A_dopant': A, 'B_dopant': B,
                'A_base': A_base, 'B_base': B_base,
                'target_cation_charge_range': [round(target_charge_min, 2), round(target_charge_max, 2)],
                'FeB_charge_range': [round(min_FB_charge, 2), round(max_FB_charge, 2)],
            })

    df['is_neutral'] = is_neutral_flags
    neutral_df = df[df['is_neutral']].drop(columns='is_neutral')
    non_neutral_df = df[~df['is_neutral']].drop(columns='is_neutral')

    if debug and debug_info:
        print("\nNon-neutral entries:")
        for info in debug_info:
            print(f"- A: {info['A_base']}_{info['A_dopant']} | B: {info['B_base']}_{info['B_dopant']} | x: {info['x']:.3f} | y: {info['y']:.3f}")
            print(f"  Required cation charge range: {info['target_cation_charge_range']}")
            print(f"  Fe/B cation charge range:     {info['FeB_charge_range']}\n")

    if save_csv:
        neutral_df.to_csv(neutral_csv_path, index=False)
        non_neutral_df.to_csv(non_neutral_csv_path, index=False)

    return neutral_df
