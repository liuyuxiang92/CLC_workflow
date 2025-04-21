# composition.py
import pandas as pd
from itertools import product

def generate_valid_combinations(
    formula_template,
    site_vars,
    site_dopants,
    site_values,
    include_undoped=True,
    save_csv=True,
    save_csv_path="valid_combinations.csv"
):
    x_var, y_var = list(site_vars.keys())
    A_site, B_site = site_vars[x_var], site_vars[y_var]
    A_base, B_base = site_dopants[A_site]['base'], site_dopants[B_site]['base']
    A_dopants, B_dopants = site_dopants[A_site]['dopants'], site_dopants[B_site]['dopants']
    x_vals, y_vals = site_values[x_var], site_values[y_var]

    x_inner = [v for v in x_vals if 0 < v < 1]
    y_inner = [v for v in y_vals if 0 < v < 1]

    combinations = []

    def add_entry(x, y, A=None, B=None):
        parts = []

        if x != 1.0:
            parts.append(f"{A_base}{1 - x:.3f}")
        if x != 0.0 and A:
            parts.append(f"{A}{x:.3f}")

        if y != 1.0:
            parts.append(f"{B_base}{1 - y:.3f}")
        if y != 0.0 and B:
            parts.append(f"{B}{y:.3f}")

        parts.append("O3")
        formula = "".join(parts)

        combinations.append({
            "x": round(x, 3), "1_x": round(1 - x, 3),
            "y": round(y, 3), "1_y": round(1 - y, 3),
            f"{A_site}_base": A_base, f"{B_site}_base": B_base,
            f"{A_site}_dopant": A, f"{B_site}_dopant": B,
            "formula": formula
        })

    # Interior
    for A, B in product(A_dopants, B_dopants):
        for x in x_inner:
            for y in y_inner:
                add_entry(x, y, A, B)

    # Edges
    for B in B_dopants:
        for y in y_inner:
            add_entry(0.0, y, None, B)
            for A in A_dopants:
                add_entry(1.0, y, A, B)

    for A in A_dopants:
        for x in x_inner:
            add_entry(x, 0.0, A, None)
            for B in B_dopants:
                add_entry(x, 1.0, A, B)

    # Corners
    if include_undoped:
        add_entry(0.0, 0.0, None, None)

    for B in B_dopants:
        add_entry(0.0, 1.0, None, B)

    for A in A_dopants:
        add_entry(1.0, 0.0, A, None)

    for A, B in product(A_dopants, B_dopants):
        add_entry(1.0, 1.0, A, B)

    df = pd.DataFrame(combinations)

    if save_csv:
        df.to_csv(save_csv_path, index=False)

    return df
