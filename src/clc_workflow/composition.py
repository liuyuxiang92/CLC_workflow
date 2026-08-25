# composition.py
import pandas as pd
from itertools import product

def generate_valid_combinations(
    formula_extra_part,
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

    combinations = []

    def add_entry(x, y, A_dopant=None, B_dopant=None):
        parts = []

        if x != 1.0:
            parts.append(f"{A_base}{1 - x:.3f}")
        if x != 0.0 and A_dopant:
            parts.append(f"{A_dopant}{x:.3f}")

        if y != 1.0:
            parts.append(f"{B_base}{1 - y:.3f}")
        if y != 0.0 and B_dopant:
            parts.append(f"{B_dopant}{y:.3f}")

        parts.append(formula_extra_part)
        formula = "".join(parts)

        # 6 dp, not 3: the point of rounding is to clean the float noise out of a grid
        # built as 0.125*i, but 3 dp also destroys any composition finer than a thousandth
        # -- (Sr0.75Ba0.25)0.95 is x = 0.2375, which came back as 0.238 and then failed to
        # land on a whole site.  6 dp cleans the noise and keeps every fraction that a
        # reasonable supercell can hold.
        combinations.append({
            "x": round(x, 6), "1_x": round(1 - x, 6),
            "y": round(y, 6), "1_y": round(1 - y, 6),
            f"{A_site}_base": A_base, f"{B_site}_base": B_base,
            f"{A_site}_dopant": A_dopant, f"{B_site}_dopant": B_dopant,
            "formula": formula
        })

    # Exactly the requested (x, y) grid -- no value is ever invented.  At x == 0 the A
    # dopant is absent, so every A_dopant would label the same compound; collapse it to
    # one row with A_dopant = None.  Same for y == 0 on the B site.  (At x == 1 the base
    # is what is absent, and each dopant IS a distinct compound, so all are kept.)  That
    # per-endpoint collapse is the only thing the old interior/edge/corner split was for
    # -- but it hardcoded 0.0 and 1.0 into the grid whether or not they were requested,
    # so x_values=[0.625], y_values=[0.750] returned 9 rows instead of 1.
    for x, y in product(x_vals, y_vals):
        if x == 0.0 and y == 0.0 and not include_undoped:
            continue
        A_choices = [None] if x == 0.0 else A_dopants
        B_choices = [None] if y == 0.0 else B_dopants
        for A, B in product(A_choices, B_choices):
            add_entry(x, y, A, B)

    df = pd.DataFrame(combinations)

    if save_csv:
        df.to_csv(save_csv_path, index=False)

    return df
