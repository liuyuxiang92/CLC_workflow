import pandas as pd

def filter_compositions(
    df,
    modes=["all"],
    *,
    n=None,
    n_per_combination=1,
    seed=42,
    allowed_pairs=None,
    x_range=(0.0, 1.0),
    y_range=(0.0, 1.0)
):
    df_filtered = df.copy()
    
    if "element_combination" in modes and allowed_pairs:
        df_filtered = df_filtered[df_filtered.apply(
            lambda row: (row["A_dopant"], row["B_dopant"]) in allowed_pairs, axis=1
        )]
    
    if "concentration_range" in modes:
        df_filtered = df_filtered[
            df_filtered["x"].between(*x_range) &
            df_filtered["y"].between(*y_range)
        ]
    
    if "n_per_combination" in modes and n_per_combination > 0:
        df_filtered = (
            df_filtered
            .groupby(["A_dopant", "B_dopant"])
            .apply(lambda g: g.sample(n=min(n_per_combination, len(g)), random_state=seed))
            .reset_index(drop=True)
        )
    
    if "random_n" in modes and n is not None:
        df_filtered = df_filtered.sample(n=n, random_state=seed)
    
    return df_filtered

