import pandas as pd

def filter_compositions(
    df,
    modes=["all"],
    *,
    n=None,
    n_per_combination=1,
    seed=42,
    allowed_element_pairs=None,
    x_range=(0.0, 1.0),
    y_range=(0.0, 1.0)
):
    df_filtered = df.copy()
    
#    if "element_combination" in modes and allowed_pairs:
#        df_filtered = df_filtered[df_filtered.apply(
#            lambda row: (row["A_dopant"], row["B_dopant"]) in allowed_pairs, axis=1
#        )]
    # Filter by same-site element pairs
    if "element_combination" in modes and allowed_element_pairs:
        filtered_rows = []
    
        for _, row in df_filtered.iterrows():
            # Track individual site pair matches
            a_match = b_match = True  # default to True if not specified
    
            # A-site check
            if "A" in allowed_element_pairs:
                a_match = False
                if pd.notna(row["A_base"]) and pd.notna(row["A_dopant"]):
                    a_pair = tuple(sorted([row["A_base"], row["A_dopant"]]))
                    allowed_A_pairs = [tuple(sorted(p)) for p in allowed_element_pairs["A"]]
                    if a_pair in allowed_A_pairs:
                        a_match = True
    
            # B-site check
            if "B" in allowed_element_pairs:
                b_match = False
                if pd.notna(row["B_base"]) and pd.notna(row["B_dopant"]):
                    b_pair = tuple(sorted([row["B_base"], row["B_dopant"]]))
                    allowed_B_pairs = [tuple(sorted(p)) for p in allowed_element_pairs["B"]]
                    if b_pair in allowed_B_pairs:
                        b_match = True
    
            # Keep if either A or B specified; both must match if both are given
            if ("A" in allowed_element_pairs and "B" in allowed_element_pairs and a_match and b_match) or \
               ("A" in allowed_element_pairs and "B" not in allowed_element_pairs and a_match) or \
               ("B" in allowed_element_pairs and "A" not in allowed_element_pairs and b_match):
                filtered_rows.append(row)
    
        df_filtered = pd.DataFrame(filtered_rows)
    
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

