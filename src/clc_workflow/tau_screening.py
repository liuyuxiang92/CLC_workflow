import numpy as np
import pandas as pd

from clc_workflow.ionic_data import radii_dict, oxidation_state_bounds, OXIDATION_STATES


# Constants
r_Sr = radii_dict["Sr"][2]
r_O = radii_dict["O"][-2]
r_Ov = radii_dict["Ov"][0]

def compute_tau_value(row, delta=0.0, tau_upper=4.3, debug=False):
    if not isinstance(delta, (list, tuple, np.ndarray)):
        delta = [delta]

    x, y = float(row['x']), float(row['y'])
    A = row['A_dopant']
    B = row['B_dopant']
    A_base = row['A_base']
    B_base = row['B_base']
    formula = row.get("formula", "")

    if debug:
        print(f"formula = {formula}")
    A_elem = A if A else A_base
    B_elem = B if B else B_base

    if A_elem not in radii_dict or B_elem not in radii_dict:
        return np.nan, np.nan, False  # not stable

    n_A = list(radii_dict[A_elem].keys())[0]
    r_A = radii_dict[A_elem][n_A]

    if B_elem in oxidation_state_bounds:
        n_B_min, n_B_max = oxidation_state_bounds[B_elem]
    else:
        n_B_min = n_B_max = list(radii_dict[B_elem].keys())[0]

    r_B_min = radii_dict[B_elem][n_B_max]
    r_B_max = radii_dict[B_elem][n_B_min]
    r_Fe_min = radii_dict["Fe"][4]
    r_Fe_max = radii_dict["Fe"][3]

    # Store all tau values for final min/max if none pass
    all_tau_values = []
    successful_taus = []  # Store taus that meet both n_B conditions

    for d in delta:
        # Track taus for this delta
        delta_taus = []
        both_passed = True  # Assume both will pass until proven otherwise
        #skip_delta = False 
        
        for n_B in [n_B_min, n_B_max]:

            n_A_eff = 2 * (1-x) + n_A * x
            n_O = 6 - 2 * d
            n_B_criter = n_O - n_A_eff

            n_B_eff_upper = (1 - y) * 4 + y * n_B_max
            n_B_eff_lower = (1 - y) * 3 + y * n_B_min

            #if (n_B_criter < n_B_eff_lower or n_B_criter > n_B_eff_upper):
            #    skip_delta = True
            #    print(f"formula = {formula},  δ={d:.3f}" )
            #    break
            
            if n_B_min == n_B_max:
                r_B = r_B_min
            else:
                C1 = (n_B_max * r_B_max - n_B_min * r_B_min) / (n_B_max - n_B_min)
                C2 = (r_B_max - r_B_min) / (n_B_max - n_B_min)
                r_B = C1 - C2 * n_B

            if y == 1.0:
                n_Fe = (6 - 2 * d - 2 * (1 - x) - n_A * x)
            else:
                C3 = (6 - 2 * d - 2 * (1 - x) - n_A * x) / (1 - y)
                C4 = y / (1 - y)
                n_Fe = C3 - C4 * n_B

            r_Fe = (4 - n_Fe) * r_Fe_max + (n_Fe - 3) * r_Fe_min
            r_B_eff = r_Fe * (1 - y) + r_B * y

            r_O_eff = r_O * ((3 - d) / 3) + r_Ov * d / 3
            r_A_eff = r_Sr * (1 - x) + r_A * x

            try:
                r_A_r_B = r_A_eff / r_B_eff
                ln_term = np.log(r_A_r_B)
                n_A_eff = 2 * (1 - x) + n_A * x
                tau = (r_O_eff / r_B_eff) - n_A_eff * (n_A_eff - r_A_r_B/ln_term)
                
                all_tau_values.append(tau)
                delta_taus.append(tau)
                
                if debug:
                    print(f"[DEBUG] {formula} | δ={d:.3f}, n_B={n_B:.3f}, r_O_eff={r_O_eff:.3f}, r_B_eff={r_B_eff:.3f},n_A_eff={n_A_eff:.3f},r_A/r_B={r_A_r_B:.3f}, ln_term={ln_term:.3f}, τ={tau:.4f}")

                if tau > tau_upper:
                    both_passed = False

            except Exception as e:
                if debug:
                    print(f"[ERROR] {formula} failed τ calc: {e}")
                both_passed = False
                #continue
        #if skip_delta:
        #   continue
        
        # After checking both n_B values for this delta
        if both_passed and len(delta_taus) == 2:
            successful_taus.extend(delta_taus)
            # Return immediately if we find a delta where both n_B values pass
            tau_avg = sum(delta_taus)/2  # or use min/max as needed
            #return tau_avg, tau_avg, True

    if successful_taus:
        # This would only be reached if we didn't return above
        tau_avg = sum(successful_taus)/len(successful_taus)
        #if debug:
        #     print(f"[DEBUG] {formula} | δ={d:.3f}, n_B={n_B:.3f}, r_O_eff={r_O_eff:.3f}, r_B_eff={r_B_eff:.3f},n_A_eff={n_A_eff:.3f},r_A/r_B={r_A_r_B:.3f}, ln_term={ln_term:.3f}, τ={tau:.4f}")
        return tau_avg, tau_avg, True

    if not all_tau_values:
        return np.nan, np.nan, False

    return min(all_tau_values), max(all_tau_values), False

def filter_tau_stable_compositions(
    df,
    delta=0.0,
    tau_upper=4.3,
    save_csv=True,
    output_path="neutral_with_tau_screened.csv",
    debug=False
):
    tau_min_list = []
    tau_max_list = []
    is_stable_flags = []
    if(debug):
       count = 0
    for _, row in df.iterrows():
        tau_min, tau_max, is_stable = compute_tau_value(
            row,
            delta=delta,
            tau_upper=tau_upper,
            debug=debug
        )
        if(debug):
           count +=1
           print(f"This is {count}")
        tau_min_list.append(tau_min)
        tau_max_list.append(tau_max)
        is_stable_flags.append(is_stable)

    df['tau_min'] = tau_min_list
    df['tau_max'] = tau_max_list
    df['is_stable_tau'] = is_stable_flags
    df_filtered = df[df['is_stable_tau']].copy()

    if debug:
        print(f"[INFO] Total: {len(df)} | τ valid: {np.isfinite(df['tau_min']).sum()} | Passed screen: {len(df_filtered)}")

    if save_csv:
        df_filtered.to_csv(output_path, index=False)

    return df_filtered

