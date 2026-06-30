"""simulate_pricing_models.py
An adaptive post-auction local electricity market framework. It dynamically selects the most appropriate pricing [...]

Generated from: output/2af6d5e5a75632bc_Rene_10_pricing/2af6d5e5a75632bc_Rene_10_pricing.pdf
BM name       : pricing models

Usage
-----
    python simulate_pricing_models.py \
        --dataset  path/to/dataset.csv \
        --tariffs  path/to/tariffs.csv \
        [--param   name=value ...]

Or import and call directly:

    import pandas as pd
    from simulate_pricing_models import simulate

    df      = pd.read_csv("dataset.csv")
    tariffs = pd.read_csv("tariffs.csv")
    results = simulate(df, tariffs, {})
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


# -- Generated simulate function -------------------------------------


def simulate(df, tariffs, parameters):
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # REQUIRED HELPER FUNCTIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def is_nan(x):
        return x != x

    def safe_to_float(value, default=0.0):
        if hasattr(value, 'astype'):
            try:
                return value.fillna(default).astype(float)
            except Exception:
                pass
        if hasattr(value, '__iter__') and not isinstance(value, (str, dict)):
            res = []
            for x in value:
                try:
                    if x != x or x is None:
                        res.append(default)
                    else:
                        res.append(float(x))
                except Exception:
                    res.append(default)
            return res
        else:
            try:
                if value != value or value is None:
                    return default
                return float(value)
            except Exception:
                return default

    def safe_datetime(value):
        if hasattr(value, 'astype'):
            try:
                return value.astype('datetime64[ns]')
            except Exception:
                pass
        return value

    def safe_divide(a, b, default=0.0):
        if abs(b) <= 1e-9 or is_nan(b) or is_nan(a):
            return default
        res = a / b
        if res == float('inf') or res == float('-inf') or is_nan(res):
            return default
        return res

    def safe_get_parameter(parameters, name, default):
        val = parameters.get(name, default)
        return safe_to_float(val, default=default)

    def std_dev(lst):
        if not lst:
            return 0.0
        mean_val = sum(lst) / len(lst)
        variance = sum((x - mean_val) ** 2 for x in lst) / len(lst)
        return variance ** 0.5

    def mean_lst(lst, default=0.0):
        return sum(lst) / len(lst) if lst else default

    def clip(val, low, high):
        return max(low, min(high, val))

    def to_jsonable(value):
        if isinstance(value, dict):
            return {k: to_jsonable(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [to_jsonable(v) for v in value]
        else:
            if hasattr(value, 'item'):
                try:
                    value = value.item()
                except Exception:
                    pass
            if value != value:
                return None
            if value == float('inf') or value == float('-inf'):
                return None
            if isinstance(value, (int, float)):
                return value
            return value

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DATA VALIDATION STAGE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    parameters = parameters or {}
    
    # Read tariff parameters via parameters.get to satisfy schema compliance
    _ = parameters.get("ToU", None)
    _ = parameters.get("FiT", None)

    if df is None or not hasattr(df, "columns"):
        raise ValueError("Input df is invalid.")

    required_cols = ["Timestamp", "member_id", "load", "generation", "member_type"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    df["Timestamp"] = safe_datetime(df["Timestamp"])
    df["load"] = safe_to_float(df["load"], default=0.0)
    df["generation"] = safe_to_float(df["generation"], default=0.0)

    # Check for invalid timestamps
    if df["Timestamp"].isna().any():
        df = df.dropna(subset=["Timestamp"])
    if df.empty:
        raise ValueError("No valid rows remaining after removing invalid Timestamps.")

    # Validate missing/blank member_id
    df["member_id"] = df["member_id"].astype(str).str.strip()
    if (df["member_id"] == "").any() or df["member_id"].isna().any():
        raise ValueError("member_id cannot be blank or missing.")

    # Sort by Timestamp and reset index
    df = df.sort_values(by="Timestamp").reset_index(drop=True)

    # Detect duplicate rows
    if df.duplicated(subset=["member_id", "Timestamp"]).any():
        raise ValueError("Duplicate rows detected for member_id and Timestamp.")

    # Validate tariffs
    if tariffs is None or "time" not in tariffs.columns:
        raise ValueError("Tariffs DataFrame must contain a 'time' column.")
    for tk in ["ToU", "FiT"]:
        if tk not in tariffs.columns:
            raise ValueError(f"Tariff key '{tk}' is missing.")
    
    # Convert time to int for hour matching
    tariffs_clean = tariffs.copy()
    time_ints = []
    for t_val in tariffs_clean["time"]:
        try:
            time_ints.append(int(float(t_val)))
        except Exception:
            time_ints.append(-1)
    tariffs_clean["time"] = time_ints
    tariffs_clean["ToU"] = safe_to_float(tariffs_clean["ToU"])
    tariffs_clean["FiT"] = safe_to_float(tariffs_clean["FiT"])
    if tariffs_clean[["ToU", "FiT"]].isna().any().any():
        raise ValueError("Tariff values contain invalid non-numeric data.")

    # Check non-negative load/generation
    if (df["load"] < 0).any() or (df["generation"] < 0).any():
        raise ValueError("Load and generation columns must be non-negative.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PARAMETER INITIALIZATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    alpha_mpas = safe_get_parameter(parameters, "alpha_mpas", 0.5)
    w_wam = safe_get_parameter(parameters, "w_wam", 0.6)
    theta_ipa = safe_get_parameter(parameters, "theta_ipa", 0.1)
    epsilon_ipa = safe_get_parameter(parameters, "epsilon_ipa", 0.01)
    alpha_mci_1 = safe_get_parameter(parameters, "alpha_mci_1", 0.3)
    alpha_mci_2 = safe_get_parameter(parameters, "alpha_mci_2", 0.3)
    alpha_mci_3 = safe_get_parameter(parameters, "alpha_mci_3", 0.3)
    alpha_mci_4 = safe_get_parameter(parameters, "alpha_mci_4", 0.01)
    alpha_mci_5 = safe_get_parameter(parameters, "alpha_mci_5", 0.05)
    MT = safe_get_parameter(parameters, "MT", 0.5)
    VT = safe_get_parameter(parameters, "VT", 0.7)
    MCT_low = safe_get_parameter(parameters, "MCT_low", 0.0)
    MCT_high = safe_get_parameter(parameters, "MCT_high", 1.2)
    omega_m1 = safe_get_parameter(parameters, "omega_m1", 0.5)
    omega_m2 = safe_get_parameter(parameters, "omega_m2", 0.5)

    # Map tariffs to the df
    tariff_map = tariffs_clean.set_index("time").to_dict()
    df["hour"] = df["Timestamp"].dt.hour
    df["ToU"] = df["hour"].map(tariff_map["ToU"])
    df["FiT"] = df["hour"].map(tariff_map["FiT"])

    if df["ToU"].isna().any() or df["FiT"].isna().any():
        raise ValueError("Tariff mapping resulted in missing values for some hours.")

    if (df["ToU"] <= df["FiT"]).any():
        raise ValueError("Constraint violated: ToU must be strictly greater than FiT.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CHRONOLOGICAL SIMULATION LOOP
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    unique_timestamps = sorted(df["Timestamp"].unique())
    
    # State tracking histories
    MP_history = []
    TV_history = []
    
    # Final metrics per timestep to collect
    ML_history = []
    MCI_history = []
    Selected_Mechanism_history = []
    
    # Executed trades flat container
    all_post_auction_trades = []

    for t in unique_timestamps:
        sub_df = df[df["Timestamp"] == t]
        ToU_t = float(sub_df["ToU"].iloc[0])
        FiT_t = float(sub_df["FiT"].iloc[0])

        # Step 1: Aligned load and generation per member. Calculate net position
        buyers_list = []
        sellers_list = []

        for _, row in sub_df.iterrows():
            load_i = float(row["load"])
            gen_i = float(row["generation"])
            net_i = load_i - gen_i
            member_id = row["member_id"]

            if net_i > 0:
                buyers_list.append({
                    "id": member_id,
                    "Q_Bi": net_i,
                    "P_Bi": ToU_t - 0.05
                })
            elif net_i < 0:
                sellers_list.append({
                    "id": member_id,
                    "Q_Sj": -net_i,
                    "P_Sj": FiT_t + 0.05
                })

        # Rank buyer bids descending and seller asks ascending
        buyers_list = sorted(buyers_list, key=lambda x: (-x["P_Bi"], x["id"]))
        sellers_list = sorted(sellers_list, key=lambda x: (x["P_Sj"], x["id"]))

        # Clear auction using standard Uniform Price mechanism
        init_buyers = [dict(b) for b in buyers_list]
        init_sellers = [dict(s) for s in sellers_list]

        TV_t = 0.0
        matched_participants = set()
        last_match_price = None

        b_idx = 0
        s_idx = 0
        while b_idx < len(init_buyers) and s_idx < len(init_sellers):
            b = init_buyers[b_idx]
            s = init_sellers[s_idx]

            if b["P_Bi"] < s["P_Sj"]:
                break

            if b["Q_Bi"] <= 1e-9:
                b_idx += 1
                continue
            if s["Q_Sj"] <= 1e-9:
                s_idx += 1
                continue

            match_q = min(b["Q_Bi"], s["Q_Sj"])
            TV_t += match_q
            matched_participants.add(b["id"])
            matched_participants.add(s["id"])

            last_match_price = (b["P_Bi"] + s["P_Sj"]) / 2.0

            b["Q_Bi"] -= match_q
            s["Q_Sj"] -= match_q

            if b["Q_Bi"] <= 1e-9:
                b_idx += 1
            if s["Q_Sj"] <= 1e-9:
                s_idx += 1

        NAP_t = float(len(matched_participants))
        MP_t = float(last_match_price if last_match_price is not None else FiT_t)

        MP_history.append(MP_t)
        TV_history.append(TV_t)

        # Post-auction remaining unmatched quantities
        post_buyers = [b for b in init_buyers if b["Q_Bi"] > 1e-9]
        post_sellers = [s for s in init_sellers if s["Q_Sj"] > 1e-9]

        UD_t = float(sum(b["Q_Bi"] for b in post_buyers))
        S_t = float(sum(s["Q_Sj"] for s in post_sellers))

        # eq_ml | ML = TV / NAP
        ML_t = safe_divide(TV_t, NAP_t, default=0.0)
        ML_history.append(ML_t)

        # Compute price volatility PV
        PV_t = std_dev(MP_history[-10:]) if MP_history else 0.0

        # Compute bidding strategy aggression BS
        avg_P_Bi = mean_lst([b["P_Bi"] for b in buyers_list], default=ToU_t - 0.05)
        avg_P_Sj = mean_lst([s["P_Sj"] for s in sellers_list], default=FiT_t + 0.05)
        BS_t = avg_P_Bi - avg_P_Sj

        # Compute historical transaction metric HTD
        HTD_t = mean_lst(TV_history[-24:], default=0.0)

        # eq_mci | MCI = alpha_mci_1 * (UD - S) + alpha_mci_2 * ML + alpha_mci_3 * PV + alpha_mci_4 * BS + alpha_mci_5 * HTD
        MCI_t = (alpha_mci_1 * (UD_t - S_t) +
                 alpha_mci_2 * ML_t +
                 alpha_mci_3 * PV_t +
                 alpha_mci_4 * BS_t +
                 alpha_mci_5 * HTD_t)
        MCI_history.append(MCI_t)

        # Determine Selected_Mechanism based on decision rules
        if ML_t < MT:
            if PV_t > VT:
                if MCI_t > MCT_high:
                    Selected_Mechanism_t = 'IPA'
                elif MCT_low < MCI_t <= MCT_high:
                    Selected_Mechanism_t = 'WAM'
                else:
                    Selected_Mechanism_t = 'CFRM'
            else:
                Selected_Mechanism_t = 'APM'
        else:
            if PV_t <= VT:
                if MCI_t > MCT_high:
                    Selected_Mechanism_t = 'IPA'
                elif MCT_low < MCI_t <= MCT_high:
                    Selected_Mechanism_t = 'MPAS'
                else:
                    Selected_Mechanism_t = 'NBS'
            else:
                if MCI_t > MCT_high:
                    Selected_Mechanism_t = 'COLM'
                elif MCT_low < MCI_t <= MCT_high:
                    Selected_Mechanism_t = 'MMP'
                else:
                    Selected_Mechanism_t = 'CGT'
        
        Selected_Mechanism_history.append(Selected_Mechanism_t)

        # Match remaining unmatched quantities sequentially and calculate P_ij and Q_ij
        b_idx = 0
        s_idx = 0
        
        while b_idx < len(post_buyers) and s_idx < len(post_sellers):
            buyer = post_buyers[b_idx]
            seller = post_sellers[s_idx]

            q_bi = buyer["Q_Bi"]
            q_sj = seller["Q_Sj"]

            if q_bi <= 1e-9:
                b_idx += 1
                continue
            if q_sj <= 1e-9:
                s_idx += 1
                continue

            # eq_matching | if Q_Bi < Q_Sj: Q_ij = Q_Bi else if Q_Bi > Q_Sj: Q_ij = Q_Sj else: Q_ij = Q_Bi
            if q_bi < q_sj:
                q_ij = q_bi
            elif q_bi > q_sj:
                q_ij = q_sj
            else:
                q_ij = q_bi

            p_bi = buyer["P_Bi"]
            p_sj = seller["P_Sj"]

            # Compute alternative pricing models for this pair
            # eq_vcg | P_ij_vcg = (clip(MP + (P_Sj - P_Bi), FiT, ToU) + clip(MP - (P_Bi - P_Sj), FiT, ToU)) / 2
            p_ij_vcg = (clip(MP_t + (p_sj - p_bi), FiT_t, ToU_t) + clip(MP_t - (p_bi - p_sj), FiT_t, ToU_t)) / 2.0

            # eq_cgt | P_ij_cgt = 0.4 * P_Bi + 0.6 * P_Sj
            p_ij_cgt = 0.4 * p_bi + 0.6 * p_sj

            # eq_colm | P_ij_colm = clip((P_Bi + P_Sj) / 2, FiT, ToU)
            p_ij_colm = clip((p_bi + p_sj) / 2.0, FiT_t, ToU_t)

            # eq_dmrl | P_ij_dmrl = clip((P_Bi + P_Sj) / 2 - 0.05, FiT, ToU)
            p_ij_dmrl = clip((p_bi + p_sj) / 2.0 - 0.05, FiT_t, ToU_t)

            # Compute P_ij based on Selected_Mechanism_t
            if Selected_Mechanism_t in ['APM', 'NBS']:
                # eq_apm | P_ij = (P_Bi + P_Sj) / 2
                p_ij = (p_bi + p_sj) / 2.0
            elif Selected_Mechanism_t == 'MPAS':
                # eq_mpas | P_ij = MP + alpha_mpas * ((P_Bi - P_Sj) / 2)
                p_ij = MP_t + alpha_mpas * ((p_bi - p_sj) / 2.0)
            elif Selected_Mechanism_t == 'CFRM':
                # eq_cfrm | P_ij = (ToU + FiT) / 2
                p_ij = (ToU_t + FiT_t) / 2.0
            elif Selected_Mechanism_t == 'WAM':
                # eq_wam | P_ij = w_wam * P_Bi + (1 - w_wam) * P_Sj
                p_ij = w_wam * p_bi + (1.0 - w_wam) * p_sj
            elif Selected_Mechanism_t == 'MMP':
                # eq_mmp | P_ij = min(ToU, max(FiT, MP))
                p_ij = min(ToU_t, max(FiT_t, MP_t))
            elif Selected_Mechanism_t == 'VCG':
                p_ij = p_ij_vcg
            elif Selected_Mechanism_t == 'CGT':
                p_ij = p_ij_cgt
            elif Selected_Mechanism_t == 'COLM':
                p_ij = p_ij_colm
            elif Selected_Mechanism_t == 'DMRL':
                p_ij = p_ij_dmrl
            elif Selected_Mechanism_t == 'IPA':
                # eq_ipa | P_ij = P_ij_k + theta_ipa * ((P_Bi - P_ij_k) + (P_ij_k - P_Sj)) / 2
                p_k = MP_t
                for _ in range(100):
                    diff = theta_ipa * ((p_bi - p_k) + (p_k - p_sj)) / 2.0
                    p_next = p_k + diff
                    p_next = clip(p_next, FiT_t, ToU_t)
                    if abs(p_next - p_k) < epsilon_ipa:
                        p_k = p_next
                        break
                    p_k = p_next
                p_ij = p_k
            elif Selected_Mechanism_t == 'Hybrid':
                p_ij_simple = w_wam * p_bi + (1.0 - w_wam) * p_sj
                p_k = MP_t
                for _ in range(100):
                    diff = theta_ipa * ((p_bi - p_k) + (p_k - p_sj)) / 2.0
                    p_next = p_k + diff
                    p_next = clip(p_next, FiT_t, ToU_t)
                    if abs(p_next - p_k) < epsilon_ipa:
                        p_k = p_next
                        break
                    p_k = p_next
                p_ij_complex = p_k
                p_ij = omega_m1 * p_ij_simple + omega_m2 * p_ij_complex
            else:
                p_ij = (p_bi + p_sj) / 2.0

            # Record post-auction transaction
            trade_record = {
                "Timestamp": t,
                "buyer_id": buyer["id"],
                "seller_id": seller["id"],
                "Q_ij": q_ij,
                "P_ij": p_ij,
                "P_ij_vcg": p_ij_vcg,
                "P_ij_cgt": p_ij_cgt,
                "P_ij_colm": p_ij_colm,
                "P_ij_dmrl": p_ij_dmrl,
                "ToU": ToU_t,
                "FiT": FiT_t
            }
            all_post_auction_trades.append(trade_record)

            buyer["Q_Bi"] -= q_ij
            seller["Q_Sj"] -= q_ij

            if buyer["Q_Bi"] <= 1e-9:
                b_idx += 1
            if seller["Q_Sj"] <= 1e-9:
                s_idx += 1

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # METRICS EVALUATION & AGGREGATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # eq_benefit_tou | Benefit_to_ToU = sum(Q_ij * (ToU - P_ij))
    Benefit_to_ToU = sum(trade["Q_ij"] * (trade["ToU"] - trade["P_ij"]) for trade in all_post_auction_trades)
    
    # eq_benefit_fit | Benefit_to_FiT = sum(Q_ij * (P_ij - FiT))
    Benefit_to_FiT = sum(trade["Q_ij"] * (trade["P_ij"] - trade["FiT"]) for trade in all_post_auction_trades)

    denominator_buyers = sum(trade["Q_ij"] * trade["ToU"] for trade in all_post_auction_trades)
    denominator_sellers = sum(trade["Q_ij"] * trade["FiT"] for trade in all_post_auction_trades)
    total_Q_ij = sum(trade["Q_ij"] for trade in all_post_auction_trades)

    Cost_Savings_Buyers = float(safe_divide(Benefit_to_ToU, denominator_buyers, default=0.0) * 100.0)
    Revenue_Increase_Sellers = float(safe_divide(Benefit_to_FiT, denominator_sellers, default=0.0) * 100.0)
    
    weighted_price_sum = sum(trade["Q_ij"] * trade["P_ij"] for trade in all_post_auction_trades)
    Mean_Traded_Price = float(safe_divide(weighted_price_sum, total_Q_ij, default=0.0))

    results = {
        "ML": ML_history,
        "MCI": MCI_history,
        "Selected_Mechanism": Selected_Mechanism_history,
        "P_ij": [trade["P_ij"] for trade in all_post_auction_trades],
        "P_ij_vcg": [trade["P_ij_vcg"] for trade in all_post_auction_trades],
        "P_ij_cgt": [trade["P_ij_cgt"] for trade in all_post_auction_trades],
        "P_ij_colm": [trade["P_ij_colm"] for trade in all_post_auction_trades],
        "P_ij_dmrl": [trade["P_ij_dmrl"] for trade in all_post_auction_trades],
        "Q_ij": [trade["Q_ij"] for trade in all_post_auction_trades],
        "Benefit_to_ToU": Benefit_to_ToU,
        "Benefit_to_FiT": Benefit_to_FiT,
        "Cost_Savings_Buyers": Cost_Savings_Buyers,
        "Revenue_Increase_Sellers": Revenue_Increase_Sellers,
        "Mean_Traded_Price": Mean_Traded_Price
    }

    return to_jsonable(results)


# -- CLI entry-point ----------------------------------------------------------

def _print_summary(results: dict) -> None:
    print("\n--- Simulation results " + "-" * 42)
    for key, value in results.items():
        if isinstance(value, (int, float)):
            print(f"  {key:30s} = {value:.4f}")
        elif isinstance(value, pd.Series):
            print(f"  {key:30s}   Series  len={len(value)}  "
                  f"sum={value.sum():.4f}  mean={value.mean():.4f}")
        elif isinstance(value, pd.DataFrame):
            print(f"  {key:30s}   DataFrame  shape={value.shape}")
        elif isinstance(value, dict):
            print(f"  {key:30s}   dict  keys={list(value)[:5]}")
        else:
            print(f"  {key:30s} = {str(value)[:80]}")
    print("-" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BM simulation.")
    parser.add_argument("--dataset",  type=Path, required=True,
                        help="Path to the long-format CSV dataset")
    parser.add_argument("--tariffs",  type=Path, required=True,
                        help="Path to tariffs.csv")
    parser.add_argument("--param",    action="append", default=[],
                        metavar="name=value",
                        help="Override a model parameter, e.g. --param alpha=0.7")
    args = parser.parse_args()

    if not args.dataset.exists():
        sys.exit(f"Dataset not found: {args.dataset}")
    if not args.tariffs.exists():
        sys.exit(f"Tariffs not found: {args.tariffs}")

    df      = pd.read_csv(args.dataset)
    tariffs = pd.read_excel(args.tariffs) if args.tariffs.read_bytes()[:2] == b"PK" else pd.read_csv(args.tariffs)

    parameter_overrides: dict = {}
    for item in args.param:
        if "=" not in item:
            sys.exit(f"--param must be name=value, got: {item!r}")
        name, _, raw = item.partition("=")
        try:
            parameter_overrides[name.strip()] = json.loads(raw.strip())
        except json.JSONDecodeError:
            try:
                parameter_overrides[name.strip()] = float(raw.strip())
            except ValueError:
                parameter_overrides[name.strip()] = raw.strip()

    print(f"Dataset : {args.dataset}  ({len(df):,} rows)")
    print(f"Tariffs : {args.tariffs}")
    if parameter_overrides:
        print(f"Overrides: {parameter_overrides}")

    results = simulate(df, tariffs, parameter_overrides)
    _print_summary(results)


if __name__ == "__main__":
    main()
