"""simulate_newton_raphson_pricing.py
Two-step pricing mechanism for local electricity markets (LEM) using Uniform Pricing (UP) with price caps [...]

Generated from: output/c377cf21603ae0f5_ESR_Newton-Raphson/c377cf21603ae0f5_ESR_Newton-Raphson.pdf
BM name       : Newton-Raphson pricing

Usage
-----
    python simulate_newton_raphson_pricing.py \
        --dataset  path/to/dataset.csv \
        --tariffs  path/to/tariffs.csv \
        [--param   name=value ...]

Or import and call directly:

    import pandas as pd
    from simulate_newton_raphson_pricing import simulate

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
    import numpy as np
    import pandas as pd

    # Define helper functions near the top
    def safe_numeric(value, default=None):
        if isinstance(value, pd.Series):
            res = pd.to_numeric(value, errors="coerce")
            if default is not None:
                res = res.fillna(default)
            return res
        else:
            try:
                val = pd.to_numeric(value, errors="coerce")
                if pd.isna(val):
                    return default
                return val
            except Exception:
                return default

    def safe_datetime(value):
        return pd.to_datetime(value, errors="coerce")

    def safe_divide(a, b, default=0.0):
        a_num = safe_numeric(a, default=0.0)
        b_num = safe_numeric(b, default=0.0)
        if isinstance(a_num, pd.Series) or isinstance(b_num, pd.Series):
            cond = (b_num.abs() <= 1e-9) | b_num.isna()
            res = np.where(cond, default, a_num / b_num)
            res = np.where(np.isfinite(res), res, default)
            return pd.Series(res, index=b_num.index if isinstance(b_num, pd.Series) else a_num.index)
        else:
            if abs(b_num) <= 1e-9 or pd.isna(b_num):
                return default
            res = a_num / b_num
            if not np.isfinite(res):
                return default
            return res

    def to_jsonable(value):
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        elif isinstance(value, pd.Series):
            d = []
            for k, v in value.items():
                v_val = None if pd.isna(v) or not np.isfinite(v) else v
                if isinstance(v_val, (np.integer, np.floating)):
                    v_val = float(v_val) if isinstance(v_val, np.floating) else int(v_val)
                d.append(v_val)
            return d
        elif isinstance(value, np.ndarray):
            return [None if pd.isna(x) else x for x in value.tolist()]
        elif isinstance(value, (int, float, np.integer, np.floating)):
            if pd.isna(value) or not np.isfinite(value):
                return None
            return float(value) if isinstance(value, (float, np.floating)) else int(value)
        elif isinstance(value, dict):
            return {str(k): to_jsonable(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [to_jsonable(v) for v in value]
        elif pd.isna(value):
            return None
        else:
            return value

    # DATA VALIDATION STAGE
    if parameters is None:
        parameters = {}
    
    if df is None or not hasattr(df, "columns"):
        raise ValueError("Input df must be a pandas DataFrame")
        
    required_cols = ["Timestamp", "member_id", "load", "generation"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
        
    df = df.copy()
    
    df["Timestamp"] = safe_datetime(df["Timestamp"])
    df["load"] = safe_numeric(df["load"], default=0.0)
    df["generation"] = safe_numeric(df["generation"], default=0.0)
    
    df = df.dropna(subset=["Timestamp"])
    if len(df) == 0:
        raise ValueError("No valid rows remaining after filtering invalid Timestamps")
        
    df["member_id"] = df["member_id"].astype(str).str.strip()
    if df["member_id"].isna().any() or (df["member_id"] == "").any() or (df["member_id"] == "nan").any():
        raise ValueError("Some member_id values are missing or blank")
        
    df = df.sort_values(by="Timestamp").reset_index(drop=True)
    
    if df.duplicated(subset=["member_id", "Timestamp"]).any():
        raise ValueError("Duplicate rows detected for member_id and Timestamp")
        
    if tariffs is None or not hasattr(tariffs, "columns") or "time" not in tariffs.columns:
        raise ValueError("tariffs must be a DataFrame-like object with a 'time' column")
        
    for tk in ["ToU", "FiT"]:
        if tk not in tariffs.columns:
            raise ValueError(f"Tariff key column {tk} is missing in tariffs")
        tariffs[tk] = safe_numeric(tariffs[tk], default=None)
        if tariffs[tk].isna().any():
            raise ValueError(f"Tariff key column {tk} has missing or invalid numeric values")
            
    hours_in_df = df["Timestamp"].dt.hour.unique()
    tariffs_hours = tariffs["time"].unique()
    missing_hours = [h for h in hours_in_df if h not in tariffs_hours]
    if missing_hours:
        raise ValueError(f"Tariffs are missing for hours: {missing_hours}")

    # Map tariffs to DataFrame rows
    tariffs_df = tariffs.set_index("time")
    df["ToU"] = df["Timestamp"].dt.hour.map(tariffs_df["ToU"]).fillna(0.0)
    df["FiT"] = df["Timestamp"].dt.hour.map(tariffs_df["FiT"]).fillna(0.0)

    # Eq_1 | D_mt = max(0, load - generation)
    df["D_mt"] = np.maximum(0.0, df["load"] - df["generation"])

    # Eq_2 | S_mt = max(0, generation - load)
    df["S_mt"] = np.maximum(0.0, df["generation"] - df["load"])

    # Step 2: Bidding and Offer Price Generation
    rs = np.random.RandomState(42)
    u = rs.uniform(0.0, 1.0, len(df))
    df["P_bid_mt"] = df["FiT"] + u * (df["ToU"] - df["FiT"])
    v = rs.uniform(0.0, 1.0, len(df))
    df["P_ask_mt"] = df["FiT"] + v * (df["ToU"] - df["FiT"])

    # Eq_3 | Sort buyers descending by P_bid_mt and sellers ascending by P_ask_mt. P_UP_t is the price where cumulative demand intersects cumulative supply, i.e., Sum(D_mt * I(P_bid_mt >= P_UP_t)) >= Sum(S_mt * I(P_ask_mt <= P_UP_t)) and the prior element constraint holds.
    P_UP_dict = {}
    grouped = df.groupby("Timestamp")
    for ts, group in grouped:
        bids_t = group[group["D_mt"] > 0]
        asks_t = group[group["S_mt"] > 0]
        ToU_t = group["ToU"].iloc[0]
        FiT_t = group["FiT"].iloc[0]
        
        if len(bids_t) == 0 or len(asks_t) == 0:
            P_UP_t = 0.5 * (ToU_t + FiT_t)
        else:
            candidate_prices = sorted(list(set(bids_t["P_bid_mt"]).union(set(asks_t["P_ask_mt"]))))
            valid_prices = []
            for p in candidate_prices:
                dem = bids_t[bids_t["P_bid_mt"] >= p]["D_mt"].sum()
                sup = asks_t[asks_t["P_ask_mt"] <= p]["S_mt"].sum()
                if dem >= sup:
                    valid_prices.append(p)
            if len(valid_prices) > 0:
                P_UP_t = max(valid_prices)
            else:
                P_UP_t = min(candidate_prices)
                
        P_UP_dict[ts] = P_UP_t

    df["P_UP_t"] = df["Timestamp"].map(P_UP_dict)

    # Compute QT_UP_b_mt and QT_UP_s_mt based on clearing price P_UP_t
    QT_UP_b_vals = []
    QT_UP_s_vals = []

    for ts, group in grouped:
        P_UP_t = P_UP_dict[ts]
        active_b_mask = (group["D_mt"] > 0) & (group["P_bid_mt"] >= P_UP_t)
        active_s_mask = (group["S_mt"] > 0) & (group["P_ask_mt"] <= P_UP_t)
        
        D_active = group.loc[active_b_mask, "D_mt"].sum()
        S_active = group.loc[active_s_mask, "S_mt"].sum()
        
        Q_cleared = min(D_active, S_active)
        
        for idx, row in group.iterrows():
            if active_b_mask.loc[idx]:
                val_b = row["D_mt"] * safe_divide(Q_cleared, D_active)
            else:
                val_b = 0.0
            QT_UP_b_vals.append((idx, val_b))
            
            if active_s_mask.loc[idx]:
                val_s = row["S_mt"] * safe_divide(Q_cleared, S_active)
            else:
                val_s = 0.0
            QT_UP_s_vals.append((idx, val_s))

    QT_UP_b_df = pd.DataFrame(QT_UP_b_vals, columns=["idx", "QT_UP_b_mt"]).set_index("idx")
    QT_UP_s_df = pd.DataFrame(QT_UP_s_vals, columns=["idx", "QT_UP_s_mt"]).set_index("idx")

    df["QT_UP_b_mt"] = QT_UP_b_df["QT_UP_b_mt"]
    df["QT_UP_s_mt"] = QT_UP_s_df["QT_UP_s_mt"]

    # Eq_4 | Unmatched_D_mt = D_mt - QT_UP_b_mt
    df["Unmatched_D_mt"] = df["D_mt"] - df["QT_UP_b_mt"]

    # Eq_5 | Unmatched_S_mt = S_mt - QT_UP_s_mt
    df["Unmatched_S_mt"] = df["S_mt"] - df["QT_UP_s_mt"]

    # Eq_QMaxT | QMaxT_t = min(Sum_m(Unmatched_D_mt), Sum_m(Unmatched_S_mt))
    QMaxT_dict = {}
    for ts, group in df.groupby("Timestamp"):
        QMaxT_dict[ts] = min(group["Unmatched_D_mt"].sum(), group["Unmatched_S_mt"].sum())
    df["QMaxT_t"] = df["Timestamp"].map(QMaxT_dict)

    # Eq_8 initialization
    df["P_s_b_mt"] = 0.0

    # Secondary Trading Setup
    beta_dict = {}
    P_beta_dict = {}
    df["QT_step2_b_mt"] = 0.0
    df["QT_step2_s_mt"] = 0.0
    df["P_Tx_mt"] = 0.0

    mode = parameters.get("mode", "PPA")
    alpha = parameters.get("alpha", 1.0)
    max_iter = parameters.get("max_iter", 100)
    tol = parameters.get("tol", 1e-5)

    for ts, group in df.groupby("Timestamp"):
        ToU_t = group["ToU"].iloc[0]
        FiT_t = group["FiT"].iloc[0]
        Q_max_t = QMaxT_dict[ts]
        
        unmatched_b = group[group["Unmatched_D_mt"] > 0]
        unmatched_s = group[group["Unmatched_S_mt"] > 0]
        
        N_b = len(unmatched_b)
        N_s = len(unmatched_s)
        
        if N_b == 0 or N_s == 0 or Q_max_t <= 0:
            P_beta_t = 0.5 * (ToU_t + FiT_t)
            beta_t = 0.0
            beta_dict[ts] = beta_t
            P_beta_dict[ts] = P_beta_t
            continue
            
        mean_bid = unmatched_b["P_bid_mt"].mean()
        mean_ask = unmatched_s["P_ask_mt"].mean()
        
        # Eq_6 | beta_i = beta_{i-1} - f(beta_{i-1})/f'(beta_{i-1}) where f(beta) = min(Sum(Unmatched_D_mt), Sum(Unmatched_S_mt)) - min(D_beta, S_beta)
        beta_val = 0.0
        k = 0.5 * (mean_bid - mean_ask)
        
        for _ in range(max_iter):
            # Eq_7 | P_beta_t = 0.5 * ( (1/Nb) * Sum( (1 + beta)*P_bid_mt ) + (1/Ns) * Sum( (1 - beta)*P_ask_mt ) )
            P_beta = 0.5 * ((1.0 + beta_val) * mean_bid + (1.0 - beta_val) * mean_ask)
            
            exp_b = np.clip(alpha * (P_beta - unmatched_b["P_bid_mt"]), -50, 50)
            exp_s = np.clip(alpha * (unmatched_s["P_ask_mt"] - P_beta), -50, 50)
            
            d_i = unmatched_b["Unmatched_D_mt"] / (1.0 + np.exp(exp_b))
            s_j = unmatched_s["Unmatched_S_mt"] / (1.0 + np.exp(exp_s))
            
            D_beta = d_i.sum()
            S_beta = s_j.sum()
            
            f_beta = Q_max_t - min(D_beta, S_beta)
            
            d_i_prime = -alpha * k * d_i * (1.0 - safe_divide(d_i, unmatched_b["Unmatched_D_mt"]))
            s_j_prime = alpha * k * s_j * (1.0 - safe_divide(s_j, unmatched_s["Unmatched_S_mt"]))
            
            if D_beta < S_beta:
                f_prime = -d_i_prime.sum()
            else:
                f_prime = -s_j_prime.sum()
                
            if abs(f_prime) < 1e-12:
                break
                
            beta_new = beta_val - f_beta / f_prime
            beta_new = np.clip(beta_new, -1.0, 1.0)
            
            if abs(beta_new - beta_val) < tol:
                beta_val = beta_new
                break
            beta_val = beta_new
            
        beta_t = beta_val
        # Eq_7 | P_beta_t = 0.5 * ( (1/Nb) * Sum( (1 + beta)*P_bid_mt ) + (1/Ns) * Sum( (1 - beta)*P_ask_mt ) )
        P_beta_t = 0.5 * ((1.0 + beta_t) * mean_bid + (1.0 - beta_t) * mean_ask)
            
        beta_dict[ts] = beta_t
        P_beta_dict[ts] = P_beta_t
        
        if mode == "PPA":
            exp_b_final = np.clip(alpha * (P_beta_t - unmatched_b["P_bid_mt"]), -50, 50)
            exp_s_final = np.clip(alpha * (unmatched_s["P_ask_mt"] - P_beta_t), -50, 50)
            
            d_i_final = unmatched_b["Unmatched_D_mt"] / (1.0 + np.exp(exp_b_final))
            s_j_final = unmatched_s["Unmatched_S_mt"] / (1.0 + np.exp(exp_s_final))
            
            D_total = d_i_final.sum()
            S_total = s_j_final.sum()
            
            Q_cleared_step2 = min(D_total, S_total)
            
            for idx, val in d_i_final.items():
                df.at[idx, "QT_step2_b_mt"] = val * safe_divide(Q_cleared_step2, D_total)
            for idx, val in s_j_final.items():
                df.at[idx, "QT_step2_s_mt"] = val * safe_divide(Q_cleared_step2, S_total)
                
            df.loc[unmatched_b.index, "P_Tx_mt"] = P_beta_t
            df.loc[unmatched_s.index, "P_Tx_mt"] = P_beta_t
            
        elif mode == "P2P":
            unmatched_b_sorted = unmatched_b.sort_values(by="P_bid_mt", ascending=False).copy()
            unmatched_s_sorted = unmatched_s.sort_values(by="P_ask_mt", ascending=True).copy()
            
            b_idx = 0
            s_idx = 0
            while b_idx < len(unmatched_b_sorted) and s_idx < len(unmatched_s_sorted):
                b_row = unmatched_b_sorted.iloc[b_idx]
                s_row = unmatched_s_sorted.iloc[s_idx]
                
                b_id = unmatched_b_sorted.index[b_idx]
                s_id = unmatched_s_sorted.index[s_idx]
                
                # Check price compatibility
                if b_row["P_bid_mt"] >= s_row["P_ask_mt"]:
                    qty = min(unmatched_b_sorted.at[b_id, "Unmatched_D_mt"], unmatched_s_sorted.at[s_id, "Unmatched_S_mt"])
                    if qty > 0:
                        df.at[b_id, "QT_step2_b_mt"] += qty
                        df.at[s_id, "QT_step2_s_mt"] += qty
                        
                        # Eq_8 | P_s_b_mt = 0.5 * (P_ask_mt + P_bid_mt) for adjacent pairs in sorted unmatched lists
                        p_match = 0.5 * (b_row["P_bid_mt"] + s_row["P_ask_mt"])
                        df.at[b_id, "P_s_b_mt"] = p_match
                        df.at[s_id, "P_s_b_mt"] = p_match
                        df.at[b_id, "P_Tx_mt"] = p_match
                        df.at[s_id, "P_Tx_mt"] = p_match
                        
                        unmatched_b_sorted.at[b_id, "Unmatched_D_mt"] -= qty
                        unmatched_s_sorted.at[s_id, "Unmatched_S_mt"] -= qty
                    
                    if unmatched_b_sorted.at[b_id, "Unmatched_D_mt"] <= 1e-9:
                        b_idx += 1
                    if unmatched_s_sorted.at[s_id, "Unmatched_S_mt"] <= 1e-9:
                        s_idx += 1
                else:
                    break

    df["beta"] = df["Timestamp"].map(beta_dict)
    df["P_beta_t"] = df["Timestamp"].map(P_beta_dict)

    # Summed quantities
    df["QT_B_mt"] = df["QT_UP_b_mt"] + df["QT_step2_b_mt"]
    df["QT_S_mt"] = df["QT_UP_s_mt"] + df["QT_step2_s_mt"]

    # Eq_9 | PayBack_b_mt = QT_step2_b_mt * (P_UP_t - P_Tx_mt) if P_Tx_mt < P_UP_t else 0
    df["PayBack_b_mt"] = np.where(df["P_Tx_mt"] < df["P_UP_t"], df["QT_step2_b_mt"] * (df["P_UP_t"] - df["P_Tx_mt"]), 0.0)

    # Eq_10 | PayBack_s_mt = QT_step2_s_mt * (P_Tx_mt - P_UP_t) if P_Tx_mt > P_UP_t else 0
    df["PayBack_s_mt"] = np.where(df["P_Tx_mt"] > df["P_UP_t"], df["QT_step2_s_mt"] * (df["P_Tx_mt"] - df["P_UP_t"]), 0.0)

    # Eq_11 | Redistribution_t = (Sum_m(PayBack_b_mt) + Sum_m(PayBack_s_mt)) / N_participants
    payback_sums = df.groupby("Timestamp")[["PayBack_b_mt", "PayBack_s_mt"]].transform("sum")
    N_participants = df["member_id"].nunique()
    df["Redistribution_t"] = (payback_sums["PayBack_b_mt"] + payback_sums["PayBack_s_mt"]) / N_participants

    # Eq_12 | IPay_mt = (D_mt * ToU) if load >= generation else (S_mt * FiT)
    df["IPay_mt"] = np.where(df["load"] >= df["generation"], df["D_mt"] * df["ToU"], df["S_mt"] * df["FiT"])

    # Eq_13 | FPay_mt = (QT_B_mt * P_UP_t + (D_mt - QT_B_mt) * ToU - Redistribution_t) if load >= generation else (QT_S_mt * P_UP_t + (S_mt - QT_S_mt) * FiT + Redistribution_t) (with adjustments based on P_Tx_mt for secondary trading matches)
    df["FPay_mt"] = np.where(
        df["load"] >= df["generation"],
        (df["QT_UP_b_mt"] * df["P_UP_t"] + df["QT_step2_b_mt"] * df["P_Tx_mt"] + (df["D_mt"] - df["QT_B_mt"]) * df["ToU"] - df["Redistribution_t"]),
        (df["QT_UP_s_mt"] * df["P_UP_t"] + df["QT_step2_s_mt"] * df["P_Tx_mt"] + (df["S_mt"] - df["QT_S_mt"]) * df["FiT"] + df["Redistribution_t"])
    )

    # Eq_14 | CST_m = ((IPay_mt - FPay_mt)/IPay_mt)*100 for buyers, and ((FPay_mt - IPay_mt)/IPay_mt)*100 for sellers (integrated over evaluation horizon)
    member_sums = df.groupby("member_id")[["load", "generation", "IPay_mt", "FPay_mt"]].sum()
    CST_m_series = []
    for m in member_sums.index:
        load_m = member_sums.loc[m, "load"]
        gen_m = member_sums.loc[m, "generation"]
        ipay_m = member_sums.loc[m, "IPay_mt"]
        fpay_m = member_sums.loc[m, "FPay_mt"]
        
        if ipay_m == 0:
            cst = 0.0
        else:
            if load_m >= gen_m:
                cst = ((ipay_m - fpay_m) / ipay_m) * 100.0
            else:
                cst = ((fpay_m - ipay_m) / ipay_m) * 100.0
        CST_m_series.append((m, cst))
    CST_m_df = pd.DataFrame(CST_m_series, columns=["member_id", "CST_m"]).set_index("member_id")
    df["CST_m"] = df["member_id"].map(CST_m_df["CST_m"])

    # Eq_15 | GDI = (Sum_t(Sum_m(load_mt - QT_B_mt)) / Sum_t(Sum_m(load_mt))) * 100
    sum_load = df["load"].sum()
    sum_net_load = (df["load"] - df["QT_B_mt"]).sum()
    GDI = safe_divide(sum_net_load, sum_load) * 100.0

    # Eq_16 | STR = Sum_t(Sum_m(QT_S_mt)) / Sum_t(Sum_m(S_mt))
    STR = safe_divide(df["QT_S_mt"].sum(), df["S_mt"].sum())

    # Eq_17 | DCR = Sum_t(Sum_m(QT_B_mt)) / Sum_t(Sum_m(D_mt))
    DCR = safe_divide(df["QT_B_mt"].sum(), df["D_mt"].sum())

    # Eq_18 | ETR = Sum_t(Sum_m(QT_B_mt)) / Sum_t(Sum_m(load_mt))
    ETR = safe_divide(df["QT_B_mt"].sum(), df["load"].sum())

    # Eq_19 | TEI = Sum_t(Sum_m(FPay_mt)) / Sum_t(Sum_m(generation_mt))
    TEI = safe_divide(df["FPay_mt"].sum(), df["generation"].sum())

    # Eq_20 | CST = ((Sum_t(Sum_m(IPay_mt)) - Sum_t(Sum_m(FPay_mt))) / Sum_t(Sum_m(IPay_mt))) * 100
    sum_ipay = df["IPay_mt"].sum()
    sum_fpay = df["FPay_mt"].sum()
    CST = safe_divide(sum_ipay - sum_fpay, sum_ipay) * 100.0

    # Eq_21 | Gini = Sum_{m=1}^N Sum_{n=1}^N |CST_m - CST_n| / (2 * N^2 * mean(CST_m))
    cst_vals = CST_m_df["CST_m"].values
    N_members = len(cst_vals)
    if N_members <= 1 or np.mean(cst_vals) == 0:
        Gini = 0.0
    else:
        diff_sum = np.sum(np.abs(cst_vals[:, None] - cst_vals[None, :]))
        Gini = safe_divide(diff_sum, 2.0 * (N_members ** 2) * np.mean(cst_vals))

    results = {
        "CST": to_jsonable(CST),
        "CST_m": to_jsonable(df["CST_m"]),
        "DCR": to_jsonable(DCR),
        "D_mt": to_jsonable(df["D_mt"]),
        "ETR": to_jsonable(ETR),
        "FPay_mt": to_jsonable(df["FPay_mt"]),
        "GDI": to_jsonable(GDI),
        "Gini": to_jsonable(Gini),
        "IPay_mt": to_jsonable(df["IPay_mt"]),
        "P_UP_t": to_jsonable(df["P_UP_t"]),
        "P_beta_t": to_jsonable(df["P_beta_t"]),
        "P_s_b_mt": to_jsonable(df["P_s_b_mt"]),
        "PayBack_b_mt": to_jsonable(df["PayBack_b_mt"]),
        "PayBack_s_mt": to_jsonable(df["PayBack_s_mt"]),
        "QMaxT_t": to_jsonable(df["QMaxT_t"]),
        "QT_B_mt": to_jsonable(df["QT_B_mt"]),
        "QT_S_mt": to_jsonable(df["QT_S_mt"]),
        "Redistribution_t": to_jsonable(df["Redistribution_t"]),
        "STR": to_jsonable(STR),
        "S_mt": to_jsonable(df["S_mt"]),
        "TEI": to_jsonable(TEI),
        "Unmatched_D_mt": to_jsonable(df["Unmatched_D_mt"]),
        "Unmatched_S_mt": to_jsonable(df["Unmatched_S_mt"]),
        "beta": to_jsonable(df["beta"])
    }

    return results


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
