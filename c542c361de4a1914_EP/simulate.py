"""simulate_p2p_trading.py
A bi-level sequential optimization and peer-to-peer (P2P) trading model inside an energy community, featuring [...]

Generated from: output/c542c361de4a1914_EP/c542c361de4a1914_EP.pdf
BM name       : P2P trading

Usage
-----
    python simulate_p2p_trading.py \
        --dataset  path/to/dataset.csv \
        --tariffs  path/to/tariffs.csv \
        [--param   name=value ...]

Or import and call directly:

    import pandas as pd
    from simulate_p2p_trading import simulate

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
    pd = globals().get('pd') or globals().get('pandas') or __import__('pandas')
    np = globals().get('np') or globals().get('numpy') or __import__('numpy')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # REQUIRED HELPER FUNCTIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def safe_numeric(value, default=None):
        if isinstance(value, pd.DataFrame):
            return value.apply(lambda col: pd.to_numeric(col, errors="coerce")).fillna(default) if default is not None else value.apply(lambda col: pd.to_numeric(col, errors="coerce"))
        elif isinstance(value, pd.Series):
            res = pd.to_numeric(value, errors="coerce")
            return res.fillna(default) if default is not None else res
        elif isinstance(value, (list, np.ndarray)):
            s = pd.Series(value)
            res = pd.to_numeric(s, errors="coerce")
            if default is not None:
                res = res.fillna(default)
            return res.values
        else:
            try:
                val = pd.to_numeric(value, errors="coerce")
                if pd.isna(val) and default is not None:
                    return default
                return val
            except:
                return default if default is not None else np.nan

    def safe_datetime(value):
        return pd.to_datetime(value, errors="coerce")

    def safe_divide(a, b, default=0.0):
        a_num = safe_numeric(a, default=0.0)
        b_num = safe_numeric(b, default=0.0)
        if isinstance(a_num, pd.Series) or isinstance(b_num, pd.Series):
            if not isinstance(a_num, pd.Series):
                a_num = pd.Series(a_num, index=b_num.index)
            if not isinstance(b_num, pd.Series):
                b_num = pd.Series(b_num, index=a_num.index)
            cond = (b_num.abs() <= 1e-9) | pd.isna(b_num) | (~np.isfinite(b_num))
            res = np.where(cond, default, a_num / b_num)
            s = pd.Series(res, index=a_num.index)
            return s.replace([float("inf"), float("-inf")], default).fillna(default)
        elif isinstance(a_num, np.ndarray) or isinstance(b_num, np.ndarray):
            cond = (np.abs(b_num) <= 1e-9) | pd.isna(b_num) | (~np.isfinite(b_num))
            res = np.where(cond, default, a_num / b_num)
            s = pd.Series(res)
            s = s.replace([float("inf"), float("-inf")], default).fillna(default)
            return s.values
        else:
            if abs(b_num) <= 1e-9 or pd.isna(b_num) or not np.isfinite(b_num):
                return default
            res = a_num / b_num
            if not np.isfinite(res):
                return default
            return res

    def to_jsonable(value):
        if isinstance(value, pd.DataFrame):
            return value.where(pd.notnull(value), None).to_dict(orient="records")
        elif isinstance(value, pd.Series):
            val_cleaned = value.where(pd.notnull(value), None)
            if isinstance(val_cleaned.index, pd.MultiIndex):
                return {f"{k[0]}--{k[1]}": v for k, v in val_cleaned.to_dict().items()}
            elif isinstance(val_cleaned.index, pd.DatetimeIndex):
                return {k.isoformat(): v for k, v in val_cleaned.to_dict().items()}
            else:
                return {str(k): v for k, v in val_cleaned.to_dict().items()}
        elif isinstance(value, np.ndarray):
            return np.where(pd.isna(value), None, value).tolist()
        elif isinstance(value, (np.integer, np.floating)):
            return None if pd.isna(value) else value.item()
        elif isinstance(value, float) and (pd.isna(value) or not np.isfinite(value)):
            return None
        elif isinstance(value, dict):
            return {str(k): to_jsonable(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [to_jsonable(v) for v in value]
        return value

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DATA VALIDATION STAGE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    parameters = parameters or {}
    
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
        raise ValueError("No valid rows remaining after Timestamp validation")
        
    df["member_id"] = df["member_id"].astype(str).str.strip()
    if (df["member_id"] == "").any() or df["member_id"].isna().any():
        raise ValueError("member_id contains missing or blank values")
        
    df = df.sort_values(by="Timestamp").reset_index(drop=True)
    
    if df.duplicated(subset=["member_id", "Timestamp"]).any():
        raise ValueError("Duplicate rows detected for member_id and Timestamp")
        
    # Tariffs validation
    if tariffs is None or not hasattr(tariffs, "columns"):
        raise ValueError("tariffs must be a pandas DataFrame")
    if "time" not in tariffs.columns or "ToU" not in tariffs.columns or "FiT" not in tariffs.columns:
        raise ValueError("tariffs must contain 'time', 'ToU', and 'FiT' columns")
        
    tariffs = tariffs.copy()
    tariffs["ToU"] = safe_numeric(tariffs["ToU"])
    tariffs["FiT"] = safe_numeric(tariffs["FiT"])
    if tariffs["ToU"].isna().any() or tariffs["FiT"].isna().any():
        raise ValueError("tariffs contain non-numeric values")
        
    required_hours = set(range(24))
    tariff_hours = set(tariffs["time"].astype(int).tolist())
    if not required_hours.issubset(tariff_hours):
        raise ValueError("tariffs are missing required hours (0-23)")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PARAMETER INITIALIZATION & MAPPING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _ToU_param = parameters.get("ToU", None)
    _FIT_param = parameters.get("FIT", None)
    M = safe_numeric(parameters.get("M", len(df["member_id"].unique())), default=122)
    beta = safe_numeric(parameters.get("beta", 0.05), default=0.05)
    G_sh = parameters.get("G_sh", 0.0)
    rho_m = parameters.get("rho_m", 0.09)
    rho_EC = safe_numeric(parameters.get("rho_EC", 0.1), default=0.1)
    G_EC = parameters.get("G_EC", 0.0)
    C_EC = parameters.get("C_EC", 0.0)
    F_m = parameters.get("F_m", 0.0)
    FP_m = parameters.get("FP_m", 0.2)
    p_f_m = parameters.get("p_f_m", 0.8)
    DP_init_m = parameters.get("DP_init_m", 0.6)
    SP_init_m = parameters.get("SP_init_m", 0.4)

    # Map hour-of-day tariffs
    tou_map = tariffs.set_index("time")["ToU"]
    fit_map = tariffs.set_index("time")["FiT"]
    df["hour"] = df["Timestamp"].dt.hour
    df["ToU"] = df["hour"].map(tou_map)
    df["FIT"] = df["hour"].map(fit_map)

    # Robust mapping function for member-specific and hourly parameters
    def map_param_to_df(param, df, key_col, default):
        if isinstance(param, dict):
            return df[key_col].map(param).fillna(default)
        elif isinstance(param, pd.Series):
            return df[key_col].map(param).fillna(default)
        elif isinstance(param, (list, np.ndarray)):
            if len(param) == len(df):
                return safe_numeric(param, default=default)
            elif len(param) == 24 and key_col == "hour":
                mapping = {h: val for h, val in enumerate(param)}
                return df["hour"].map(mapping).fillna(default)
        return safe_numeric(param, default=default)

    # Map member parameters
    rho_m_val = map_param_to_df(rho_m, df, "member_id", 0.09)
    F_m_val = map_param_to_df(F_m, df, "member_id", 0.0)
    FP_m_val = map_param_to_df(FP_m, df, "member_id", 0.2)
    p_f_m_val = map_param_to_df(p_f_m, df, "member_id", 0.8)
    DP_init_m_val = map_param_to_df(DP_init_m, df, "member_id", 0.6)
    SP_init_m_val = map_param_to_df(SP_init_m, df, "member_id", 0.4)

    # Map hourly parameters
    G_sh_val = map_param_to_df(G_sh, df, "hour", 0.0)
    G_EC_val = map_param_to_df(G_EC, df, "hour", 0.0)
    C_EC_val = map_param_to_df(C_EC, df, "hour", 0.0)

    # Assign temp columns for equations
    df["_F_m_val"] = F_m_val
    df["_FP_m_val"] = FP_m_val
    df["_p_f_m_val"] = p_f_m_val
    df["_C_EC_val"] = C_EC_val
    df["_G_EC_val"] = G_EC_val
    df["_G_sh_val"] = G_sh_val

    # Define aggregate community variables at the Timestamp level
    G_sh_t = df.groupby("Timestamp")["_G_sh_val"].first()
    G_EC_t = df.groupby("Timestamp")["_G_EC_val"].first()
    C_EC_t = df.groupby("Timestamp")["_C_EC_val"].first()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # EQUATION EXECUTION WITH DEPENDENCY GUARDS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # eq_D_m
    df["D_m"] = np.maximum(0.0, df["load"] - df["generation"] - rho_m_val * G_sh_val)

    # eq_S_m
    df["S_m"] = np.maximum(0.0, df["generation" ] + rho_m_val * G_sh_val - df["load"])

    # eq_D_EC
    D_EC_series = df.groupby("Timestamp")["D_m"].sum() + C_EC_t
    df["D_EC"] = df["Timestamp"].map(D_EC_series)

    # eq_S_EC
    S_EC_series = df.groupby("Timestamp")["S_m"].sum() + G_EC_t + rho_EC * G_sh_t
    df["S_EC"] = df["Timestamp"].map(S_EC_series)

    # eq_Q_b
    df["Q_b"] = np.where(df["S_EC"] < df["D_EC"], df["D_m"] - df["_F_m_val"], df["D_m" ] + df["_F_m_val"])

    # eq_Q_s
    df["Q_s"] = np.where(df["S_EC"] < df["D_EC"], df["S_m"] + df["_F_m_val"], df["S_m"] - df["_F_m_val"])

    # eq_DP_m
    dp_deficit_div = safe_divide(df["D_m"] * df["ToU"] + df["_p_f_m_val"] * df["_F_m_val"] * df["_FP_m_val"], df["D_m"] - df["_F_m_val"], default=0.0)
    dp_deficit = np.where(df["D_m"] - df["_F_m_val"] > 0.0, dp_deficit_div, df["ToU"])
    dp_surplus_div = safe_divide(df["D_m"] * df["ToU"], df["D_m"] + df["_F_m_val"], default=0.0)
    dp_surplus = np.where(df["D_m"] + df["_F_m_val"] > 0.0, dp_surplus_div, df["ToU"])
    df["DP_m"] = np.where(df["S_EC"] < df["D_EC"], dp_deficit, dp_surplus)

    # eq_SP_m
    sp_deficit_div = safe_divide(df["S_m"] * df["FIT"] - df["_p_f_m_val"] * df["_F_m_val"] * df["_FP_m_val"], df["S_m"] + df["_F_m_val"], default=0.0)
    sp_deficit = np.where(df["S_m"] + df["_F_m_val"] > 0.0, sp_deficit_div, df["FIT"])
    sp_surplus_div = safe_divide(df["S_m"] * df["FIT"], df["S_m"] - df["_F_m_val"], default=0.0)
    sp_surplus = np.where(df["S_m"] - df["_F_m_val"] > 0.0, sp_surplus_div, df["FIT"])
    df["SP_m"] = np.where(df["S_EC"] < df["D_EC"], sp_deficit, sp_surplus)

    # eq_QD
    QD_series = df.groupby("Timestamp")["Q_b"].sum()
    df["QD"] = df["Timestamp"].map(QD_series)

    # eq_QS
    QS_series = df.groupby("Timestamp")["Q_s"].sum()
    df["QS"] = df["Timestamp"].map(QS_series)

    # eq_Q_LEM_b
    ratio_b = safe_divide(df["QS"], df["QD"], default=0.0)
    val_b = df["Q_b"] * ratio_b
    df["Q_LEM_b"] = np.where(df["QD"] > 0.0, np.minimum(df["Q_b"], val_b), 0.0)

    # eq_Q_LEM_s
    ratio_s = safe_divide(df["QD"], df["QS"], default=0.0)
    val_s = df["Q_s"] * ratio_s
    df["Q_LEM_s"] = np.where(df["QS"] > 0.0, np.minimum(df["Q_s"], val_s), 0.0)

    # eq_P_clearing
    p_clearing_surplus = df["FIT"] + beta * (df["ToU"] - df["FIT"])
    p_clearing_deficit = df["ToU"] - beta * (df["ToU"] - df["FIT"])
    df["P_clearing"] = np.where(df["QS"] >= df["QD"], p_clearing_surplus, p_clearing_deficit)

    # eq_Cost_m
    df["Cost_m"] = df["Q_LEM_b"] * df["P_clearing"] + (df["Q_b"] - df["Q_LEM_b"]) * df["ToU"] + df["_F_m_val"] * df["_FP_m_val"]

    # eq_Rev_m
    df["Rev_m"] = df["Q_LEM_s"] * df["P_clearing"] + (df["Q_s"] - df["Q_LEM_s"]) * df["FIT"] - df["_F_m_val"] * df["_FP_m_val"]

    # eq_Nett_minus_m
    df["Nett_minus_m"] = np.maximum(0.0, df["load"] - df["generation"] - rho_m_val * G_sh_val)

    # eq_Nett_plus_m
    df["Nett_plus_m"] = np.maximum(0.0, df["generation"] + rho_m_val * G_sh_val - df["load"])

    # eq_delta_Q_m
    df["delta_Q_m"] = (df["Nett_minus_m"] - df["Q_b"]) - (df["Nett_plus_m"] - df["Q_s"])

    # eq_Cost_settled_m
    df["Cost_settled_m"] = df["Cost_m"] + np.where(df["delta_Q_m"] > 0.0, df["delta_Q_m"] * df["ToU"], 0.0)

    # eq_Rev_settled_m
    df["Rev_settled_m"] = df["Rev_m"] + np.where(df["delta_Q_m"] < 0.0, df["delta_Q_m"].abs() * df["FIT"], 0.0)

    # eq_Nett_minus_EC
    diff_sum = (df["Nett_minus_m"] - df["Nett_plus_m"]).groupby(df["Timestamp"]).sum()
    nett_minus_ec_series = np.maximum(0.0, diff_sum + C_EC_t - G_EC_t - G_sh_t)
    df["Nett_minus_EC"] = df["Timestamp"].map(nett_minus_ec_series)

    # eq_Nett_plus_EC
    nett_plus_ec_series = np.maximum(0.0, G_EC_t + G_sh_t - C_EC_t - diff_sum)
    df["Nett_plus_EC"] = df["Timestamp"].map(nett_plus_ec_series)

    # eq_VS_EC_t
    cost_rev_diff_sum = (df["Cost_settled_m"] - df["Rev_settled_m"]).groupby(df["Timestamp"]).sum()
    tou_t = df.groupby("Timestamp")["ToU"].first()
    fit_t = df.groupby("Timestamp")["FIT"].first()
    vs_ec_t_series = cost_rev_diff_sum - (nett_minus_ec_series * tou_t - nett_plus_ec_series * fit_t)
    df["VS_EC_t"] = df["Timestamp"].map(vs_ec_t_series)

    # eq_EQ_m
    df["EQ_m"] = safe_divide(df["VS_EC_t"], M)

    # eq_delta_Q_minus_sum
    dq_pos_sum_series = df["delta_Q_m"].clip(lower=0.0).groupby(df["Timestamp"]).sum()
    df["delta_Q_minus_sum"] = df["Timestamp"].map(dq_pos_sum_series)

    # eq_delta_Q_plus_sum
    dq_neg_sum_series = df["delta_Q_m"].clip(upper=0.0).abs().groupby(df["Timestamp"]).sum()
    df["delta_Q_plus_sum"] = df["Timestamp"].map(dq_neg_sum_series)

    # eq_q_minus_m
    div_minus = safe_divide(df["delta_Q_m"], df["delta_Q_minus_sum"], default=0.0)
    df["q_minus_m"] = np.where((df["delta_Q_m"] > 0.0) & (df["delta_Q_minus_sum"] > 0.0), div_minus, 0.0)

    # eq_q_plus_m
    div_plus = safe_divide(df["delta_Q_m"].abs(), df["delta_Q_plus_sum"], default=0.0)
    df["q_plus_m"] = np.where((df["delta_Q_m"] < 0.0) & (df["delta_Q_plus_sum"] > 0.0), div_plus, 0.0)

    # eq_VS_m
    cond_neg = df["delta_Q_m"] < 0.0
    cond_pos = df["delta_Q_m"] > 0.0
    val_neg = (1.0 + df["q_plus_m"]) * df["EQ_m"]
    val_pos = (1.0 - df["q_minus_m"]) * df["EQ_m"]
    df["VS_m"] = np.where(cond_neg, val_neg, np.where(cond_pos, val_pos, df["EQ_m"]))

    # eq_Payment_init
    df["Payment_init"] = df["Cost_settled_m"] - df["Rev_settled_m"]

    # eq_Payment_final
    df["Payment_final"] = df["Payment_init"] - df["VS_m"]

    # eq_Cost_final
    df["Cost_final"] = np.maximum(0.0, df["Payment_final"])

    # eq_Rev_final
    df["Rev_final"] = np.maximum(0.0, -df["Payment_final"])

    # eq_Payment_no_EC
    df["Payment_no_EC"] = df["Nett_minus_m"] * df["ToU"] - df["Nett_plus_m"] * df["FIT"]

    # eq_Gain_m
    df["Gain_m"] = df["Payment_no_EC"] - df["Payment_final"]

    # eq_SSI_m
    min_val_ssi = np.minimum(df["load"], df["generation"] + G_sh_val * rho_m_val)
    div_ssi = safe_divide(min_val_ssi, df["load"], default=1.0)
    df["SSI_m"] = np.where(df["load"] > 0.0, div_ssi, 1.0)

    # eq_SCI_m
    gen_sh_sum = df["generation"] + G_sh_val * rho_m_val
    div_sci = safe_divide(np.minimum(df["load"], gen_sh_sum), gen_sh_sum, default=1.0)
    df["SCI_m"] = np.where(gen_sh_sum > 0.0, div_sci, 1.0)

    # eq_QI_m
    div_qi_b = safe_divide(df["Q_b"], df["Nett_minus_m"], default=0.0)
    div_qi_s = safe_divide(df["Q_s"], df["Nett_plus_m"], default=0.0)
    df["QI_m"] = np.where(df["Nett_minus_m"] > 0.0, div_qi_b, np.where(df["Nett_plus_m" ] > 0.0, div_qi_s, 0.0))

    # eq_SSI_EC
    load_sum = df.groupby("Timestamp")["load"].sum()
    gen_sum = df.groupby("Timestamp")["generation"].sum()
    denom_ssi_ec = load_sum + C_EC_t
    num_ssi_ec = np.minimum(load_sum + C_EC_t, G_EC_t + G_sh_t + gen_sum)
    div_ssi_ec = safe_divide(num_ssi_ec, denom_ssi_ec, default=1.0)
    ssi_ec_series = pd.Series(np.where(denom_ssi_ec > 0.0, div_ssi_ec, 1.0), index=load_sum.index)
    df["SSI_EC"] = df["Timestamp"].map(ssi_ec_series)

    # eq_SCI_EC
    denom_sci_ec = G_EC_t + G_sh_t + gen_sum
    num_sci_ec = np.minimum(load_sum + C_EC_t, denom_sci_ec)
    div_sci_ec = safe_divide(num_sci_ec, denom_sci_ec, default=1.0)
    sci_ec_series = pd.Series(np.where(denom_sci_ec > 0.0, div_sci_ec, 1.0), index=load_sum.index)
    df["SCI_EC"] = df["Timestamp"].map(sci_ec_series)

    # eq_DLS_EC
    df["_dls_min"] = np.minimum(df["load"], df["generation"] + G_sh_val * rho_m_val)
    dls_min_sum = df.groupby("Timestamp")["_dls_min"].sum()
    qb_sum = df.groupby("Timestamp")["Q_b"].sum()
    num_dls = dls_min_sum + qb_sum
    denom_dls = load_sum
    div_dls = safe_divide(num_dls, denom_dls, default=1.0)
    dls_ec_series = pd.Series(np.where(denom_dls > 0.0, div_dls, 1.0), index=load_sum.index)
    df["DLS_EC"] = df["Timestamp"].map(dls_ec_series)

    # eq_x_m
    div_x = safe_divide(df["VS_m"], df["generation"] + G_sh_val * rho_m_val, default=0.0)
    df["x_m"] = np.where((df["generation"] + G_sh_val * rho_m_val) > 0.0, div_x, 0.0)

    # eq_FI_EC
    df["_x_m_sq"] = df["x_m"] ** 2
    sum_x = df.groupby("Timestamp")["x_m"].sum()
    sum_x_sq = df.groupby("Timestamp")["_x_m_sq"].sum()
    denom_fi = M * sum_x_sq
    num_fi = sum_x ** 2
    div_fi = safe_divide(num_fi, denom_fi, default=1.0)
    fi_ec_series = pd.Series(np.where(sum_x_sq > 0.0, div_fi, 1.0), index=sum_x_sq.index)
    df["FI_EC"] = df["Timestamp"].map(fi_ec_series)

    # eq_Cost_0_m
    df["Cost_0_m"] = np.where(df["D_m"] > 0.0, df["D_m"] * DP_init_m_val, 0.0)

    # eq_Rev_0_m
    df["Rev_0_m"] = np.where(df["S_m"] > 0.0, df["S_m"] * SP_init_m_val, 0.0)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RESULTS MATRIX & VECTOR CONVERSION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    unique_ts = pd.Series(df["Timestamp"].unique()).sort_values().reset_index(drop=True)

    def make_matrix_series(df, col):
        ts_str = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        idx = pd.MultiIndex.from_arrays([ts_str, df["member_id"]], names=["Timestamp", "member_id"])
        return pd.Series(df[col].values, index=idx)

    def make_vector_series(unique_ts, values):
        ts_str = unique_ts.dt.strftime("%Y-%m-%d %H:%M:%S")
        return pd.Series(values, index=ts_str)

    D_m_res = make_matrix_series(df, "D_m")
    S_m_res = make_matrix_series(df, "S_m")
    D_EC_res = make_vector_series(unique_ts, D_EC_series.loc[unique_ts].values)
    S_EC_res = make_vector_series(unique_ts, S_EC_series.loc[unique_ts].values)
    Q_b_res = make_matrix_series(df, "Q_b")
    Q_s_res = make_matrix_series(df, "Q_s")
    DP_m_res = make_matrix_series(df, "DP_m")
    SP_m_res = make_matrix_series(df, "SP_m")
    QD_res = make_vector_series(unique_ts, QD_series.loc[unique_ts].values)
    QS_res = make_vector_series(unique_ts, QS_series.loc[unique_ts].values)
    Q_LEM_b_res = make_matrix_series(df, "Q_LEM_b")
    Q_LEM_s_res = make_matrix_series(df, "Q_LEM_s")
    P_clearing_res = make_vector_series(unique_ts, df.groupby("Timestamp")["P_clearing"].first().loc[unique_ts].values)
    Cost_m_res = make_matrix_series(df, "Cost_m")
    Rev_m_res = make_matrix_series(df, "Rev_m")
    Nett_minus_m_res = make_matrix_series(df, "Nett_minus_m")
    Nett_plus_m_res = make_matrix_series(df, "Nett_plus_m")
    delta_Q_m_res = make_matrix_series(df, "delta_Q_m")
    Cost_settled_m_res = make_matrix_series(df, "Cost_settled_m")
    Rev_settled_m_res = make_matrix_series(df, "Rev_settled_m")
    Nett_minus_EC_res = make_vector_series(unique_ts, nett_minus_ec_series.loc[unique_ts].values)
    Nett_plus_EC_res = make_vector_series(unique_ts, nett_plus_ec_series.loc[unique_ts].values)
    VS_EC_t_res = make_vector_series(unique_ts, vs_ec_t_series.loc[unique_ts].values)
    EQ_m_res = make_matrix_series(df, "EQ_m")
    delta_Q_minus_sum_res = make_vector_series(unique_ts, dq_pos_sum_series.loc[unique_ts].values)
    delta_Q_plus_sum_res = make_vector_series(unique_ts, dq_neg_sum_series.loc[unique_ts].values)
    q_minus_m_res = make_matrix_series(df, "q_minus_m")
    q_plus_m_res = make_matrix_series(df, "q_plus_m")
    VS_m_res = make_matrix_series(df, "VS_m")
    Payment_init_res = make_matrix_series(df, "Payment_init")
    Payment_final_res = make_matrix_series(df, "Payment_final")
    Cost_final_res = make_matrix_series(df, "Cost_final")
    Rev_final_res = make_matrix_series(df, "Rev_final")
    Payment_no_EC_res = make_matrix_series(df, "Payment_no_EC")
    Gain_m_res = make_matrix_series(df, "Gain_m")
    SSI_m_res = make_matrix_series(df, "SSI_m")
    SCI_m_res = make_matrix_series(df, "SCI_m")
    QI_m_res = make_matrix_series(df, "QI_m")
    SSI_EC_res = make_vector_series(unique_ts, ssi_ec_series.loc[unique_ts].values)
    SCI_EC_res = make_vector_series(unique_ts, sci_ec_series.loc[unique_ts].values)
    DLS_EC_res = make_vector_series(unique_ts, dls_ec_series.loc[unique_ts].values)
    x_m_res = make_matrix_series(df, "x_m")
    FI_EC_res = make_vector_series(unique_ts, fi_ec_series.loc[unique_ts].values)
    Cost_0_m_res = make_matrix_series(df, "Cost_0_m")
    Rev_0_m_res = make_matrix_series(df, "Rev_0_m")

    results = {
        "D_m": D_m_res,
        "S_m": S_m_res,
        "D_EC": D_EC_res,
        "S_EC": S_EC_res,
        "Q_b": Q_b_res,
        "Q_s": Q_s_res,
        "DP_m": DP_m_res,
        "SP_m": SP_m_res,
        "QD": QD_res,
        "QS": QS_res,
        "Q_LEM_b": Q_LEM_b_res,
        "Q_LEM_s": Q_LEM_s_res,
        "P_clearing": P_clearing_res,
        "Cost_m": Cost_m_res,
        "Rev_m": Rev_m_res,
        "Nett_minus_m": Nett_minus_m_res,
        "Nett_plus_m": Nett_plus_m_res,
        "delta_Q_m": delta_Q_m_res,
        "Cost_settled_m": Cost_settled_m_res,
        "Rev_settled_m": Rev_settled_m_res,
        "Nett_minus_EC": Nett_minus_EC_res,
        "Nett_plus_EC": Nett_plus_EC_res,
        "VS_EC_t": VS_EC_t_res,
        "EQ_m": EQ_m_res,
        "delta_Q_minus_sum": delta_Q_minus_sum_res,
        "delta_Q_plus_sum": delta_Q_plus_sum_res,
        "q_minus_m": q_minus_m_res,
        "q_plus_m": q_plus_m_res,
        "VS_m": VS_m_res,
        "Payment_init": Payment_init_res,
        "Payment_final": Payment_final_res,
        "Cost_final": Cost_final_res,
        "Rev_final": Rev_final_res,
        "Payment_no_EC": Payment_no_EC_res,
        "Gain_m": Gain_m_res,
        "SSI_m": SSI_m_res,
        "SCI_m": SCI_m_res,
        "QI_m": QI_m_res,
        "SSI_EC": SSI_EC_res,
        "SCI_EC": SCI_EC_res,
        "DLS_EC": DLS_EC_res,
        "x_m": x_m_res,
        "FI_EC": FI_EC_res,
        "Cost_0_m": Cost_0_m_res,
        "Rev_0_m": Rev_0_m_res,
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
