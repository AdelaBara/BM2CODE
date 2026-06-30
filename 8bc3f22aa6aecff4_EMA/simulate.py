"""simulate_value_share.py
An agent-based energy community model integrating household electricity consumption, renewable energy sources, [...]

Generated from: output/8bc3f22aa6aecff4_EMA/8bc3f22aa6aecff4_EMA.pdf
BM name       : Value share

Usage
-----
    python simulate_value_share.py \
        --dataset  path/to/dataset.csv \
        --tariffs  path/to/tariffs.csv \
        [--param   name=value ...]

Or import and call directly:

    import pandas as pd
    from simulate_value_share import simulate

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

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # REQUIRED HELPER FUNCTIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def safe_numeric(value, default=None):
        if isinstance(value, pd.DataFrame):
            return value.apply(lambda x: pd.to_numeric(x, errors="coerce")).fillna(default if default is not None else np.nan)
        elif isinstance(value, pd.Series):
            res = pd.to_numeric(value, errors="coerce")
            if default is not None:
                res = res.fillna(default)
            return res
        else:
            try:
                res = pd.to_numeric(value, errors="coerce")
                if pd.isna(res):
                    return default if default is not None else np.nan
                return float(res)
            except Exception:
                return default if default is not None else np.nan

    def safe_datetime(value):
        return pd.to_datetime(value, errors="coerce")

    def safe_divide(a, b, default=0.0):
        a_num = safe_numeric(a, default=0.0)
        b_num = safe_numeric(b, default=0.0)
        
        if isinstance(a_num, (pd.Series, pd.DataFrame)) or isinstance(b_num, (pd.Series, pd.DataFrame)):
            if isinstance(b_num, (pd.Series, pd.DataFrame)):
                mask = b_num.abs() <= 1e-9
                res = a_num / b_num
                res = res.mask(mask, default)
                res = res.fillna(default)
                if isinstance(res, pd.DataFrame):
                    res = res.apply(lambda col: col.map(lambda val: val if (pd.notna(val) and np.isfinite(val)) else default))
                else:
                    res = res.map(lambda val: val if (pd.notna(val) and np.isfinite(val)) else default)
                return res
            else:
                if abs(b_num) <= 1e-9 or not np.isfinite(b_num):
                    if isinstance(a_num, pd.DataFrame):
                        return pd.DataFrame(default, index=a_num.index, columns=a_num.columns)
                    else:
                        return pd.Series(default, index=a_num.index)
                res = a_num / b_num
                if isinstance(res, pd.DataFrame):
                    res = res.apply(lambda col: col.map(lambda val: val if (pd.notna(val) and np.isfinite(val)) else default))
                else:
                    res = res.map(lambda val: val if (pd.notna(val) and np.isfinite(val)) else default)
                return res
        else:
            if abs(b_num) <= 1e-9 or not np.isfinite(b_num):
                return default
            res = a_num / b_num
            return res if np.isfinite(res) else default

    def safe_get_parameter(parameters, name, default):
        val = parameters.get(name, default)
        return safe_numeric(val, default=default)

    def to_jsonable(value):
        if isinstance(value, pd.DataFrame):
            if isinstance(value.index, pd.DatetimeIndex):
                value = value.copy()
                value.index = value.index.astype(str)
            value = value.where(pd.notnull(value), None)
            return value.to_dict(orient="dict")
        elif isinstance(value, pd.Series):
            if isinstance(value.index, pd.DatetimeIndex):
                value = value.copy()
                value.index = value.index.astype(str)
            value = value.where(pd.notnull(value), None)
            return value.to_dict()
        elif isinstance(value, np.ndarray):
            return [None if pd.isna(x) else x for x in value.tolist()]
        elif isinstance(value, (list, tuple)):
            return [to_jsonable(v) for v in value]
        elif isinstance(value, dict):
            return {str(k): to_jsonable(v) for k, v in value.items()}
        else:
            if pd.isna(value):
                return None
            if hasattr(value, "item"):
                return value.item()
            return value

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DATA VALIDATION STAGE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    parameters = parameters or {}

    if df is None or not hasattr(df, "columns"):
        raise ValueError("df must be a pandas DataFrame")

    required_cols = ["Timestamp", "member_id", "load", "generation"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()
    df["Timestamp"] = safe_datetime(df["Timestamp"])
    df["load"] = safe_numeric(df["load"], default=0.0)
    df["generation"] = safe_numeric(df["generation"], default=0.0)
    if "temperature" in df.columns:
        df["temperature"] = safe_numeric(df["temperature"], default=15.0)

    df = df.dropna(subset=["Timestamp"])
    if df.empty:
        raise ValueError("No valid rows remain after dropping invalid Timestamps")

    df["member_id"] = df["member_id"].astype(str).str.strip()
    if df["member_id"].eq("").any() or df["member_id"].isna().any():
        raise ValueError("Missing or blank member_id found")

    df = df.sort_values(by="Timestamp").reset_index(drop=True)

    if df.duplicated(subset=["member_id", "Timestamp"]).any():
        raise ValueError("Duplicate rows found for member_id and Timestamp")

    if tariffs is None or not hasattr(tariffs, "columns"):
        raise ValueError("tariffs must be a pandas DataFrame")
    if "time" not in tariffs.columns:
        raise ValueError("tariffs must contain a 'time' column")

    tariffs = tariffs.copy()
    for col in tariffs.columns:
        tariffs[col] = safe_numeric(tariffs[col])

    required_tariff_keys = ["ToU", "FiT"]
    for tk in required_tariff_keys:
        if tk not in tariffs.columns:
            raise ValueError(f"Missing required tariff key in tariffs: {tk}")

    tariffs_indexed = tariffs.set_index("time")
    for h in range(24):
        if h not in tariffs_indexed.index or pd.isna(tariffs_indexed.loc[h, "ToU"]) or pd.isna(tariffs_indexed.loc[h, "FiT"]):
            raise ValueError(f"Missing tariff value for hour {h}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PARAMETER INITIALIZATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Gas_Price = safe_get_parameter(parameters, "Gas_Price", 0.31)
    Petrol_Price = safe_get_parameter(parameters, "Petrol_Price", 7.0)
    UC_PV = safe_get_parameter(parameters, "UC_PV", 3500.0)
    UC_HP = safe_get_parameter(parameters, "UC_HP", 1710000.0)
    UC_EVCS = safe_get_parameter(parameters, "UC_EVCS", 100000.0)
    Y_spec = safe_get_parameter(parameters, "Y_spec", 1200.0)
    m = safe_get_parameter(parameters, "m", 114)
    V_i = safe_get_parameter(parameters, "V_i", 175.0)
    k_ins = safe_get_parameter(parameters, "k_ins", 0.2)
    T_in = safe_get_parameter(parameters, "T_in", 24.0)
    COP = safe_get_parameter(parameters, "COP", 3.2)
    fuel_cons = safe_get_parameter(parameters, "fuel_cons", 8.0)
    km_365 = safe_get_parameter(parameters, "km_365", 10000.0)
    EC_EV = safe_get_parameter(parameters, "EC_EV", 15.0)
    eta_EV = safe_get_parameter(parameters, "eta_EV", 0.9)
    EF_el = safe_get_parameter(parameters, "EF_el", 0.25)
    EF_H = safe_get_parameter(parameters, "EF_H", 0.181)
    EF_T = safe_get_parameter(parameters, "EF_T", 2.31)
    I_HP = safe_get_parameter(parameters, "I_HP", 1)
    I_EV = safe_get_parameter(parameters, "I_EV", 1)
    I_LEM = safe_get_parameter(parameters, "I_LEM", 0)
    I_ind_cov = safe_get_parameter(parameters, "I_ind_cov", 1)

    # Pivot load and generation to wide format
    CP_i_t = df.pivot(index="Timestamp", columns="member_id", values="load").fillna(0.0)
    G_i_t = df.pivot(index="Timestamp", columns="member_id", values="generation").fillna(0.0)

    # Outdoor Temperature Series Alignment
    if "temperature" in df.columns:
        T_out_t = df.groupby("Timestamp")["temperature"].first().reindex(CP_i_t.index).fillna(15.0)
    elif "T_out_t" in df.columns:
        T_out_t = df.groupby("Timestamp")["T_out_t"].first().reindex(CP_i_t.index).fillna(15.0)
    else:
        T_out_t_val = parameters.get("T_out_t", None)
        if T_out_t_val is not None:
            if isinstance(T_out_t_val, (list, pd.Series, np.ndarray)):
                T_out_t = pd.Series(T_out_t_val, index=CP_i_t.index).fillna(15.0)
            else:
                T_out_t = pd.Series(float(T_out_t_val), index=CP_i_t.index)
        else:
            T_out_t = pd.Series(15.0, index=CP_i_t.index)

    # Tariff Series Alignment
    hours = CP_i_t.index.hour
    ToU_t = hours.map(tariffs_indexed["ToU"]).to_series()
    ToU_t.index = CP_i_t.index
    FiT_t = hours.map(tariffs_indexed["FiT"]).to_series()
    FiT_t.index = CP_i_t.index

    if "P_LEM" in tariffs_indexed.columns:
        P_LEM_t = hours.map(tariffs_indexed["P_LEM"]).to_series()
    else:
        P_LEM_t_val = parameters.get("P_LEM_t", None)
        if P_LEM_t_val is not None:
            P_LEM_t = pd.Series(P_LEM_t_val, index=CP_i_t.index)
        else:
            P_LEM_t = (ToU_t + FiT_t) * 0.5
    P_LEM_t.index = CP_i_t.index

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MODEL EQUATIONS IMPLEMENTATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Derived Variable C_EC_t
    C_EC_t = 0.03 * CP_i_t.sum(axis=1)

    # Eq6 | P_PV = sum_t(C_EC_t) / Y_spec
    P_PV = safe_divide(C_EC_t.sum(), Y_spec)

    # Derived Variable G_EC_t
    g_mean = G_i_t.mean(axis=1)
    g_mean_max = g_mean.max()
    G_EC_t = P_PV * safe_divide(g_mean, g_mean_max)

    # Eq8 | C_sh_t = C_EC_t / m
    C_sh_t = safe_divide(C_EC_t, m)

    # Eq9 | G_sh_t = G_EC_t / m
    G_sh_t = safe_divide(G_EC_t, m)

    # Eq19 | Q_H_i_t = V_i * max(0, T_in - T_out_t) * k_ins
    Q_H_i_t = pd.DataFrame({col: V_i * np.maximum(0.0, T_in - T_out_t) * k_ins for col in CP_i_t.columns}, index=CP_i_t.index)

    # Eq19_CH_i_t | CH_i_t = I_HP * (Q_H_i_t / COP)
    CH_i_t = I_HP * safe_divide(Q_H_i_t, COP)

    # Eq20_CEV_i_t | CEV_i_t = I_EV * ((EC_EV / eta_EV) * (km_365 / 8760) / 100)
    CEV_i_t = pd.DataFrame(I_EV * ((safe_divide(EC_EV, eta_EV) * safe_divide(km_365, 8760.0)) / 100.0), index=CP_i_t.index, columns=CP_i_t.columns)

    # Eq12 | C_i_t = CP_i_t + CH_i_t + CEV_i_t
    C_i_t = CP_i_t + CH_i_t + CEV_i_t

    # Eq10 | D_i_t = max(0, C_i_t + C_sh_t - (G_i_t + G_sh_t))
    D_i_t = (C_i_t.add(C_sh_t, axis=0) - G_i_t.add(G_sh_t, axis=0)).clip(lower=0.0)

    # Eq11 | S_i_t = max(0, G_i_t + G_sh_t - (C_i_t + C_sh_t))
    S_i_t = (G_i_t.add(G_sh_t, axis=0) - C_i_t.add(C_sh_t, axis=0)).clip(lower=0.0)

    # Eq14 | D_EC_t = max(0, sum_i(D_i_t) - sum_i(S_i_t))
    D_EC_t = (D_i_t.sum(axis=1) - S_i_t.sum(axis=1)).clip(lower=0.0)

    # Eq15 | S_EC_t = max(0, sum_i(S_i_t) - sum_i(D_i_t))
    S_EC_t = (S_i_t.sum(axis=1) - D_i_t.sum(axis=1)).clip(lower=0.0)

    # Derived Variables QB_i_t, QS_i_t (LEM Trading)
    if I_LEM == 1:
        d_sum = D_i_t.sum(axis=1)
        s_sum = S_i_t.sum(axis=1)
        ratio_buy = safe_divide(s_sum, d_sum)
        ratio_sell = safe_divide(d_sum, s_sum)
        min_ratio_buy = np.minimum(1.0, ratio_buy)
        min_ratio_sell = np.minimum(1.0, ratio_sell)
        QB_i_t = D_i_t.multiply(min_ratio_buy, axis=0)
        QS_i_t = S_i_t.multiply(min_ratio_sell, axis=0)
    else:
        QB_i_t = pd.DataFrame(0.0, index=D_i_t.index, columns=D_i_t.columns)
        QS_i_t = pd.DataFrame(0.0, index=S_i_t.index, columns=S_i_t.columns)

    # Eq_Cost_H_i | Cost_H_i = (1 - I_HP) * sum_t((Q_H_i_t / 0.9) * Gas_Price)
    Cost_H_i = (1.0 - I_HP) * (safe_divide(Q_H_i_t, 0.9) * Gas_Price).sum(axis=0)

    # Eq_Cost_T_i | Cost_T_i = (1 - I_EV) * ((km_365 / 100) * fuel_cons * Petrol_Price)
    Cost_T_i = pd.Series((1.0 - I_EV) * (safe_divide(km_365, 100.0) * fuel_cons * Petrol_Price), index=CP_i_t.columns)

    # Eq1 | Pay_EC_0 = sum_t((C_EC_t + sum_i(CP_i_t)) * ToU_t) + sum_i(Cost_H_i + Cost_T_i)
    Pay_EC_0 = ((C_EC_t + CP_i_t.sum(axis=1)) * ToU_t).sum() + Cost_H_i.sum() + Cost_T_i.sum()

    # Eq2 | Pay_EC = sum_t(D_EC_t * ToU_t - S_EC_t * FiT_t) + (1 - I_HP) * sum_i(Cost_H_i) + (1 - I_EV) * sum_i(Cost_T_i)
    Pay_EC = (D_EC_t * ToU_t - S_EC_t * FiT_t).sum() + (1.0 - I_HP) * Cost_H_i.sum() + (1.0 - I_EV) * Cost_T_i.sum()

    # Eq13_Pay_i | Pay_i = (1 - I_LEM) * sum_t(D_i_t * ToU_t - S_i_t * FiT_t) + I_LEM * sum_t(QB_i_t * P_LEM_t + (D_i_t - QB_i_t) * ToU_t - (QS_i_t * P_LEM_t + (S_i_t - QS_i_t) * FiT_t))
    term_no_lem = (1.0 - I_LEM) * (D_i_t.multiply(ToU_t, axis=0) - S_i_t.multiply(FiT_t, axis=0))
    term_lem = I_LEM * (
        QB_i_t.multiply(P_LEM_t, axis=0) +
        (D_i_t - QB_i_t).multiply(ToU_t, axis=0) -
        (QS_i_t.multiply(P_LEM_t, axis=0) + (S_i_t - QS_i_t).multiply(FiT_t, axis=0))
    )
    Pay_i = (term_no_lem + term_lem).sum(axis=0)

    # Eq17 | VS_EC = sum_i(Pay_i) - Pay_EC
    VS_EC = Pay_i.sum() - Pay_EC

    # Eq18_VS_i | VS_i = VS_EC / m
    VS_i = safe_divide(VS_EC, m)

    # Eq18_Pay_final | Pay_i_final = Pay_i - VS_i
    Pay_i_final = Pay_i - VS_i

    # Eq5 | Cost_EC = P_PV * UC_PV + I_HP * UC_HP + I_EV * UC_EVCS
    Cost_EC = P_PV * UC_PV + I_HP * UC_HP + I_EV * UC_EVCS

    # Eq7 | PBP_EC = Cost_EC / (Pay_EC_0 - Pay_EC)
    Pay_EC_diff = Pay_EC_0 - Pay_EC
    if Pay_EC_diff <= 1e-9:
        PBP_EC = 999.0
    else:
        PBP_EC = safe_divide(Cost_EC, Pay_EC_diff, default=999.0)

    # Eq25 | Gu_EC_t = min(C_EC_t + sum_i(C_i_t), G_EC_t + sum_i(G_i_t))
    Gu_EC_t = np.minimum(C_EC_t + C_i_t.sum(axis=1), G_EC_t + G_i_t.sum(axis=1))

    # Eq28 | CO2_EC_0 = sum_t(C_EC_t + sum_i(CP_i_t)) * EF_el + sum_i(sum_t(Q_H_i_t / 0.9)) * EF_H + m * (km_365 / 100) * fuel_cons * EF_T
    CO2_EC_0 = ((C_EC_t + CP_i_t.sum(axis=1)).sum() * EF_el +
                (safe_divide(Q_H_i_t, 0.9).sum().sum() * EF_H) +
                m * safe_divide(km_365, 100.0) * fuel_cons * EF_T)

    # Eq30 | delta_CO2_H = I_HP * (sum_i(sum_t(CH_i_t)) * COP * EF_H - sum_i(sum_t(max(0, CH_i_t - (G_i_t + G_sh_t)))) * EF_el)
    G_i_plus_sh = G_i_t.add(G_sh_t, axis=0)
    net_CH = (CH_i_t - G_i_plus_sh).clip(lower=0.0)
    delta_CO2_H = I_HP * (CH_i_t.sum().sum() * COP * EF_H - net_CH.sum().sum() * EF_el)

    # Eq31 | delta_CO2_EV = I_EV * (m * (km_365 / 100) * fuel_cons * EF_T - sum_t(sum_i(max(0, CEV_i_t - (G_i_t + G_sh_t)))) * EF_el)
    net_CEV = (CEV_i_t - G_i_plus_sh).clip(lower=0.0)
    delta_CO2_EV = I_EV * (m * safe_divide(km_365, 100.0) * fuel_cons * EF_T - net_CEV.sum().sum() * EF_el)

    # Eq29 | delta_CO2 = sum_t(G_EC_t + sum_i(G_i_t) - sum_i(CEV_i_t) - sum_i(CH_i_t)) * EF_el + delta_CO2_H + delta_CO2_EV
    delta_CO2 = (G_EC_t + G_i_t.sum(axis=1) - CEV_i_t.sum(axis=1) - CH_i_t.sum(axis=1)).sum() * EF_el + delta_CO2_H + delta_CO2_EV

    # Eq22 | CS_EC = ((Pay_EC_0 - Pay_EC) / Pay_EC_0) * 100
    CS_EC = safe_divide((Pay_EC_0 - Pay_EC), Pay_EC_0) * 100.0

    # Eq23 | GDI_EC = (sum_t(D_EC_t) / sum_t(C_EC_t + sum_i(C_i_t))) * 100
    GDI_EC = safe_divide(D_EC_t.sum(), (C_EC_t.sum() + C_i_t.sum().sum())) * 100.0

    # Eq24 | SCI_EC = (sum_t(Gu_EC_t) / sum_t(G_EC_t + sum_i(G_i_t))) * 100
    SCI_EC = safe_divide(Gu_EC_t.sum(), (G_EC_t.sum() + G_i_t.sum().sum())) * 100.0

    # Eq26 | SSI_EC = (sum_t(Gu_EC_t) / sum_t(C_EC_t + sum_i(C_i_t))) * 100
    SSI_EC = safe_divide(Gu_EC_t.sum(), (C_EC_t.sum() + C_i_t.sum().sum())) * 100.0

    # Eq27 | FI_EC = (1 / m) * ((sum_i(VS_i / (sum_t(G_i_t) + sum_t(G_sh_t))))^2 / sum_i((VS_i / (sum_t(G_i_t) + sum_t(G_sh_t)))^2)) * 100
    denom = G_i_t.sum(axis=0) + I_ind_cov * G_sh_t.sum()
    normalized_share = safe_divide(VS_i, denom)
    sum_ns = normalized_share.sum()
    sum_ns_sq = (normalized_share ** 2).sum()
    FI_EC = (1.0 / m) * safe_divide(sum_ns ** 2, sum_ns_sq) * 100.0

    # Eq32 | CO2_red_index = (delta_CO2 / CO2_EC_0) * 100
    CO2_red_index = safe_divide(delta_CO2, CO2_EC_0) * 100.0

    # Eq33 | f_EC = omega_1 * FI_EC + omega_2 * CS_EC + omega_3 * SSI_EC + omega_4 * SCI_EC - omega_5 * GDI_EC + omega_6 * (1 / PBP_EC) * 100 + omega_7 * CO2_red_index
    omega_1 = safe_get_parameter(parameters, "omega_1", 1.0)
    omega_2 = safe_get_parameter(parameters, "omega_2", 1.0)
    omega_3 = safe_get_parameter(parameters, "omega_3", 1.0)
    omega_4 = safe_get_parameter(parameters, "omega_4", 1.0)
    omega_5 = safe_get_parameter(parameters, "omega_5", 1.0)
    omega_6 = safe_get_parameter(parameters, "omega_6", 1.0)
    omega_7 = safe_get_parameter(parameters, "omega_7", 1.0)
    f_EC = omega_1 * FI_EC + omega_2 * CS_EC + omega_3 * SSI_EC + omega_4 * SCI_EC - omega_5 * GDI_EC + omega_6 * safe_divide(1.0, PBP_EC) * 100.0 + omega_7 * CO2_red_index

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RESULTS EXPORT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    results = {
        "C_EC_t": C_EC_t,
        "Q_H_i_t": Q_H_i_t,
        "CH_i_t": CH_i_t,
        "CEV_i_t": CEV_i_t,
        "C_i_t": C_i_t,
        "P_PV": P_PV,
        "G_EC_t": G_EC_t,
        "C_sh_t": C_sh_t,
        "G_sh_t": G_sh_t,
        "D_i_t": D_i_t,
        "S_i_t": S_i_t,
        "D_EC_t": D_EC_t,
        "S_EC_t": S_EC_t,
        "QB_i_t": QB_i_t,
        "QS_i_t": QS_i_t,
        "Cost_H_i": Cost_H_i,
        "Cost_T_i": Cost_T_i,
        "Pay_EC_0": Pay_EC_0,
        "Pay_EC": Pay_EC,
        "Pay_i": Pay_i,
        "VS_EC": VS_EC,
        "VS_i": VS_i,
        "Pay_i_final": Pay_i_final,
        "Cost_EC": Cost_EC,
        "PBP_EC": PBP_EC,
        "Gu_EC_t": Gu_EC_t,
        "CO2_EC_0": CO2_EC_0,
        "delta_CO2_H": delta_CO2_H,
        "delta_CO2_EV": delta_CO2_EV,
        "delta_CO2": delta_CO2,
        "CS_EC": CS_EC,
        "GDI_EC": GDI_EC,
        "SCI_EC": SCI_EC,
        "SSI_EC": SSI_EC,
        "FI_EC": FI_EC,
        "CO2_red_index": CO2_red_index,
        "f_EC": f_EC
    }

    return {k: to_jsonable(v) for k, v in results.items()}


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
