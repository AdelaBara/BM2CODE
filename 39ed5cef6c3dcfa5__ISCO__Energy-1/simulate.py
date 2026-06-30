"""simulate_value_share.py
A computational model for energy supply planning in collective self-consumption communities that ensures fair [...]

Generated from: output/39ed5cef6c3dcfa5__ISCO__Energy-1/39ed5cef6c3dcfa5__ISCO__Energy-1.pdf
BM name       : Value Share

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
    pd = globals().get('pd') or __import__('pandas')
    np = globals().get('np') or __import__('numpy')
    linprog = globals().get('linprog') or __import__('scipy.optimize', fromlist=['linprog']).linprog

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # REQUIRED HELPER FUNCTIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def safe_numeric(value, default=None):
        if isinstance(value, pd.Series):
            res = pd.to_numeric(value, errors="coerce")
            if default is not None:
                res = res.fillna(default)
            return res
        else:
            try:
                num = pd.to_numeric(value, errors="coerce")
                if pd.isna(num):
                    return default
                return float(num)
            except Exception:
                return default

    def safe_datetime(value):
        return pd.to_datetime(value, errors="coerce")

    def safe_divide(a, b, default=0.0):
        a_num = safe_numeric(a, default=0.0)
        b_num = safe_numeric(b, default=0.0)
        if isinstance(a_num, pd.Series) or isinstance(b_num, pd.Series):
            cond = (b_num.abs() <= 1e-9) | b_num.isna()
            return np.where(cond, default, a_num / b_num)
        else:
            if abs(b_num) <= 1e-9 or b_num is None:
                return default
            res = a_num / b_num
            if not np.isfinite(res):
                return default
            return float(res)

    def safe_get_parameter(parameters, name, default):
        val = parameters.get(name, default)
        if val is None:
            return default
        num = safe_numeric(val, default=None)
        return default if num is None else num

    def to_jsonable(value):
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        elif isinstance(value, pd.Series):
            return value.to_dict()
        elif isinstance(value, dict):
            return {k: to_jsonable(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [to_jsonable(v) for v in value]
        elif isinstance(value, (np.integer, np.floating)):
            val = value.item()
            return None if pd.isna(val) else val
        elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return None
        elif pd.isna(value):
            return None
        else:
            return value

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DATA VALIDATION STAGE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    parameters = parameters or {}
    if df is None or not hasattr(df, "columns"):
        raise ValueError("Input df is None or not a DataFrame")

    required_cols = ["Timestamp", "member_id", "load", "generation"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    df["Timestamp"] = safe_datetime(df["Timestamp"])
    df["load"] = safe_numeric(df["load"], default=0.0)
    df["generation"] = safe_numeric(df["generation"], default=0.0)

    df = df.dropna(subset=["Timestamp"])
    if df.empty:
        raise ValueError("No valid rows remaining after dropping invalid Timestamps")

    df["member_id"] = df["member_id"].astype(str).str.strip()
    if (df["member_id"] == "").any() or df["member_id"].isna().any():
        raise ValueError("Some member_id values are missing or blank")

    df = df.sort_values(by="Timestamp").reset_index(drop=True)

    duplicates = df.duplicated(subset=["member_id", "Timestamp"]).any()
    if duplicates:
        raise ValueError("Duplicate rows detected for member_id and Timestamp")

    if tariffs is None or not hasattr(tariffs, "columns"):
        raise ValueError("tariffs must be a pandas DataFrame")
    if "time" not in tariffs.columns:
        raise ValueError("tariffs must contain a 'time' column")

    tariffs_df = tariffs.copy()
    tariffs_df["time"] = safe_numeric(tariffs_df["time"], default=-1)

    required_tariff_keys = ["ToU", "FiT"]
    for key in required_tariff_keys:
        if key not in tariffs_df.columns:
            raise ValueError(f"Tariff column '{key}' is missing")
        tariffs_df[key] = safe_numeric(tariffs_df[key], default=None)
        if tariffs_df[key].isna().any():
            raise ValueError(f"Tariff column '{key}' contains NaN or non-numeric values")

    available_hours = set(tariffs_df["time"].astype(int))
    required_hours = set(df["Timestamp"].dt.hour.unique())
    missing_hours = required_hours - available_hours
    if missing_hours:
        raise ValueError(f"Missing tariff values for hours: {missing_hours}")

    tariffs_map = tariffs_df.set_index("time")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PARAMETERS INITIALIZATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    delta = safe_get_parameter(parameters, "delta", 0.25)
    Bt_param = parameters.get("Bt", None)
    beta_param = parameters.get("beta", None)
    mu = safe_get_parameter(parameters, "mu", 0.0)
    ec = safe_get_parameter(parameters, "ec", 0.95)
    ed = safe_get_parameter(parameters, "ed", 0.95)
    Cs = safe_get_parameter(parameters, "Cs", 10.5)
    Smin = safe_get_parameter(parameters, "Smin", 0.0)
    F_charge = safe_get_parameter(parameters, "F_charge", 5.0)
    F_discharge = safe_get_parameter(parameters, "F_discharge", 5.0)
    s0 = safe_get_parameter(parameters, "s0", 5.25)

    # Validation Checks
    if (df["load"] < 0).any():
        raise ValueError("Load (Dj,t) must be non-negative")
    if (df["generation"] < 0).any():
        raise ValueError("Generation (CPV_t) must be non-negative")
    if not (Cs >= Smin >= 0):
        raise ValueError("Battery capacity parameters must satisfy Cs >= Smin >= 0")
    if not (0 < ec <= 1.0) or not (0 < ed <= 1.0):
        raise ValueError("Charging and discharging efficiencies must be in (0, 1]")

    RULE = str(parameters.get("RULE", "None")).strip().lower()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PROCESS INPUTS & CONSTRUCT MATRICES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    D_pivot = df.pivot(index="Timestamp", columns="member_id", values="load").fillna(0.0)
    member_ids = sorted(list(D_pivot.columns))
    timestamps = sorted(list(D_pivot.index))
    D_pivot = D_pivot.reindex(index=timestamps, columns=member_ids)

    N_J = len(member_ids)
    N_T = len(timestamps)

    CPV_series = df.groupby("Timestamp")["generation"].mean().reindex(timestamps).fillna(0.0)
    CPV_list = CPV_series.tolist()

    # Eq_D_j | D_j == sum(Dj,t for t in T)
    D_j_dict = {j_id: float(D_pivot[j_id].sum()) for j_id in member_ids}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DEFINE LP INDEX MAPS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def idx(j, t, var_type):
        return (j * N_T + t) * 5 + var_type

    def s_idx(t):
        return 5 * N_J * N_T + t

    if RULE == "proportional":
        num_vars = 5 * N_J * N_T + N_T + 1
    else:
        num_vars = 5 * N_J * N_T + N_T

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # VARIABLE BOUNDS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    bounds = []
    for j in range(N_J):
        for t in range(N_T):
            bounds.extend([(0.0, None)] * 5)
    # Eq_2 | Smin <= st <= Cs
    for t in range(N_T):
        bounds.append((Smin, Cs))
    if RULE == "proportional":
        bounds.append((0.0, None))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # COST FUNCTION (Eq_1)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    c = [0.0] * num_vars
    Bt_list = []
    beta_list = []
    for t_val in timestamps:
        hr = t_val.hour
        if Bt_param is not None:
            Bt_list.append(safe_numeric(Bt_param, default=0.0))
        else:
            Bt_list.append(float(tariffs_map.loc[hr, "ToU"]))
            
        if beta_param is not None:
            beta_list.append(safe_numeric(beta_param, default=0.0))
        else:
            beta_list.append(float(tariffs_map.loc[hr, "FiT"]))

    for j in range(N_J):
        for t in range(N_T):
            c[idx(j, t, 1)] = delta * mu
            c[idx(j, t, 2)] = delta * Bt_list[t]
            c[idx(j, t, 4)] = -delta * beta_list[t]

    A_eq = []
    b_eq = []
    A_ub = []
    b_ub = []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CONSTRAINTS FORMULATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Eq_4 | st = st-1 + ec * sum(zj,t for j in J) - ed * sum(yj,t for j in J)
    for t in range(N_T):
        row = [0.0] * num_vars
        row[s_idx(t)] = 1.0
        if t == 0:
            for j in range(N_J):
                row[idx(j, 0, 3)] = -ec
                row[idx(j, 0, 1)] = ed
            A_eq.append(row)
            b_eq.append(s0)
        else:
            row[s_idx(t - 1)] = -1.0
            for j in range(N_J):
                row[idx(j, t, 3)] = -ec
                row[idx(j, t, 1)] = ed
            A_eq.append(row)
            b_eq.append(0.0)

    # Eq_5 | s_T == s_0
    row_last_soc = [0.0] * num_vars
    row_last_soc[s_idx(N_T - 1)] = 1.0
    A_eq.append(row_last_soc)
    b_eq.append(s0)

    # Eq_3 | sum(pj,t for j in J) == CPV_t
    for t in range(N_T):
        row = [0.0] * num_vars
        for j in range(N_J):
            row[idx(j, t, 0)] = 1.0
        A_eq.append(row)
        b_eq.append(CPV_list[t])

    # Eq_8 | Dj,t = pj,t + yj,t + ij,t - zj,t - gj,t
    for j in range(N_J):
        for t in range(N_T):
            row = [0.0] * num_vars
            row[idx(j, t, 0)] = 1.0
            row[idx(j, t, 1)] = 1.0
            row[idx(j, t, 2)] = 1.0
            row[idx(j, t, 3)] = -1.0
            row[idx(j, t, 4)] = -1.0
            A_eq.append(row)
            b_eq.append(float(D_pivot.iloc[t, j]))

    # Eq_6 | sum(yj,t for j in J) <= F_discharge * delta
    for t in range(N_T):
        row = [0.0] * num_vars
        for j in range(N_J):
            row[idx(j, t, 1)] = 1.0
        A_ub.append(row)
        b_ub.append(F_discharge * delta)

    # Eq_7 | sum(zj,t for j in J) <= F_charge * delta
    for t in range(N_T):
        row = [0.0] * num_vars
        for j in range(N_J):
            row[idx(j, t, 3)] = 1.0
        A_ub.append(row)
        b_ub.append(F_charge * delta)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FAIRNESS RULE CONSTRAINTS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if RULE == "proportional":
        # Eq_18 | sum(p1,t for t in T) / sum(D1,t for t in T) == sum(pj,t for t in T) / sum(Dj,t for t in T)
        for j in range(N_J):
            row = [0.0] * num_vars
            for t in range(N_T):
                row[idx(j, t, 0)] = 1.0
            row[-1] = -D_j_dict[member_ids[j]]
            A_eq.append(row)
            b_eq.append(0.0)

    elif RULE == "maxmin":
        # Eq_20_21 | sum(pj,t for t in T) <= min(sum(Dj,t for t in T), (sum(CPV_t for t in T) - sum(pk,t for k in range(1, j) for t in T)) / (|J| - j + 1))
        J_info = [(j, member_ids[j], D_j_dict[member_ids[j]]) for j in range(N_J)]
        J_info_sorted = sorted(J_info, key=lambda x: x[2])

        for idx_sorted, (j, j_id, D_j_val) in enumerate(J_info_sorted):
            row1 = [0.0] * num_vars
            for t in range(N_T):
                row1[idx(j, t, 0)] = 1.0
            A_ub.append(row1)
            b_ub.append(D_j_val)

            row2 = [0.0] * num_vars
            factor = float(N_J - idx_sorted)
            for t in range(N_T):
                row2[idx(j, t, 0)] = factor
            for prev_idx in range(idx_sorted):
                prev_j = J_info_sorted[prev_idx][0]
                for t in range(N_T):
                    row2[idx(prev_j, t, 0)] = 1.0
            A_ub.append(row2)
            b_ub.append(sum(CPV_list))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LP SOLVER EXECUTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    res = linprog(
        c,
        A_ub=A_ub if A_ub else None,
        b_ub=b_ub if b_ub else None,
        A_eq=A_eq if A_eq else None,
        b_eq=b_eq if b_eq else None,
        bounds=bounds,
        method="highs",
    )

    if res.x is None or not res.success:
        raise ValueError(f"Optimization failed: {res.message}")

    x = res.x

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # POST-PROCESSING & METRIC EXTRACTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    pj_t_dict = {}
    yj_t_dict = {}
    ij_t_dict = {}
    zj_t_dict = {}
    gj_t_dict = {}

    for j in range(N_J):
        j_id = member_ids[j]
        pj_t_dict[j_id] = [float(x[idx(j, t, 0)]) for t in range(N_T)]
        yj_t_dict[j_id] = [float(x[idx(j, t, 1)]) for t in range(N_T)]
        ij_t_dict[j_id] = [float(x[idx(j, t, 2)]) for t in range(N_T)]
        zj_t_dict[j_id] = [float(x[idx(j, t, 3)]) for t in range(N_T)]
        gj_t_dict[j_id] = [float(x[idx(j, t, 4)]) for t in range(N_T)]

    st_list = [float(x[s_idx(t)]) for t in range(N_T)]

    # Eq_P_j | P_j == sum(pj,t for t in T)
    P_j_dict = {j_id: sum(pj_t_dict[j_id]) for j_id in member_ids}

    # lambda_val calculation
    if RULE == "proportional":
        lambda_val = float(x[-1])
    else:
        ratios = []
        for j_id in member_ids:
            pj_val = P_j_dict[j_id]
            dj_val = D_j_dict[j_id]
            ratios.append(safe_divide(pj_val, dj_val))
        lambda_val = float(np.mean(ratios)) if ratios else 0.0

    # Eq_1 | Total_Cost = delta * sum(mu * yj,t + Bt * ij,t - beta * gj,t for j in J for t in T)
    total_cost_val = 0.0
    for j in range(N_J):
        j_id = member_ids[j]
        for t in range(N_T):
            y_val = yj_t_dict[j_id][t]
            i_val = ij_t_dict[j_id][t]
            g_val = gj_t_dict[j_id][t]
            total_cost_val += mu * y_val + Bt_list[t] * i_val - beta_list[t] * g_val
    total_cost_val *= delta

    results = {
        "Total_Cost": total_cost_val,
        "pj,t": pj_t_dict,
        "yj,t": yj_t_dict,
        "ij,t": ij_t_dict,
        "zj,t": zj_t_dict,
        "gj,t": gj_t_dict,
        "st": st_list,
        "P_j": P_j_dict,
        "D_j": D_j_dict,
        "lambda_val": lambda_val,
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
