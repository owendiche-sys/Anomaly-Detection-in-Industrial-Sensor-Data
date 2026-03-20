
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_DATA_FILE = "data.csv"


# =========================
# UI
# =========================
def inject_css() -> None:
    st.markdown(
        """
        <style>
          .stApp {
            background: #fafafa;
            color: #111827;
          }
          section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid rgba(15, 23, 42, 0.08);
          }
          .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1350px;
          }
          .hero {
            background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 22px;
            padding: 22px 22px 18px 22px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            margin-bottom: 16px;
          }
          .hero-title {
            font-size: 2rem;
            line-height: 1.1;
            font-weight: 800;
            margin-bottom: 8px;
            color: #0f172a;
          }
          .hero-text {
            color: rgba(15, 23, 42, 0.76);
            font-size: 1rem;
            line-height: 1.65;
            margin-bottom: 0;
          }
          .section-label {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #475569;
            margin-bottom: 10px;
          }
          .card {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 18px;
            padding: 16px 16px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
          }
          .card h3, .card h4 {
            margin: 0 0 10px 0;
            color: #0f172a;
          }
          .muted {
            color: rgba(17, 24, 39, 0.78);
            font-size: 0.95rem;
            line-height: 1.7;
          }
          .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 12px;
          }
          .kpi {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 18px;
            padding: 15px 15px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
          }
          .kpi .label {
            color: rgba(17, 24, 39, 0.70);
            font-size: 0.84rem;
            margin-bottom: 6px;
          }
          .kpi .value {
            font-size: 1.45rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.08;
          }
          .kpi .sub {
            margin-top: 7px;
            color: rgba(17, 24, 39, 0.66);
            font-size: 0.84rem;
          }
          .insight-box {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
          }
          .insight-box h3 {
            margin: 0 0 10px 0;
            color: #0f172a;
          }
          .insight-box p {
            color: rgba(17, 24, 39, 0.80);
            line-height: 1.7;
            margin: 0;
          }
          .mini-note {
            color: rgba(15, 23, 42, 0.70);
            font-size: 0.9rem;
            line-height: 1.6;
          }
          @media (max-width: 1100px) {
            .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          }
          @media (max-width: 640px) {
            .kpi-grid { grid-template-columns: repeat(1, minmax(0, 1fr)); }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, text: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="section-label">Industrial anomaly monitoring</div>'
        f'<div class="hero-title">{title}</div><p class="hero-text">{text}</p></div>',
        unsafe_allow_html=True,
    )


def card(title: str, body_html: str) -> None:
    st.markdown(
        f'<div class="card"><h3>{title}</h3><div class="muted">{body_html}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (
        '<div class="kpi">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'{sub_html}'
        '</div>'
    )


# =========================
# Loading and normalization
# =========================
@dataclass
class DataMeta:
    source_label: str
    raw_rows: int
    raw_cols: int


def _decode_bytes(b: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode(errors="ignore")


def _detect_delimiter(sample_text: str) -> str:
    lines = [ln for ln in sample_text.splitlines() if ln.strip()]
    if not lines:
        return ","
    header = lines[0]
    candidates = [",", ";", "\t", "|"]
    counts = {c: header.count(c) for c in candidates}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def safe_read_csv(file_bytes: bytes) -> pd.DataFrame:
    txt = _decode_bytes(file_bytes[:4096])
    sep = _detect_delimiter(txt)

    df = None
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
    except Exception:
        pass

    if df is None:
        for candidate in [";", ",", "\t", "|"]:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), sep=candidate)
                break
            except Exception:
                df = None

    if df is None:
        df = pd.read_csv(io.BytesIO(file_bytes))

    return df


@st.cache_data(show_spinner=False)
def load_dataset(use_uploaded: bool, uploaded_file) -> Tuple[pd.DataFrame, DataMeta]:
    if use_uploaded and uploaded_file is not None:
        raw = uploaded_file.read()
        df = safe_read_csv(raw)
        return df, DataMeta(
            source_label=f"Uploaded file: {uploaded_file.name}",
            raw_rows=int(df.shape[0]),
            raw_cols=int(df.shape[1]),
        )

    if not os.path.exists(DEFAULT_DATA_FILE):
        raise FileNotFoundError(
            f"Could not find {DEFAULT_DATA_FILE} next to app.py. "
            "Place the file in the same folder as app.py, or upload a CSV in the sidebar."
        )

    with open(DEFAULT_DATA_FILE, "rb") as f:
        raw = f.read()

    df = safe_read_csv(raw)
    return df, DataMeta(
        source_label=f"Default file: {DEFAULT_DATA_FILE}",
        raw_rows=int(df.shape[0]),
        raw_cols=int(df.shape[1]),
    )


def _find_time_column(df: pd.DataFrame) -> Optional[str]:
    preferred = ["timestamp", "time", "datetime", "date"]
    lower_map = {c.lower(): c for c in df.columns}
    for term in preferred:
        for lc, orig in lower_map.items():
            if lc == term or term in lc:
                return orig
    for c in df.columns:
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() > 0.6:
            return c
    return None


def _find_sensor_column(df: pd.DataFrame) -> Optional[str]:
    preferred = ["sensorid", "sensor_id", "sensor", "machine", "device"]
    lower_map = {c.lower(): c for c in df.columns}
    for term in preferred:
        for lc, orig in lower_map.items():
            if lc == term or term in lc:
                return orig
    return None


def _find_value_column(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> Optional[str]:
    exclude = exclude or []
    preferred = ["value", "reading", "measurement", "metric"]
    lower_map = {c.lower(): c for c in df.columns if c not in exclude}
    for term in preferred:
        for lc, orig in lower_map.items():
            if lc == term or term in lc:
                return orig
    numeric_candidates = []
    for c in df.columns:
        if c in exclude:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.7:
            numeric_candidates.append((c, s.notna().mean()))
    if not numeric_candidates:
        return None
    numeric_candidates.sort(key=lambda x: x[1], reverse=True)
    return numeric_candidates[0][0]


def _split_combined_column(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    object_cols = [c for c in df.columns if df[c].dtype == "object"]
    for c in object_cols:
        sample = df[c].dropna().astype(str).head(25)
        if len(sample) == 0:
            continue
        semi_share = (sample.str.count(";") >= 2).mean()
        if semi_share >= 0.6:
            parts = df[c].astype(str).str.split(";", expand=True)
            if parts.shape[1] >= 3:
                out = pd.DataFrame({
                    "Timestamp": parts.iloc[:, 0],
                    "SensorId": parts.iloc[:, 1],
                    "Value": parts.iloc[:, 2],
                })
                return out
    return None


@st.cache_data(show_spinner=False)
def prepare_sensor_data(df_raw: pd.DataFrame) -> Dict[str, object]:
    df = df_raw.copy()

    combined = _split_combined_column(df)
    if combined is not None:
        long_df = combined.copy()
    else:
        time_col = _find_time_column(df)
        sensor_col = _find_sensor_column(df)
        value_col = _find_value_column(df, exclude=[c for c in [time_col, sensor_col] if c is not None])

        if time_col is not None and sensor_col is not None and value_col is not None:
            long_df = df[[time_col, sensor_col, value_col]].copy()
            long_df.columns = ["Timestamp", "SensorId", "Value"]
        else:
            # Treat as already wide if at least one numeric sensor column exists
            wide_df = df.copy()
            wide_time_col = _find_time_column(wide_df)
            if wide_time_col is not None:
                wide_df[wide_time_col] = pd.to_datetime(wide_df[wide_time_col], errors="coerce")
                wide_df = wide_df.dropna(subset=[wide_time_col]).sort_values(wide_time_col).reset_index(drop=True)
                wide_df = wide_df.rename(columns={wide_time_col: "Timestamp"})
            else:
                wide_df = wide_df.reset_index(drop=True)
                wide_df.insert(0, "RowIndex", np.arange(len(wide_df)))
            sensor_cols = [c for c in wide_df.columns if c not in {"Timestamp", "RowIndex"}]
            for c in sensor_cols:
                wide_df[c] = pd.to_numeric(wide_df[c], errors="coerce")
            sensor_cols = [c for c in sensor_cols if wide_df[c].notna().mean() > 0.5]
            keep_cols = ["Timestamp"] if "Timestamp" in wide_df.columns else ["RowIndex"]
            keep_cols += sensor_cols
            wide_df = wide_df[keep_cols].copy()

            if "Timestamp" in wide_df.columns:
                long_df = (
                    wide_df.melt(id_vars=["Timestamp"], value_vars=sensor_cols, var_name="SensorId", value_name="Value")
                    .dropna(subset=["Value"])
                    .reset_index(drop=True)
                )
            else:
                long_df = (
                    wide_df.melt(id_vars=["RowIndex"], value_vars=sensor_cols, var_name="SensorId", value_name="Value")
                    .dropna(subset=["Value"])
                    .rename(columns={"RowIndex": "RowIndex"})
                    .reset_index(drop=True)
                )
                long_df["Timestamp"] = pd.NaT

            return _finalize_sensor_bundle(long_df, wide_df)

    return _finalize_sensor_bundle(long_df, None)


def _finalize_sensor_bundle(long_df: pd.DataFrame, prebuilt_wide: Optional[pd.DataFrame]) -> Dict[str, object]:
    d = long_df.copy()
    d["Value"] = pd.to_numeric(d["Value"], errors="coerce")
    d["Timestamp"] = pd.to_datetime(d["Timestamp"], errors="coerce")
    d["SensorId"] = d["SensorId"].astype(str)

    d = d.dropna(subset=["Value"]).reset_index(drop=True)

    if d["Timestamp"].notna().sum() > 0:
        d = d.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
        wide = (
            d.pivot_table(index="Timestamp", columns="SensorId", values="Value", aggfunc="mean")
            .sort_index()
            .reset_index()
        )
        time_col = "Timestamp"
    else:
        d["RowIndex"] = np.arange(len(d))
        wide = (
            d.pivot_table(index="RowIndex", columns="SensorId", values="Value", aggfunc="mean")
            .sort_index()
            .reset_index()
        )
        time_col = None

    if prebuilt_wide is not None:
        wide = prebuilt_wide.copy()
        time_col = "Timestamp" if "Timestamp" in wide.columns else None

    if time_col is not None:
        wide = wide.sort_values(time_col).reset_index(drop=True)
    else:
        wide = wide.sort_values(wide.columns[0]).reset_index(drop=True)

    sensor_cols = [c for c in wide.columns if c not in {"Timestamp", "RowIndex"}]
    for c in sensor_cols:
        wide[c] = pd.to_numeric(wide[c], errors="coerce")
    sensor_cols = [c for c in sensor_cols if wide[c].notna().sum() > 0]

    if time_col is not None:
        wide = wide[["Timestamp"] + sensor_cols].copy()
    else:
        keep_lead = "RowIndex" if "RowIndex" in wide.columns else wide.columns[0]
        wide = wide[[keep_lead] + sensor_cols].copy()

    return {
        "long_df": d,
        "wide_df": wide,
        "time_col": time_col,
        "sensor_cols": sensor_cols,
    }


# =========================
# Feature engineering
# =========================
def robust_abs_z_matrix(sensor_df: pd.DataFrame) -> pd.DataFrame:
    x = sensor_df.apply(pd.to_numeric, errors="coerce")
    med = x.median(axis=0)
    mad = (x - med).abs().median(axis=0)
    scale = 1.4826 * mad.replace(0, np.nan)
    z = ((x - med) / scale).abs()
    return z.replace([np.inf, -np.inf], np.nan)


@st.cache_data(show_spinner=False)
def build_row_level_features(wide_df: pd.DataFrame, time_col: Optional[str], sensor_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = wide_df.copy()
    sensors = d[sensor_cols].apply(pd.to_numeric, errors="coerce")
    sensors = sensors.loc[:, sensors.notna().sum() > 0].copy()

    row = pd.DataFrame(index=d.index)
    row["sensor_mean"] = sensors.mean(axis=1)
    row["sensor_median"] = sensors.median(axis=1)
    row["sensor_std"] = sensors.std(axis=1, ddof=0).fillna(0.0)
    row["sensor_min"] = sensors.min(axis=1)
    row["sensor_max"] = sensors.max(axis=1)
    row["sensor_range"] = (row["sensor_max"] - row["sensor_min"]).fillna(0.0)
    row["active_sensor_count"] = sensors.notna().sum(axis=1)

    robust_z = robust_abs_z_matrix(sensors)
    row["max_abs_sensor_z"] = robust_z.max(axis=1)
    row["mean_abs_sensor_z"] = robust_z.mean(axis=1)
    row["sensor_count_over_3z"] = (robust_z >= 3).sum(axis=1)

    row["mean_lag1"] = row["sensor_mean"].shift(1)
    row["mean_delta_1"] = row["sensor_mean"] - row["mean_lag1"]

    window = max(12, min(72, max(12, len(row) // 20)))
    row["rolling_mean"] = row["sensor_mean"].rolling(window=window, min_periods=max(4, window // 3)).mean()
    row["rolling_std"] = row["sensor_mean"].rolling(window=window, min_periods=max(4, window // 3)).std()
    row["rolling_delta"] = row["sensor_mean"] - row["rolling_mean"]

    if time_col is not None and time_col in d.columns:
        t = pd.to_datetime(d[time_col], errors="coerce")
        row["hour"] = t.dt.hour
        row["day_of_week"] = t.dt.dayofweek
        row["month"] = t.dt.month
    else:
        row["hour"] = np.nan
        row["day_of_week"] = np.nan
        row["month"] = np.nan

    aligned = pd.concat([d[[time_col]].copy(), row], axis=1) if time_col is not None else pd.concat([d.iloc[:, [0]].copy(), row], axis=1)
    return sensors, row, aligned


# =========================
# Modeling
# =========================
@st.cache_resource(show_spinner=False)
def fit_isolation_forest(feature_df: pd.DataFrame, contamination: float, random_state: int) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    model = IsolationForest(
        n_estimators=400,
        contamination=float(np.clip(contamination, 0.001, 0.20)),
        random_state=int(random_state),
        n_jobs=-1,
    )
    pipe = Pipeline(steps=[("prep", numeric_pipe), ("model", model)])
    pipe.fit(feature_df)
    return pipe


def run_isolation_forest(pipe: Pipeline, feature_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    pred = pipe.predict(feature_df)
    normal_score = pipe.decision_function(feature_df)
    anomaly_score = -normal_score
    flags = (pred == -1).astype(int)
    return flags, anomaly_score


def flag_from_percentile(scores: np.ndarray, percentile: float) -> Tuple[np.ndarray, float]:
    s = pd.Series(scores, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return np.zeros(len(scores), dtype=int), np.nan
    threshold = float(np.quantile(s, percentile))
    flags = (np.asarray(scores) >= threshold).astype(int)
    return flags, threshold


def threshold_tradeoff(scores: np.ndarray) -> pd.DataFrame:
    s = pd.Series(scores, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return pd.DataFrame(columns=["Percentile", "Threshold", "Flagged points", "Share"])
    percentiles = np.linspace(0.80, 0.99, 10)
    rows = []
    for p in percentiles:
        thr = float(np.quantile(s, p))
        count = int((scores >= thr).sum())
        rows.append([p, thr, count, count / len(scores)])
    return pd.DataFrame(rows, columns=["Percentile", "Threshold", "Flagged points", "Share"])


def score_stability(if_scores: np.ndarray, row_df: pd.DataFrame) -> Dict[str, float]:
    ref = pd.to_numeric(row_df["max_abs_sensor_z"], errors="coerce").to_numpy(dtype=float)
    a = np.asarray(if_scores, dtype=float)
    mask = np.isfinite(a) & np.isfinite(ref)
    if mask.sum() < 5:
        return {"score_correlation": np.nan, "top5_overlap": np.nan, "top10_overlap": np.nan}
    a = a[mask]
    ref = ref[mask]
    corr = float(np.corrcoef(a, ref)[0, 1]) if np.std(a) > 0 and np.std(ref) > 0 else np.nan

    def overlap(p: float) -> float:
        a_thr = np.quantile(a, 1 - p)
        r_thr = np.quantile(ref, 1 - p)
        a_set = set(np.where(a >= a_thr)[0].tolist())
        r_set = set(np.where(ref >= r_thr)[0].tolist())
        union = len(a_set | r_set)
        return float(len(a_set & r_set) / union) if union else np.nan

    return {
        "score_correlation": corr,
        "top5_overlap": overlap(0.05),
        "top10_overlap": overlap(0.10),
    }


def longest_run(flags: np.ndarray) -> Tuple[int, Optional[int], Optional[int]]:
    arr = np.asarray(flags).astype(int)
    best_len = 0
    best_end = None
    cur_len = 0
    for i, val in enumerate(arr):
        if val == 1:
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_end = i
        else:
            cur_len = 0
    if best_end is None:
        return 0, None, None
    best_start = best_end - best_len + 1
    return best_len, best_start, best_end


def point_label(wide_df: pd.DataFrame, idx: Optional[int], time_col: Optional[str]) -> str:
    if idx is None or idx < 0 or idx >= len(wide_df):
        return "Not available"
    if time_col is not None and time_col in wide_df.columns:
        ts = pd.to_datetime(wide_df.iloc[idx][time_col], errors="coerce")
        if pd.notna(ts):
            return ts.strftime("%Y-%m-%d %H:%M")
    lead_col = wide_df.columns[0]
    return f"{lead_col} {wide_df.iloc[idx][lead_col]}"


def top_sensor_volatility(sensor_df: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame({
        "Sensor": sensor_df.columns,
        "Mean": sensor_df.mean(axis=0).values,
        "Std": sensor_df.std(axis=0).values,
        "Missing share": sensor_df.isna().mean(axis=0).values,
    })
    stats["Coefficient of variation"] = stats["Std"] / stats["Mean"].abs().replace(0, np.nan)
    return stats.sort_values(["Std", "Missing share"], ascending=[False, True]).reset_index(drop=True)


def top_event_sensor_contributions(sensor_df: pd.DataFrame, event_idx: int) -> pd.DataFrame:
    robust_z = robust_abs_z_matrix(sensor_df)
    row = robust_z.iloc[event_idx].sort_values(ascending=False)
    actual = sensor_df.iloc[event_idx]
    baseline = sensor_df.median(axis=0)
    out = pd.DataFrame({
        "Sensor": row.index,
        "Abs robust z": row.values,
        "Event value": actual[row.index].values,
        "Baseline median": baseline[row.index].values,
        "Deviation from baseline": (actual[row.index] - baseline[row.index]).values,
    })
    return out.dropna(subset=["Abs robust z"]).reset_index(drop=True)


def compute_story(wide_df: pd.DataFrame, time_col: Optional[str], sensor_df: pd.DataFrame, row_df: pd.DataFrame, flags: np.ndarray, scores: np.ndarray) -> Dict[str, object]:
    mean_series = pd.to_numeric(row_df["sensor_mean"], errors="coerce")
    recent_n = max(12, min(96, max(12, len(mean_series) // 10)))
    recent = mean_series.tail(recent_n)
    baseline = mean_series.iloc[:-recent_n] if len(mean_series) > recent_n else mean_series

    level_change = np.nan
    if baseline.notna().sum() > 0 and recent.notna().sum() > 0 and abs(float(baseline.mean())) > 1e-12:
        level_change = 100 * (float(recent.mean()) - float(baseline.mean())) / abs(float(baseline.mean()))

    vol_change = np.nan
    if baseline.notna().sum() > 1 and recent.notna().sum() > 1 and abs(float(baseline.std())) > 1e-12:
        vol_change = 100 * (float(recent.std()) - float(baseline.std())) / abs(float(baseline.std()))

    delta = pd.to_numeric(row_df["mean_delta_1"], errors="coerce")
    jump_idx = int(delta.abs().idxmax()) if delta.notna().any() else None
    rise_idx = int(delta.idxmax()) if delta.notna().any() else None
    drop_idx = int(delta.idxmin()) if delta.notna().any() else None
    top_idx = int(np.nanargmax(scores)) if np.isfinite(scores).any() else None
    run_len, run_start, run_end = longest_run(flags)

    hot_hour = None
    hot_day = None
    if time_col is not None and time_col in wide_df.columns:
        t = pd.to_datetime(wide_df[time_col], errors="coerce")
        flagged = np.asarray(flags).astype(bool)
        if flagged.any():
            hours = t.dt.hour[flagged]
            days = t.dt.day_name()[flagged]
            if hours.notna().any():
                hot_hour = int(hours.value_counts().idxmax())
            if days.notna().any():
                hot_day = str(days.value_counts().idxmax())

    return {
        "rows": len(wide_df),
        "sensors": sensor_df.shape[1],
        "anomaly_count": int(np.asarray(flags).sum()),
        "anomaly_share": float(np.asarray(flags).mean()) if len(flags) else np.nan,
        "level_change": level_change,
        "vol_change": vol_change,
        "jump_value": float(delta.abs().max()) if delta.notna().any() else np.nan,
        "jump_when": point_label(wide_df, jump_idx, time_col),
        "rise_value": float(delta.max()) if delta.notna().any() else np.nan,
        "rise_when": point_label(wide_df, rise_idx, time_col),
        "drop_value": float(delta.min()) if delta.notna().any() else np.nan,
        "drop_when": point_label(wide_df, drop_idx, time_col),
        "top_score": float(np.nanmax(scores)) if np.isfinite(scores).any() else np.nan,
        "top_when": point_label(wide_df, top_idx, time_col),
        "top_idx": top_idx,
        "top_value": float(mean_series.iloc[top_idx]) if top_idx is not None else np.nan,
        "run_len": run_len,
        "run_start": point_label(wide_df, run_start, time_col) if run_start is not None else "Not available",
        "run_end": point_label(wide_df, run_end, time_col) if run_end is not None else "Not available",
        "hot_hour": hot_hour,
        "hot_day": hot_day,
        "p05": float(mean_series.quantile(0.05)) if mean_series.notna().any() else np.nan,
        "p95": float(mean_series.quantile(0.95)) if mean_series.notna().any() else np.nan,
        "median": float(mean_series.median()) if mean_series.notna().any() else np.nan,
        "recent_n": recent_n,
    }


# =========================
# Formatting
# =========================
def fmt_num(x: float, digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "Not available"
    return f"{float(x):,.{digits}f}"


def fmt_pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "Not available"
    return f"{float(x):+.1f}%"


# =========================
# Plotting
# =========================
def plot_signal_with_flags(wide_df: pd.DataFrame, time_col: Optional[str], row_df: pd.DataFrame, flags: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    x = wide_df[time_col] if time_col is not None and time_col in wide_df.columns else np.arange(len(wide_df))
    y = pd.to_numeric(row_df["sensor_mean"], errors="coerce").to_numpy(dtype=float)

    fig = plt.figure(figsize=(10, 4.3))
    plt.plot(x, y, linewidth=1.1)
    idx = np.where(np.asarray(flags).astype(int) == 1)[0]
    if len(idx) > 0:
        if hasattr(x, "iloc"):
            plt.scatter(x.iloc[idx], y[idx], s=16)
        else:
            plt.scatter(np.asarray(x)[idx], y[idx], s=16)
    plt.title(title)
    plt.xlabel("Timestamp" if time_col is not None else "Row")
    plt.ylabel("Cross-sensor mean")
    st.pyplot(fig, clear_figure=True)


def plot_signal_and_trend(wide_df: pd.DataFrame, time_col: Optional[str], row_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    x = wide_df[time_col] if time_col is not None and time_col in wide_df.columns else np.arange(len(wide_df))
    y = pd.to_numeric(row_df["sensor_mean"], errors="coerce")
    trend = pd.to_numeric(row_df["rolling_mean"], errors="coerce")

    fig = plt.figure(figsize=(10, 4.3))
    plt.plot(x, y, linewidth=1.0, label="Cross-sensor mean")
    plt.plot(x, trend, linewidth=2.0, label="Rolling mean")
    plt.title("System level and rolling baseline")
    plt.xlabel("Timestamp" if time_col is not None else "Row")
    plt.ylabel("Cross-sensor mean")
    plt.legend()
    st.pyplot(fig, clear_figure=True)


def plot_sensor_spread(wide_df: pd.DataFrame, time_col: Optional[str], row_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    x = wide_df[time_col] if time_col is not None and time_col in wide_df.columns else np.arange(len(wide_df))
    sensor_count = max(1, wide_df.shape[1] - (1 if time_col is not None and time_col in wide_df.columns else 0))

    if sensor_count <= 1:
        y = pd.to_numeric(row_df["rolling_std"], errors="coerce")
        title = "Rolling volatility over time"
        ylabel = "Rolling std"
    else:
        y = pd.to_numeric(row_df["sensor_std"], errors="coerce")
        title = "Cross-sensor spread over time"
        ylabel = "Cross-sensor std"

    fig = plt.figure(figsize=(10, 4.3))
    plt.plot(x, y, linewidth=1.0)
    plt.title(title)
    plt.xlabel("Timestamp" if time_col is not None else "Row")
    plt.ylabel(ylabel)
    st.pyplot(fig, clear_figure=True)


def plot_score_distribution(scores: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    s = pd.Series(scores).replace([np.inf, -np.inf], np.nan).dropna()
    fig = plt.figure(figsize=(10, 4.3))
    plt.hist(s, bins=35)
    plt.title("Anomaly score distribution")
    plt.xlabel("Isolation Forest anomaly score")
    plt.ylabel("Count")
    st.pyplot(fig, clear_figure=True)


def plot_hour_concentration(wide_df: pd.DataFrame, time_col: Optional[str], flags: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    if time_col is None or time_col not in wide_df.columns:
        st.write("Time-based concentration is not available because no timestamp column could be confirmed.")
        return
    t = pd.to_datetime(wide_df[time_col], errors="coerce")
    flagged = np.asarray(flags).astype(bool)
    if flagged.sum() == 0:
        st.write("No anomalies were flagged with the current settings.")
        return
    hours = t.dt.hour[flagged].value_counts().sort_index()
    fig = plt.figure(figsize=(10, 4.3))
    plt.bar(hours.index.astype(int), hours.values)
    plt.title("When flagged events occur")
    plt.xlabel("Hour of day")
    plt.ylabel("Flagged timestamps")
    st.pyplot(fig, clear_figure=True)


def plot_top_sensor_volatility(vol_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    top = vol_df.head(10).sort_values("Std", ascending=True)
    fig = plt.figure(figsize=(10, 4.3))
    plt.barh(top["Sensor"], top["Std"])
    plt.title("Most volatile sensors")
    plt.xlabel("Standard deviation")
    plt.ylabel("Sensor")
    st.pyplot(fig, clear_figure=True)


def plot_threshold_tradeoff(trade_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if trade_df.empty:
        st.write("Threshold tradeoff is not available.")
        return
    fig = plt.figure(figsize=(10, 4.3))
    plt.plot(trade_df["Percentile"], trade_df["Flagged points"], marker="o")
    plt.title("Operational threshold tradeoff")
    plt.xlabel("Score percentile cut-off")
    plt.ylabel("Flagged timestamps")
    st.pyplot(fig, clear_figure=True)


def plot_driver_chart(contrib_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if contrib_df.empty:
        st.write("Sensor driver chart is not available for the selected event.")
        return
    top = contrib_df.head(10).sort_values("Abs robust z", ascending=True)
    fig = plt.figure(figsize=(10, 4.3))
    plt.barh(top["Sensor"], top["Abs robust z"])
    plt.title("Sensors contributing most to the selected event")
    plt.xlabel("Absolute robust z-score")
    plt.ylabel("Sensor")
    st.pyplot(fig, clear_figure=True)


# =========================
# Pages
# =========================
def build_model_bundle(bundle: Dict[str, object], contamination: float, random_state: int) -> Dict[str, object]:
    wide_df = bundle["wide_df"]
    time_col = bundle["time_col"]
    sensor_df, row_df, aligned_df = build_row_level_features(wide_df, time_col, bundle["sensor_cols"])

    # Use both sensor matrix and compact row features for richer detection.
    row_feature_cols = [
        c for c in ["sensor_std", "sensor_range", "max_abs_sensor_z", "sensor_count_over_3z", "rolling_delta"]
        if c in row_df.columns
    ]
    feature_df = pd.concat(
        [
            sensor_df.add_prefix("sensor__"),
            row_df[row_feature_cols].copy(),
        ],
        axis=1,
    )
    feature_df = feature_df.loc[:, feature_df.notna().sum() > 0].copy()

    pipe = fit_isolation_forest(feature_df, contamination=contamination, random_state=random_state)
    flags, scores = run_isolation_forest(pipe, feature_df)
    story = compute_story(wide_df, time_col, sensor_df, row_df, flags, scores)
    trade = threshold_tradeoff(scores)
    stability = score_stability(scores, row_df)

    return {
        "wide_df": wide_df,
        "time_col": time_col,
        "sensor_df": sensor_df,
        "row_df": row_df,
        "aligned_df": aligned_df,
        "feature_df": feature_df,
        "flags": flags,
        "scores": scores,
        "story": story,
        "trade": trade,
        "stability": stability,
    }


def executive_summary(meta: DataMeta, bundle: Dict[str, object], model_bundle: Dict[str, object]) -> None:
    story = model_bundle["story"]
    wide_df = model_bundle["wide_df"]
    time_col = model_bundle["time_col"]
    row_df = model_bundle["row_df"]
    flags = model_bundle["flags"]
    scores = model_bundle["scores"]
    vol_df = top_sensor_volatility(model_bundle["sensor_df"])

    hero(
        "Industrial Sensor Anomaly Detection Dashboard",
        "This dashboard turns raw industrial sensor readings into an operational monitoring view. "
        "It prioritises the current health signal, where abnormal behaviour is clustering, and which sensors are most likely to be driving unusual events."
    )
    st.caption(meta.source_label)

    st.markdown(
        (
            '<div class="kpi-grid">'
            f'{kpi_card("Timestamps analysed", f"{story["rows"]:,}", "Cleaned wide-format monitoring rows")}'
            f'{kpi_card("Sensors tracked", f"{story["sensors"]:,}", "Numeric sensors retained for analysis")}'
            f'{kpi_card("Flagged anomalies", f"{story["anomaly_count"]:,}", f"{story["anomaly_share"]:.2%} of monitored rows")}'
            f'{kpi_card("Largest system jump", fmt_num(story["jump_value"]), story["jump_when"])}'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    hour_text = ""
    if story["hot_hour"] is not None:
        hour_text = f"Flagged events cluster most around <b>{int(story['hot_hour']):02d}:00</b>"
        if story["hot_day"] is not None:
            hour_text += f" and are most common on <b>{story['hot_day']}</b>."
        else:
            hour_text += "."

    signal_scope = "sensor" if story["sensors"] == 1 else "cross-sensor system"
    main_signal_title = "Sensor signal with flagged events" if story["sensors"] == 1 else "Cross-sensor signal with flagged events"

    card(
        "Operational readout",
        (
            f"The {signal_scope} operating band sits mostly between <b>{fmt_num(story['p05'])}</b> and <b>{fmt_num(story['p95'])}</b>, "
            f"with a median level of <b>{fmt_num(story['median'])}</b>.<br>"
            f"The latest {story['recent_n']} rows are <b>{fmt_pct(story['level_change'])}</b> versus the earlier baseline, "
            f"while volatility is <b>{fmt_pct(story['vol_change'])}</b>.<br>"
            f"The strongest model alert appears at <b>{story['top_when']}</b> with score <b>{fmt_num(story['top_score'])}</b>. "
            f"{hour_text}"
        ),
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="card"><h3>Main health signal</h3>', unsafe_allow_html=True)
        plot_signal_with_flags(wide_df, time_col, row_df, flags, main_signal_title)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card"><h3>Most volatile sensors</h3>', unsafe_allow_html=True)
        plot_top_sensor_volatility(vol_df)
        st.markdown("</div>", unsafe_allow_html=True)

    left2, right2 = st.columns(2, gap="large")
    with left2:
        st.markdown('<div class="card"><h3>When alerts are clustering</h3>', unsafe_allow_html=True)
        plot_hour_concentration(wide_df, time_col, flags)
        st.markdown("</div>", unsafe_allow_html=True)
    with right2:
        st.markdown('<div class="card"><h3>Score distribution</h3>', unsafe_allow_html=True)
        plot_score_distribution(scores)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    top_events = wide_df.copy()
    top_events["anomaly_score"] = scores
    top_events["flagged"] = flags
    display_cols = [c for c in [time_col] if c is not None] + ["anomaly_score", "flagged"]
    top_events = top_events.loc[top_events["flagged"] == 1, display_cols].sort_values("anomaly_score", ascending=False).head(10)
    st.markdown('<div class="card"><h3>Highest-priority events</h3>', unsafe_allow_html=True)
    if top_events.empty:
        st.write("No events were flagged under the current configuration.")
    else:
        st.dataframe(top_events, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def data_story_page(bundle: Dict[str, object], model_bundle: Dict[str, object]) -> None:
    wide_df = model_bundle["wide_df"]
    time_col = model_bundle["time_col"]
    row_df = model_bundle["row_df"]
    vol_df = top_sensor_volatility(model_bundle["sensor_df"])
    story = model_bundle["story"]

    hero(
        "Data story",
        "This view explains how the system behaves before focusing on the model. "
        "It shows the typical operating range, where level shifts occur, and which sensors are naturally the most unstable."
    )

    st.markdown(
        (
            '<div class="kpi-grid">'
            f'{kpi_card("Typical operating band", f"{fmt_num(story["p05"])} to {fmt_num(story["p95"])}", ("Sensor range, 5th to 95th percentile" if story["sensors"] == 1 else "Cross-sensor mean, 5th to 95th percentile"))}'
            f'{kpi_card("Recent level change", fmt_pct(story["level_change"]), f"Last {story["recent_n"]} rows versus earlier baseline")}'
            f'{kpi_card("Recent volatility change", fmt_pct(story["vol_change"]), ("Rolling volatility shift" if story["sensors"] == 1 else "Cross-sensor spread shift"))}'
            f'{kpi_card("Most volatile sensor", vol_df.iloc[0]["Sensor"] if not vol_df.empty else "Not available", fmt_num(vol_df.iloc[0]["Std"]) if not vol_df.empty else "Not available")}'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    card(
        "What the data is showing",
        (
            f"The monitored signal experiences its sharpest rise of <b>{fmt_num(story['rise_value'])}</b> at <b>{story['rise_when']}</b> "
            f"and its sharpest drop of <b>{fmt_num(story['drop_value'])}</b> at <b>{story['drop_when']}</b>.<br>"
            f"The current operating level is <b>{fmt_pct(story['level_change'])}</b> relative to the earlier baseline, "
            f"which helps separate gradual drift from isolated spikes."
        ),
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="card"><h3>System level and baseline</h3>', unsafe_allow_html=True)
        plot_signal_and_trend(wide_df, time_col, row_df)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card"><h3>Cross-sensor spread</h3>', unsafe_allow_html=True)
        plot_sensor_spread(wide_df, time_col, row_df)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    left2, right2 = st.columns(2, gap="large")
    with left2:
        st.markdown('<div class="card"><h3>Sensor volatility ranking</h3>', unsafe_allow_html=True)
        st.dataframe(vol_df.head(12), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right2:
        st.markdown('<div class="card"><h3>Distribution of system level</h3>', unsafe_allow_html=True)
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 4.3))
        plt.hist(pd.to_numeric(row_df["sensor_mean"], errors="coerce").dropna(), bins=40)
        plt.title("Cross-sensor mean distribution")
        plt.xlabel("Cross-sensor mean")
        plt.ylabel("Count")
        st.pyplot(fig, clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)


def detection_lab_page(bundle: Dict[str, object]) -> None:
    wide_df = bundle["wide_df"]
    time_col = bundle["time_col"]
    sensor_cols = bundle["sensor_cols"]

    hero(
        "Detection lab",
        "This view turns anomaly detection into an operational decision tool. "
        "It lets you control expected alert volume, inspect stability against a robust statistical reference, and choose a practical score cut-off."
    )

    st.markdown('<div class="card"><h3>Model controls</h3>', unsafe_allow_html=True)
    contamination = st.slider("Expected anomaly share during model fitting", 0.005, 0.10, 0.03, 0.005)
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
    percentile_cut = st.slider("Operational score percentile for alerting", 0.80, 0.99, 0.95, 0.01)
    st.markdown("</div>", unsafe_allow_html=True)

    sensor_df, row_df, aligned_df = build_row_level_features(wide_df, time_col, sensor_cols)
    row_feature_cols = [
        c for c in ["sensor_std", "sensor_range", "max_abs_sensor_z", "sensor_count_over_3z", "rolling_delta"]
        if c in row_df.columns
    ]
    feature_df = pd.concat(
        [
            sensor_df.add_prefix("sensor__"),
            row_df[row_feature_cols].copy(),
        ],
        axis=1,
    )
    feature_df = feature_df.loc[:, feature_df.notna().sum() > 0].copy()
    pipe = fit_isolation_forest(feature_df, contamination=float(contamination), random_state=int(random_state))
    fit_flags, scores = run_isolation_forest(pipe, feature_df)
    op_flags, op_threshold = flag_from_percentile(scores, percentile_cut)
    trade_df = threshold_tradeoff(scores)
    stability = score_stability(scores, row_df)
    story = compute_story(wide_df, time_col, sensor_df, row_df, op_flags, scores)

    st.markdown(
        (
            '<div class="kpi-grid">'
            f'{kpi_card("Fit-time anomalies", f"{int(fit_flags.sum()):,}", f"{float(fit_flags.mean()):.2%} of monitored rows")}'
            f'{kpi_card("Operational alerts", f"{int(op_flags.sum()):,}", f"Percentile cut-off {percentile_cut:.0%}")}'
            f'{kpi_card("Score stability", fmt_num(stability["score_correlation"]), "Correlation with robust statistical reference")}'
            f'{kpi_card("Top 10% overlap", fmt_pct(100 * stability["top10_overlap"]) if np.isfinite(stability["top10_overlap"]) else "Not available", "Agreement with robust reference")}'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    card(
        "Model evaluation",
        (
            f"This project is unsupervised, so there is no labelled fault target for accuracy-style evaluation. "
            f"Instead, model quality is checked through score stability and overlap with a robust cross-sensor deviation reference.<br>"
            f"The current operational score threshold is <b>{fmt_num(op_threshold, 4)}</b>, which flags <b>{int(op_flags.sum()):,}</b> rows."
        ),
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="card"><h3>Signal with operational alerts</h3>', unsafe_allow_html=True)
        plot_signal_with_flags(wide_df, time_col, row_df, op_flags, "Cross-sensor signal with operational alerts")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card"><h3>Threshold tradeoff</h3>', unsafe_allow_html=True)
        plot_threshold_tradeoff(trade_df)
        st.markdown("</div>", unsafe_allow_html=True)

    left2, right2 = st.columns(2, gap="large")
    with left2:
        st.markdown('<div class="card"><h3>Alert score distribution</h3>', unsafe_allow_html=True)
        plot_score_distribution(scores)
        st.markdown("</div>", unsafe_allow_html=True)
    with right2:
        st.markdown('<div class="card"><h3>When operational alerts occur</h3>', unsafe_allow_html=True)
        plot_hour_concentration(wide_df, time_col, op_flags)
        st.markdown("</div>", unsafe_allow_html=True)

    top_idx = story["top_idx"]
    contrib_df = top_event_sensor_contributions(sensor_df, top_idx) if top_idx is not None else pd.DataFrame()
    st.write("")
    left3, right3 = st.columns(2, gap="large")
    with left3:
        st.markdown('<div class="card"><h3>Drivers of the strongest event</h3>', unsafe_allow_html=True)
        plot_driver_chart(contrib_df)
        st.markdown("</div>", unsafe_allow_html=True)
    with right3:
        st.markdown('<div class="card"><h3>Driver table for the strongest event</h3>', unsafe_allow_html=True)
        if contrib_df.empty:
            st.write("Driver details are not available.")
        else:
            st.dataframe(contrib_df.head(12), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


def insights_page(bundle: Dict[str, object], model_bundle: Dict[str, object]) -> None:
    story = model_bundle["story"]
    stability = model_bundle["stability"]
    sensor_df = model_bundle["sensor_df"]
    contrib_df = top_event_sensor_contributions(sensor_df, story["top_idx"]) if story["top_idx"] is not None else pd.DataFrame()

    hero(
        "Insights",
        "This page summarises the monitoring story in decision-ready language. "
        "The first section focuses on what the data is doing. The second section explains what the model adds."
    )

    data_text = (
        f"The monitored signal operates mostly inside a {"sensor-level" if story["sensors"] == 1 else "cross-sensor mean"} band of <b>{fmt_num(story['p05'])}</b> to <b>{fmt_num(story['p95'])}</b>, "
        f"with a median level of <b>{fmt_num(story['median'])}</b>.<br>"
        f"The latest segment is <b>{fmt_pct(story['level_change'])}</b> versus the earlier baseline, while volatility is "
        f"<b>{fmt_pct(story['vol_change'])}</b>, which helps separate drift from noise.<br>"
        f"The sharpest positive system move occurs at <b>{story['rise_when']}</b> and the sharpest negative move occurs at "
        f"<b>{story['drop_when']}</b>."
    )

    model_text = (
        f"The Isolation Forest configuration flags <b>{story['anomaly_count']:,}</b> rows, or <b>{story['anomaly_share']:.2%}</b> of monitored timestamps.<br>"
        f"The strongest alert appears at <b>{story['top_when']}</b> with score <b>{fmt_num(story['top_score'])}</b>. "
        f"The longest uninterrupted alert run lasts <b>{story['run_len']}</b> rows.<br>"
        f"Model stability versus a robust statistical reference is <b>{fmt_num(stability['score_correlation'])}</b> by score correlation, "
        f"with top-decile overlap of <b>{fmt_num(100 * stability['top10_overlap']) if np.isfinite(stability['top10_overlap']) else 'Not available'}%</b>."
    )

    st.markdown('<div class="insight-box"><h3>1. Data-driven insights</h3><p>' + data_text + '</p></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="insight-box"><h3>2. Model-driven insights</h3><p>' + model_text + '</p></div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="card"><h3>Top anomaly drivers</h3>', unsafe_allow_html=True)
        plot_driver_chart(contrib_df)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card"><h3>Supporting driver table</h3>', unsafe_allow_html=True)
        if contrib_df.empty:
            st.write("Driver details are not available.")
        else:
            st.dataframe(contrib_df.head(12), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


def appendix_page(meta: DataMeta, df_raw: pd.DataFrame, bundle: Dict[str, object], model_bundle: Dict[str, object]) -> None:
    hero(
        "Appendix",
        "This section keeps raw previews, schema checks, sensor coverage, and technical monitoring tables out of the main story while keeping them available for validation."
    )

    long_df = bundle["long_df"]
    wide_df = bundle["wide_df"]
    sensor_df = model_bundle["sensor_df"]
    row_df = model_bundle["row_df"]
    scores = model_bundle["scores"]
    flags = model_bundle["flags"]
    time_col = bundle["time_col"]

    st.markdown(
        (
            '<div class="kpi-grid">'
            f'{kpi_card("Raw rows", f"{meta.raw_rows:,}", "Before long-to-wide preparation")}'
            f'{kpi_card("Raw columns", f"{meta.raw_cols:,}", "Original upload shape")}'
            f'{kpi_card("Prepared timestamps", f"{len(wide_df):,}", "Rows retained after cleaning and pivoting")}'
            f'{kpi_card("Prepared sensors", f"{sensor_df.shape[1]:,}", "Numeric sensor columns used")}'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card"><h3>Schema overview</h3>', unsafe_allow_html=True)
    schema = pd.DataFrame({
        "Column": wide_df.columns,
        "Dtype": [str(wide_df[c].dtype) for c in wide_df.columns],
        "Missing share": [float(wide_df[c].isna().mean()) for c in wide_df.columns],
    })
    st.dataframe(schema, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="card"><h3>Raw data preview</h3>', unsafe_allow_html=True)
        st.dataframe(df_raw.head(30), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card"><h3>Prepared wide data preview</h3>', unsafe_allow_html=True)
        st.dataframe(wide_df.head(30), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    top_events = wide_df.copy()
    top_events["anomaly_score"] = scores
    top_events["flagged"] = flags
    if "sensor_mean" in row_df.columns:
        top_events["cross_sensor_mean"] = row_df["sensor_mean"].values
    display_cols = [c for c in [time_col] if c is not None] + ["cross_sensor_mean", "anomaly_score", "flagged"]
    display_cols = [c for c in display_cols if c in top_events.columns]
    st.markdown('<div class="card"><h3>Technical event table</h3>', unsafe_allow_html=True)
    st.dataframe(
        top_events.sort_values("anomaly_score", ascending=False).head(25)[display_cols],
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    sensor_health = pd.DataFrame({
        "Sensor": sensor_df.columns,
        "Missing share": sensor_df.isna().mean().values,
        "Median": sensor_df.median().values,
        "Std": sensor_df.std().values,
    }).sort_values("Missing share", ascending=False)
    st.markdown('<div class="card"><h3>Sensor coverage and stability</h3>', unsafe_allow_html=True)
    st.dataframe(sensor_health.head(25), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# App
# =========================
def main() -> None:
    st.set_page_config(
        page_title="Industrial Sensor Anomaly Detection Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Executive Summary", "Data Story", "Detection Lab", "Insights", "Appendix"],
        index=0,
    )

    st.sidebar.write("")
    st.sidebar.subheader("Data loading")
    use_upload = st.sidebar.toggle("Use uploaded file instead of default", value=False)
    uploaded = st.sidebar.file_uploader(
        "Upload a dataset file",
        type=["csv"],
        help="CSV only. The app supports semicolon-packed sensor rows, long sensor tables, and prepared wide sensor tables.",
    )

    try:
        df_raw, meta = load_dataset(use_upload, uploaded)
    except Exception as e:
        st.error("Data could not be loaded.")
        st.write(str(e))
        return

    bundle = prepare_sensor_data(df_raw)
    if len(bundle["sensor_cols"]) < 1 or len(bundle["wide_df"]) < 20:
        st.error("The dataset does not contain enough prepared sensor data to build the dashboard reliably.")
        return

    model_bundle = build_model_bundle(bundle, contamination=0.03, random_state=42)

    if page == "Executive Summary":
        executive_summary(meta, bundle, model_bundle)
    elif page == "Data Story":
        data_story_page(bundle, model_bundle)
    elif page == "Detection Lab":
        detection_lab_page(bundle)
    elif page == "Insights":
        insights_page(bundle, model_bundle)
    elif page == "Appendix":
        appendix_page(meta, df_raw, bundle, model_bundle)


if __name__ == "__main__":
    main()
