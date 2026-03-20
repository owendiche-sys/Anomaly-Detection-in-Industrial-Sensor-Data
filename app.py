from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_DATA_FILE = "data.csv"


# ----------------------------
# UI
# ----------------------------
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

          .card {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 14px;
            padding: 16px 16px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
          }

          .card h3, .card h4 {
            margin: 0 0 8px 0;
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
          }

          .kpi {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 14px;
            padding: 14px 14px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
          }

          .kpi .label {
            color: rgba(17, 24, 39, 0.70);
            font-size: 0.85rem;
            margin-bottom: 6px;
          }

          .kpi .value {
            font-size: 1.35rem;
            font-weight: 700;
            color: #111827;
            line-height: 1.1;
          }

          .kpi .sub {
            margin-top: 6px;
            color: rgba(17, 24, 39, 0.65);
            font-size: 0.85rem;
          }

          @media (max-width: 1100px) {
            .kpi-grid {
              grid-template-columns: repeat(2, minmax(0, 1fr));
            }
          }

          @media (max-width: 600px) {
            .kpi-grid {
              grid-template-columns: repeat(1, minmax(0, 1fr));
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f"""
      <div class="kpi">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {sub_html}
      </div>
    """


def card(title: str, body_md: str) -> None:
    st.markdown(
        f"""
        <div class="card">
          <h3>{title}</h3>
          <div class="muted">{body_md}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# Robust loading
# ----------------------------
@dataclass
class DataMeta:
    source_label: str
    rows: int
    cols: int


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

    try:
        df = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
    except Exception:
        df = None
        for s in [";", ",", "\t", "|"]:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), sep=s)
                break
            except Exception:
                continue

        if df is None:
            df = pd.read_csv(io.BytesIO(file_bytes))

    if df.shape[1] == 1:
        col0 = df.columns[0]
        sample_val = str(df.iloc[0, 0]) if len(df) else ""
        if ";" in col0 or ";" in sample_val:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), sep=";")
            except Exception:
                pass

    return df


@st.cache_data(show_spinner=False)
def load_dataset(use_uploaded: bool, uploaded_file) -> Tuple[pd.DataFrame, DataMeta]:
    if use_uploaded and uploaded_file is not None:
        raw = uploaded_file.read()
        df = safe_read_csv(raw)
        return df, DataMeta(
            source_label=f"Uploaded file: {uploaded_file.name}",
            rows=int(df.shape[0]),
            cols=int(df.shape[1]),
        )

    if not os.path.exists(DEFAULT_DATA_FILE):
        raise FileNotFoundError(
            f"Could not find {DEFAULT_DATA_FILE} next to app.py. "
            "Place the file in the same folder as app.py, or use the upload option in the sidebar."
        )

    with open(DEFAULT_DATA_FILE, "rb") as f:
        raw = f.read()

    df = safe_read_csv(raw)
    return df, DataMeta(
        source_label=f"Default file: {DEFAULT_DATA_FILE}",
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
    )


# ----------------------------
# Schema helpers
# ----------------------------
def detect_time_col(df: pd.DataFrame) -> Optional[str]:
    preferred = ["timestamp", "time", "date", "datetime"]

    for c in df.columns:
        c_lower = c.lower()
        if c_lower in preferred or any(p in c_lower for p in preferred):
            return c

    for c in df.columns:
        if df[c].dtype == "object":
            parsed = pd.to_datetime(df[c], errors="coerce", utc=False)
            if parsed.notna().mean() > 0.6:
                return c

    return None


def detect_value_col(df: pd.DataFrame, time_col: Optional[str]) -> Optional[str]:
    preferred = ["value", "measurement", "reading", "metric"]

    for c in df.columns:
        if c.lower() in preferred:
            return c

    candidates = [c for c in df.columns if c != time_col]
    numeric_candidates = []

    for c in candidates:
        s = pd.to_numeric(df[c], errors="coerce")
        valid_share = s.notna().mean()
        if valid_share > 0.6:
            numeric_candidates.append((c, valid_share))

    if not numeric_candidates:
        return None

    numeric_candidates.sort(key=lambda x: x[1], reverse=True)
    return numeric_candidates[0][0]


def standardize_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    time_col = detect_time_col(df)

    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    return df


# ----------------------------
# Feature engineering
# ----------------------------
@st.cache_data(show_spinner=False)
def build_features(
    df: pd.DataFrame,
    time_col: Optional[str],
    value_col: str,
    rolling_window: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()

    if time_col is not None:
        d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
        d = d.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

        d["hour"] = d[time_col].dt.hour.astype("Int64")
        d["day_of_week"] = d[time_col].dt.dayofweek.astype("Int64")
        d["month"] = d[time_col].dt.month.astype("Int64")
    else:
        d = d.reset_index(drop=True)

    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[value_col]).reset_index(drop=True)

    w = max(5, int(rolling_window))
    d["value_lag1"] = d[value_col].shift(1)
    d["delta_1"] = d[value_col] - d["value_lag1"]
    d["roll_mean"] = d[value_col].rolling(window=w, min_periods=max(3, w // 3)).mean()
    d["roll_std"] = d[value_col].rolling(window=w, min_periods=max(3, w // 3)).std()

    for c in ["value_lag1", "delta_1", "roll_mean", "roll_std"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    feature_cols = [value_col, "value_lag1", "delta_1", "roll_mean", "roll_std"]
    if time_col is not None:
        feature_cols += ["hour", "day_of_week", "month"]

    X = d[feature_cols].copy()
    return X, d


# ----------------------------
# Methods
# ----------------------------
def robust_rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    w = max(5, int(window))
    x = series.astype(float)

    med = x.rolling(w, min_periods=max(3, w // 3)).median()
    mad = (x - med).abs().rolling(w, min_periods=max(3, w // 3)).median()

    scale = 1.4826 * mad
    z = (x - med) / scale.replace(0, np.nan)
    return z


@st.cache_resource(show_spinner=False)
def fit_isolation_forest(
    X: pd.DataFrame,
    contamination: float,
    random_state: int = 42,
) -> Pipeline:
    Xc = X.copy()

    cat_cols = [
        c for c in Xc.columns
        if Xc[c].dtype == "object" or str(Xc[c].dtype).startswith("category")
    ]
    num_cols = [c for c in Xc.columns if c not in cat_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )

    model = IsolationForest(
        n_estimators=300,
        contamination=float(np.clip(contamination, 0.001, 0.30)),
        random_state=random_state,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(Xc)
    return pipe


def run_isolation_forest(pipe: Pipeline, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    pred = pipe.predict(X)
    normal_score = pipe.decision_function(X)
    anomaly_score = -normal_score
    anomaly_flag = (pred == -1).astype(int)
    return anomaly_flag, anomaly_score


def compute_threshold_tradeoff(
    scores: np.ndarray,
    higher_is_more_anomalous: bool = True,
) -> pd.DataFrame:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]

    if s.size == 0:
        return pd.DataFrame(columns=["Threshold", "Anomalies", "Share"])

    qs = np.linspace(0.50, 0.99, 11)
    thrs = np.quantile(s, qs)

    rows = []
    for t in thrs:
        if higher_is_more_anomalous:
            cnt = int((scores >= t).sum())
        else:
            cnt = int((scores <= t).sum())

        rows.append([float(t), cnt, float(cnt / len(scores))])

    return pd.DataFrame(rows, columns=["Threshold", "Anomalies", "Share"]).sort_values("Threshold")


def separation_table(X: pd.DataFrame, flags: np.ndarray) -> pd.DataFrame:
    d = X.copy()
    d = d.apply(pd.to_numeric, errors="coerce")
    flags = np.asarray(flags).astype(int)

    if d.shape[0] != flags.shape[0]:
        return pd.DataFrame(columns=["Feature", "Separation score", "Anomaly mean", "Normal mean"])

    anom = d[flags == 1]
    norm = d[flags == 0]

    if len(anom) < 5 or len(norm) < 5:
        return pd.DataFrame(columns=["Feature", "Separation score", "Anomaly mean", "Normal mean"])

    rows = []
    for c in d.columns:
        a = anom[c].dropna()
        n = norm[c].dropna()

        if len(a) < 5 or len(n) < 5:
            continue

        n_std = float(n.std()) if float(n.std()) > 0 else np.nan
        score = float(abs(a.mean() - n.mean()) / n_std) if np.isfinite(n_std) else np.nan
        rows.append([c, score, float(a.mean()), float(n.mean())])

    out = pd.DataFrame(rows, columns=["Feature", "Separation score", "Anomaly mean", "Normal mean"])
    return out.sort_values("Separation score", ascending=False)


# ----------------------------
# Insight helpers
# ----------------------------
def pct_change(current: float, baseline: float) -> float:
    if not np.isfinite(current) or not np.isfinite(baseline) or abs(baseline) < 1e-12:
        return np.nan
    return 100.0 * (current - baseline) / abs(baseline)


def fmt_num(x: float, digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "Not available"
    return f"{float(x):.{digits}f}"


def fmt_pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "Not available"
    return f"{float(x):+.1f}%"


def point_label(df: pd.DataFrame, idx: Optional[int], time_col: Optional[str]) -> str:
    if idx is None or idx < 0 or idx >= len(df):
        return "Not available"

    if time_col is not None and time_col in df.columns:
        ts = pd.to_datetime(df.iloc[idx][time_col], errors="coerce")
        if pd.notna(ts):
            return ts.strftime("%Y-%m-%d %H:%M")

    return f"Index {idx}"


def longest_anomaly_run(flags: np.ndarray) -> Tuple[int, Optional[int], Optional[int]]:
    arr = np.asarray(flags).astype(int)
    best_len = 0
    best_end = None
    cur_len = 0

    for i, v in enumerate(arr):
        if v == 1:
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


def build_snapshot(
    aligned: pd.DataFrame,
    time_col: Optional[str],
    value_col: str,
    flags: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, object]:
    d = aligned.copy()
    v = pd.to_numeric(d[value_col], errors="coerce").astype(float)
    n = len(v)

    recent_n = max(10, min(96, max(10, n // 10)))
    if n <= recent_n:
        recent_n = max(5, n // 3) if n > 10 else max(1, n)

    recent = v.tail(recent_n)
    baseline = v.iloc[:-recent_n] if n > recent_n else v

    recent_mean = float(recent.mean()) if len(recent) else np.nan
    baseline_mean = float(baseline.mean()) if len(baseline) else np.nan
    recent_std = float(recent.std()) if len(recent) else np.nan
    baseline_std = float(baseline.std()) if len(baseline) else np.nan

    level_change = pct_change(recent_mean, baseline_mean)
    volatility_change = pct_change(recent_std, baseline_std)

    if "delta_1" in d.columns:
        delta = pd.to_numeric(d["delta_1"], errors="coerce").astype(float)
    else:
        delta = v.diff()

    jump_idx = int(delta.abs().idxmax()) if delta.notna().any() else None
    rise_idx = int(delta.idxmax()) if delta.notna().any() else None
    drop_idx = int(delta.idxmin()) if delta.notna().any() else None

    finite_scores = np.isfinite(scores)
    top_idx = int(np.nanargmax(scores)) if finite_scores.any() else None

    run_len, run_start, run_end = longest_anomaly_run(flags)

    hotspot_hour = None
    hotspot_day = None
    largest_gap_minutes = np.nan

    if time_col is not None and time_col in d.columns:
        t = pd.to_datetime(d[time_col], errors="coerce")

        if t.notna().any():
            gaps = t.diff().dt.total_seconds().div(60.0)
            if gaps.notna().any():
                largest_gap_minutes = float(gaps.max())

            flagged = np.asarray(flags).astype(bool)
            if flagged.any():
                hours = t.dt.hour[flagged]
                days = t.dt.day_name()[flagged]

                if hours.notna().any():
                    hotspot_hour = int(hours.value_counts().idxmax())
                if days.notna().any():
                    hotspot_day = str(days.value_counts().idxmax())

    return {
        "recent_n": recent_n,
        "anomaly_count": int(np.asarray(flags).sum()),
        "anomaly_share": float(np.asarray(flags).mean()) if len(flags) else 0.0,
        "recent_mean": recent_mean,
        "baseline_mean": baseline_mean,
        "recent_std": recent_std,
        "baseline_std": baseline_std,
        "level_change": level_change,
        "volatility_change": volatility_change,
        "jump_value": float(delta.abs().max()) if delta.notna().any() else np.nan,
        "jump_when": point_label(d, jump_idx, time_col),
        "rise_value": float(delta.max()) if delta.notna().any() else np.nan,
        "rise_when": point_label(d, rise_idx, time_col),
        "drop_value": float(delta.min()) if delta.notna().any() else np.nan,
        "drop_when": point_label(d, drop_idx, time_col),
        "top_score": float(np.nanmax(scores)) if finite_scores.any() else np.nan,
        "top_when": point_label(d, top_idx, time_col),
        "top_value": float(v.iloc[top_idx]) if top_idx is not None and top_idx < len(v) else np.nan,
        "run_len": run_len,
        "run_start": point_label(d, run_start, time_col) if run_start is not None else "Not available",
        "run_end": point_label(d, run_end, time_col) if run_end is not None else "Not available",
        "hotspot_hour": hotspot_hour,
        "hotspot_day": hotspot_day,
        "largest_gap_minutes": largest_gap_minutes,
        "p05": float(v.quantile(0.05)) if v.notna().any() else np.nan,
        "p95": float(v.quantile(0.95)) if v.notna().any() else np.nan,
        "median": float(v.median()) if v.notna().any() else np.nan,
    }


# ----------------------------
# Plotting
# ----------------------------
def plot_timeseries_with_anomalies(
    df_aligned: pd.DataFrame,
    time_col: Optional[str],
    value_col: str,
    flags: np.ndarray,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    d = df_aligned.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[value_col]).reset_index(drop=True)

    flags = np.asarray(flags).astype(int)
    if len(flags) != len(d):
        flags = np.zeros(len(d), dtype=int)

    x = d[time_col] if time_col is not None and time_col in d.columns else d.index
    y = d[value_col].astype(float)

    fig = plt.figure()
    plt.plot(x, y, linewidth=1.0)

    idx = np.where(flags == 1)[0]
    if idx.size > 0:
        if hasattr(x, "iloc"):
            plt.scatter(x.iloc[idx], y.iloc[idx], s=10)
        else:
            plt.scatter(x[idx], y.iloc[idx], s=10)

    plt.title(title)
    plt.xlabel("Time" if time_col is not None else "Index")
    plt.ylabel(value_col)
    st.pyplot(fig, clear_figure=True)


def plot_signal_with_trend(aligned: pd.DataFrame, time_col: Optional[str], value_col: str, window: int) -> None:
    import matplotlib.pyplot as plt

    d = aligned.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce").astype(float)
    d = d.dropna(subset=[value_col]).reset_index(drop=True)

    x = d[time_col] if time_col is not None and time_col in d.columns else d.index
    w = max(5, int(window))
    trend = d[value_col].rolling(window=w, min_periods=max(3, w // 3)).mean()

    fig = plt.figure()
    plt.plot(x, d[value_col], linewidth=0.9, label="Signal")
    plt.plot(x, trend, linewidth=2.0, label="Rolling mean")
    plt.title("Signal and rolling mean")
    plt.xlabel("Time" if time_col is not None else "Index")
    plt.ylabel(value_col)
    plt.legend()
    st.pyplot(fig, clear_figure=True)


def plot_rolling_volatility(aligned: pd.DataFrame, time_col: Optional[str], value_col: str, window: int) -> None:
    import matplotlib.pyplot as plt

    d = aligned.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce").astype(float)
    d = d.dropna(subset=[value_col]).reset_index(drop=True)

    x = d[time_col] if time_col is not None and time_col in d.columns else d.index
    w = max(5, int(window))
    vol = d[value_col].rolling(window=w, min_periods=max(3, w // 3)).std()

    fig = plt.figure()
    plt.plot(x, vol, linewidth=1.2)
    plt.title("Rolling volatility")
    plt.xlabel("Time" if time_col is not None else "Index")
    plt.ylabel("Rolling std")
    st.pyplot(fig, clear_figure=True)


def plot_score_distribution(scores: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    s = pd.Series(scores, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        st.write("Score distribution is not available.")
        return

    fig = plt.figure()
    plt.hist(s, bins=30)
    plt.title("Anomaly score distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    st.pyplot(fig, clear_figure=True)


def plot_anomalies_by_hour(aligned: pd.DataFrame, time_col: Optional[str], flags: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    if time_col is None or time_col not in aligned.columns:
        st.write("Time-based concentration is not available because no time column was detected.")
        return

    d = aligned.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d["flag"] = np.asarray(flags).astype(int)

    flagged = d.loc[d["flag"] == 1].copy()
    if len(flagged) == 0:
        st.write("No anomalies were flagged with the current settings.")
        return

    by_hour = flagged[time_col].dt.hour.value_counts().sort_index()

    fig = plt.figure()
    plt.bar(by_hour.index.astype(int), by_hour.values)
    plt.title("When anomalies occur")
    plt.xlabel("Hour of day")
    plt.ylabel("Flagged points")
    st.pyplot(fig, clear_figure=True)


def plot_threshold_tradeoff_chart(trade: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if trade.empty:
        st.write("Threshold tradeoff chart is not available.")
        return

    fig = plt.figure()
    plt.plot(trade["Threshold"], trade["Anomalies"], marker="o")
    plt.title("Threshold tradeoff")
    plt.xlabel("Threshold")
    plt.ylabel("Flagged anomalies")
    st.pyplot(fig, clear_figure=True)


def plot_feature_separation(sep: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if sep.empty:
        st.write("Feature driver chart is not available.")
        return

    top = sep.head(8).sort_values("Separation score", ascending=True)

    fig = plt.figure()
    plt.barh(top["Feature"], top["Separation score"])
    plt.title("Top anomaly drivers")
    plt.xlabel("Separation score")
    plt.ylabel("Feature")
    st.pyplot(fig, clear_figure=True)


# ----------------------------
# Pages
# ----------------------------
def page_summary(
    df_raw: pd.DataFrame,
    meta: DataMeta,
    time_col: Optional[str],
    value_col: Optional[str],
    show_diagnostics: bool = False,
) -> None:
    st.title("Anomaly Detection Dashboard")
    st.caption(meta.source_label)

    df = standardize_timeseries(df_raw)

    if value_col is None:
        card(
            "Data check",
            "No numeric value column could be detected. Upload a dataset with a numeric measurement column.",
        )
        return

    rolling_window = min(288, max(24, len(df) // 10))
    X, aligned = build_features(df, time_col=time_col, value_col=value_col, rolling_window=rolling_window)

    if len(X) < 50:
        st.warning("Not enough cleaned data to generate insight-first views.")
        return

    pipe = fit_isolation_forest(X, contamination=0.01, random_state=42)
    flags, scores = run_isolation_forest(pipe, X)
    snap = build_snapshot(aligned, time_col, value_col, flags, scores)

    level_text = fmt_pct(snap["level_change"])
    vol_text = fmt_pct(snap["volatility_change"])

    st.markdown(
        f"""
        <div class="kpi-grid">
          {kpi_card("Flagged anomalies", f"{snap['anomaly_count']:,}", f"{snap['anomaly_share']:.2%} of cleaned points")}
          {kpi_card("Recent level vs baseline", level_text, f"Last {snap['recent_n']} points")}
          {kpi_card("Recent volatility vs baseline", vol_text, "Rolling stability check")}
          {kpi_card("Largest single-step move", fmt_num(snap['jump_value']), snap['jump_when'])}
        </div>
        """,
        unsafe_allow_html=True,
    )

    hotspot_sentence = ""
    if snap["hotspot_hour"] is not None:
        hotspot_sentence = f"Flagged anomalies cluster most around {int(snap['hotspot_hour']):02d}:00"
        if snap["hotspot_day"] is not None:
            hotspot_sentence += f", with the strongest concentration on {snap['hotspot_day']}."
        else:
            hotspot_sentence += "."
    elif snap["hotspot_day"] is not None:
        hotspot_sentence = f"Flagged anomalies cluster most on {snap['hotspot_day']}."

    gap_sentence = ""
    if np.isfinite(snap["largest_gap_minutes"]):
        gap_sentence = f"The largest observed time gap is {snap['largest_gap_minutes']:.1f} minutes."

    card(
        "What the data is telling you",
        f"""
        Most values sit between <b>{fmt_num(snap['p05'])}</b> and <b>{fmt_num(snap['p95'])}</b>, with a median of <b>{fmt_num(snap['median'])}</b>.<br>
        Recent values are <b>{level_text}</b> versus the earlier baseline, while recent volatility is <b>{vol_text}</b>.<br>
        The biggest upward step is <b>{fmt_num(snap['rise_value'])}</b> at <b>{snap['rise_when']}</b>, and the biggest downward step is <b>{fmt_num(snap['drop_value'])}</b> at <b>{snap['drop_when']}</b>.<br>
        {hotspot_sentence} {gap_sentence}
        """.strip(),
    )

    st.write("")
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="card"><h3>Main signal</h3>', unsafe_allow_html=True)
        plot_timeseries_with_anomalies(aligned, time_col, value_col, flags, "Signal with flagged anomalies")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card"><h3>When anomalies happen</h3>', unsafe_allow_html=True)
        plot_anomalies_by_hour(aligned, time_col, flags)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><h3>Score distribution</h3>', unsafe_allow_html=True)
    plot_score_distribution(scores)
    st.markdown("</div>", unsafe_allow_html=True)

    if show_diagnostics:
        st.write("")
        st.markdown('<div class="card"><h3>Most severe anomaly events</h3>', unsafe_allow_html=True)
        out = aligned.copy()
        out["anomaly_flag"] = flags
        out["anomaly_score"] = scores
        cols = [c for c in [time_col, value_col] if c is not None] + ["anomaly_score"]
        severe = out.loc[out["anomaly_flag"] == 1, cols].sort_values("anomaly_score", ascending=False).head(20)
        st.dataframe(severe, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


def page_exploration(
    df_raw: pd.DataFrame,
    time_col: Optional[str],
    value_col: Optional[str],
    show_diagnostics: bool = False,
) -> None:
    st.title("Exploration")
    st.caption("Understand level, spread, jumps, and stability before detection.")

    df = standardize_timeseries(df_raw)

    if value_col is None:
        card("Data check", "No numeric value column could be detected.")
        return

    _, aligned = build_features(
        df,
        time_col=time_col,
        value_col=value_col,
        rolling_window=min(288, max(24, len(df) // 10)),
    )

    v = pd.to_numeric(aligned[value_col], errors="coerce").astype(float)
    if v.notna().sum() < 20:
        st.warning("Not enough valid values for exploratory analysis.")
        return

    q05 = float(v.quantile(0.05))
    q95 = float(v.quantile(0.95))
    tail_share = float(((v < q05) | (v > q95)).mean())

    delta = pd.to_numeric(aligned["delta_1"], errors="coerce").astype(float)
    rise_value = float(delta.max()) if delta.notna().any() else np.nan
    drop_value = float(delta.min()) if delta.notna().any() else np.nan
    rise_when = point_label(aligned, int(delta.idxmax()) if delta.notna().any() else None, time_col)
    drop_when = point_label(aligned, int(delta.idxmin()) if delta.notna().any() else None, time_col)

    st.markdown(
        f"""
        <div class="kpi-grid">
          {kpi_card("Typical operating band", f"{fmt_num(q05)} to {fmt_num(q95)}", "5th to 95th percentile")}
          {kpi_card("Tail share", f"{tail_share:.2%}", "Outside the central band")}
          {kpi_card("Largest upward step", fmt_num(rise_value), rise_when)}
          {kpi_card("Largest downward step", fmt_num(drop_value), drop_when)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    card(
        "Behaviour summary",
        f"""
        The series spends most of its time inside the <b>{fmt_num(q05)}</b> to <b>{fmt_num(q95)}</b> band.<br>
        About <b>{tail_share:.2%}</b> of points sit outside that central range, which tells you how heavy the tails are before any model is applied.<br>
        The sharpest upward move occurs at <b>{rise_when}</b>, while the sharpest drop occurs at <b>{drop_when}</b>.
        """.strip(),
    )

    st.write("")
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="card"><h3>Signal and trend</h3>', unsafe_allow_html=True)
        plot_signal_with_trend(aligned, time_col, value_col, window=min(96, max(12, len(aligned) // 20)))
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card"><h3>Rolling volatility</h3>', unsafe_allow_html=True)
        plot_rolling_volatility(aligned, time_col, value_col, window=min(96, max(12, len(aligned) // 20)))
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><h3>Value distribution</h3>', unsafe_allow_html=True)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.hist(v.dropna(), bins=40)
    plt.title("Value distribution")
    plt.xlabel(value_col)
    plt.ylabel("Count")
    st.pyplot(fig, clear_figure=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if show_diagnostics:
        st.write("")
        st.markdown('<div class="card"><h3>Supporting diagnostics</h3>', unsafe_allow_html=True)
        diag = pd.DataFrame(
            {
                "Statistic": ["Mean", "Median", "Std", "P5", "P95", "Min", "Max"],
                "Value": [
                    float(v.mean()),
                    float(v.median()),
                    float(v.std()),
                    q05,
                    q95,
                    float(v.min()),
                    float(v.max()),
                ],
            }
        )
        st.dataframe(diag, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


def page_detection(
    df_raw: pd.DataFrame,
    time_col: Optional[str],
    value_col: Optional[str],
    show_diagnostics: bool = False,
) -> None:
    st.title("Detection")
    st.caption("Run anomaly detection and review what changed, where, and how strongly.")

    df = standardize_timeseries(df_raw)

    if value_col is None:
        card("Data check", "No numeric value column could be detected.")
        return

    st.markdown('<div class="card"><h3>Controls</h3>', unsafe_allow_html=True)
    method = st.selectbox("Method", ["Isolation Forest", "Rolling robust z-score"], index=0)
    rolling_window = st.number_input("Rolling window (points)", min_value=20, max_value=2000, value=288, step=20)

    if method == "Isolation Forest":
        contamination = st.slider("Expected anomaly share (contamination)", 0.001, 0.10, 0.01, 0.001)
        random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
    else:
        z_threshold = st.slider("Z-score threshold", 2.0, 8.0, 4.0, 0.1)
    st.markdown("</div>", unsafe_allow_html=True)

    X, aligned = build_features(df, time_col=time_col, value_col=value_col, rolling_window=int(rolling_window))

    if len(X) < 50:
        st.warning("Not enough data after cleaning to run detection reliably.")
        return

    if method == "Isolation Forest":
        pipe = fit_isolation_forest(X, contamination=float(contamination), random_state=int(random_state))
        flags, scores = run_isolation_forest(pipe, X)
    else:
        s = pd.to_numeric(aligned[value_col], errors="coerce").astype(float)
        z = robust_rolling_zscore(s.ffill().bfill(), window=int(rolling_window))
        scores = z.abs().to_numpy()
        flags = (z.abs() >= float(z_threshold)).astype(int).to_numpy()

    snap = build_snapshot(aligned, time_col, value_col, flags, scores)

    st.markdown(
        f"""
        <div class="kpi-grid">
          {kpi_card("Flagged anomalies", f"{snap['anomaly_count']:,}", f"{snap['anomaly_share']:.2%} of points")}
          {kpi_card("Top anomaly score", fmt_num(snap['top_score']), snap['top_when'])}
          {kpi_card("Longest anomaly run", f"{snap['run_len']}", f"{snap['run_start']} to {snap['run_end']}")}
          {kpi_card("Detected extreme value", fmt_num(snap['top_value']), "Value at highest-scoring anomaly")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    card(
        "Detection summary",
        f"""
        The current configuration flags <b>{snap['anomaly_count']:,}</b> points, which is <b>{snap['anomaly_share']:.2%}</b> of the cleaned signal.<br>
        The strongest anomaly appears at <b>{snap['top_when']}</b> with a value of <b>{fmt_num(snap['top_value'])}</b> and a score of <b>{fmt_num(snap['top_score'])}</b>.<br>
        The longest uninterrupted anomaly run lasts <b>{snap['run_len']}</b> points.
        """.strip(),
    )

    st.write("")
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="card"><h3>Signal with detected anomalies</h3>', unsafe_allow_html=True)
        plot_timeseries_with_anomalies(aligned, time_col, value_col, flags, "Signal with detected anomalies")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card"><h3>Score distribution</h3>', unsafe_allow_html=True)
        plot_score_distribution(scores)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    left2, right2 = st.columns(2, gap="large")

    with left2:
        st.markdown('<div class="card"><h3>When anomalies occur</h3>', unsafe_allow_html=True)
        plot_anomalies_by_hour(aligned, time_col, flags)
        st.markdown("</div>", unsafe_allow_html=True)

    with right2:
        trade = compute_threshold_tradeoff(scores, higher_is_more_anomalous=True)
        st.markdown('<div class="card"><h3>Threshold tradeoff</h3>', unsafe_allow_html=True)
        plot_threshold_tradeoff_chart(trade)
        st.markdown("</div>", unsafe_allow_html=True)

    if show_diagnostics:
        st.write("")
        st.markdown('<div class="card"><h3>Severe flagged events</h3>', unsafe_allow_html=True)
        out = aligned.copy()
        out["anomaly_flag"] = flags
        out["anomaly_score"] = scores
        cols = [c for c in [time_col, value_col] if c is not None] + ["anomaly_score"]
        severe = out.loc[out["anomaly_flag"] == 1, cols].sort_values("anomaly_score", ascending=False).head(25)
        st.dataframe(severe, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


def page_insights(
    df_raw: pd.DataFrame,
    time_col: Optional[str],
    value_col: Optional[str],
    show_diagnostics: bool = False,
) -> None:
    st.title("Insights")
    st.caption("Decision-focused findings from both the data and the anomaly model.")

    df = standardize_timeseries(df_raw)

    if value_col is None:
        card("Data check", "No numeric value column could be detected.")
        return

    rolling_window = min(288, max(24, len(df) // 10))
    X, aligned = build_features(df, time_col=time_col, value_col=value_col, rolling_window=rolling_window)

    if len(X) < 50:
        st.warning("Not enough data after cleaning to generate reliable insights.")
        return

    pipe = fit_isolation_forest(X, contamination=0.01, random_state=42)
    flags, scores = run_isolation_forest(pipe, X)
    snap = build_snapshot(aligned, time_col, value_col, flags, scores)
    sep = separation_table(X, flags)

    data_driven_text = f"""
    The central operating range is <b>{fmt_num(snap['p05'])}</b> to <b>{fmt_num(snap['p95'])}</b>, which gives you a realistic view of normal behavior without being distorted by extremes.<br>
    The recent level is <b>{fmt_pct(snap['level_change'])}</b> versus the earlier baseline, so the signal is {'rising' if np.isfinite(snap['level_change']) and snap['level_change'] > 0 else 'falling or stable'} in its latest segment.<br>
    Recent volatility is <b>{fmt_pct(snap['volatility_change'])}</b> compared with the earlier period, which tells you whether the series is becoming noisier or calmer.<br>
    The sharpest positive move is <b>{fmt_num(snap['rise_value'])}</b> at <b>{snap['rise_when']}</b>, while the sharpest negative move is <b>{fmt_num(snap['drop_value'])}</b> at <b>{snap['drop_when']}</b>.
    """.strip()

    model_driven_text = f"""
    The model flags <b>{snap['anomaly_count']:,}</b> points, or <b>{snap['anomaly_share']:.2%}</b> of the cleaned signal.<br>
    The highest-scoring anomaly occurs at <b>{snap['top_when']}</b> with a value of <b>{fmt_num(snap['top_value'])}</b> and a score of <b>{fmt_num(snap['top_score'])}</b>.<br>
    The longest run of consecutive anomalies lasts <b>{snap['run_len']}</b> points, which helps distinguish isolated spikes from sustained abnormal periods.
    """.strip()

    st.markdown('<div class="card"><h3>1. Data-driven insights</h3>', unsafe_allow_html=True)
    st.markdown(data_driven_text, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><h3>2. Model-driven insights</h3>', unsafe_allow_html=True)
    st.markdown(model_driven_text, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="card"><h3>Top anomaly drivers</h3>', unsafe_allow_html=True)
        plot_feature_separation(sep)
        if not sep.empty:
            top = sep.iloc[0]
            st.markdown(
                f"""
                <div class="muted">
                  The strongest separating feature is <b>{top['Feature']}</b>, which means flagged points differ most clearly from normal points on this dimension.
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card"><h3>Anomaly concentration</h3>', unsafe_allow_html=True)
        plot_anomalies_by_hour(aligned, time_col, flags)
        st.markdown("</div>", unsafe_allow_html=True)

    if show_diagnostics and not sep.empty:
        st.write("")
        st.markdown('<div class="card"><h3>Supporting driver table</h3>', unsafe_allow_html=True)
        st.dataframe(sep.head(12), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# App
# ----------------------------
def main() -> None:
    st.set_page_config(
        page_title="Anomaly Detection Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Summary", "Exploration", "Detection", "Insights"], index=0)

    st.sidebar.write("")
    show_diagnostics = st.sidebar.toggle("Show supporting tables", value=False)

    st.sidebar.write("")
    st.sidebar.subheader("Data loading")
    use_upload = st.sidebar.toggle("Use uploaded file instead of default", value=False)
    uploaded = st.sidebar.file_uploader(
        "Upload a dataset file",
        type=["csv"],
        help="CSV only for this project. The loader will auto-detect common delimiters.",
    )

    try:
        df_raw, meta = load_dataset(use_upload, uploaded)
    except Exception as e:
        st.error("Data could not be loaded.")
        st.write(str(e))
        return

    time_col = detect_time_col(df_raw)
    value_col = detect_value_col(df_raw, time_col=time_col)

    if page == "Summary":
        page_summary(df_raw, meta, time_col, value_col, show_diagnostics=show_diagnostics)
    elif page == "Exploration":
        page_exploration(df_raw, time_col, value_col, show_diagnostics=show_diagnostics)
    elif page == "Detection":
        page_detection(df_raw, time_col, value_col, show_diagnostics=show_diagnostics)
    elif page == "Insights":
        page_insights(df_raw, time_col, value_col, show_diagnostics=show_diagnostics)


if __name__ == "__main__":
    main()