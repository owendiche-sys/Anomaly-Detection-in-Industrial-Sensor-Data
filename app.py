from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


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
            color: rgba(17, 24, 39, 0.70);
            font-size: 0.92rem;
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
            .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          }
          @media (max-width: 600px) {
            .kpi-grid { grid-template-columns: repeat(1, minmax(0, 1fr)); }
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
    # Use the first non-empty line (usually header) for delimiter detection
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

    # Try detected delimiter
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
    except Exception:
        # Fallback: try common separators
        for s in [";", ",", "\t", "|"]:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), sep=s)
                break
            except Exception:
                df = None
        if df is None:
            df = pd.read_csv(io.BytesIO(file_bytes))  # last attempt

    # If it still loaded as a single column that looks delimited, attempt split
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
# Schema helpers (time series oriented)
# ----------------------------
def detect_time_col(df: pd.DataFrame) -> Optional[str]:
    preferred = ["timestamp", "time", "date", "datetime"]
    for c in df.columns:
        if c.lower() in preferred or any(p in c.lower() for p in preferred):
            return c
    # fallback: find a column with many parseable datetimes
    for c in df.columns:
        if df[c].dtype == "object":
            parsed = pd.to_datetime(df[c], errors="coerce", utc=False)
            if parsed.notna().mean() > 0.6:
                return c
    return None


def detect_value_col(df: pd.DataFrame, time_col: Optional[str]) -> Optional[str]:
    # Prefer common names
    preferred = ["value", "measurement", "reading", "metric"]
    for c in df.columns:
        if c.lower() in preferred:
            return c
    # Otherwise choose the first numeric column not equal to time
    candidates = [c for c in df.columns if c != time_col]
    num = []
    for c in candidates:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.6:
            num.append((c, s.notna().mean()))
    if not num:
        return None
    num.sort(key=lambda x: x[1], reverse=True)
    return num[0][0]


def standardize_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    time_col = detect_time_col(df)
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col]).sort_values(time_col)
        df = df.reset_index(drop=True)
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
    """
    Returns:
    - X: model feature frame
    - base: aligned frame with time/value for plotting and reporting
    """
    d = df.copy()

    if time_col is not None:
        d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
        d = d.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

        # Time features (stand-alone)
        d["hour"] = d[time_col].dt.hour.astype("Int64")
        d["day_of_week"] = d[time_col].dt.dayofweek.astype("Int64")
        d["month"] = d[time_col].dt.month.astype("Int64")
    else:
        d = d.reset_index(drop=True)

    # Value
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[value_col]).reset_index(drop=True)

    # Rolling features (point-based to handle gaps)
    w = max(5, int(rolling_window))
    d["value_lag1"] = d[value_col].shift(1)
    d["delta_1"] = d[value_col] - d["value_lag1"]
    d["roll_mean"] = d[value_col].rolling(window=w, min_periods=max(3, w // 3)).mean()
    d["roll_std"] = d[value_col].rolling(window=w, min_periods=max(3, w // 3)).std()

    # Fill early NaNs safely
    for c in ["value_lag1", "delta_1", "roll_mean", "roll_std"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Feature set
    feature_cols = [value_col, "value_lag1", "delta_1", "roll_mean", "roll_std"]
    if time_col is not None:
        feature_cols += ["hour", "day_of_week", "month"]

    X = d[feature_cols].copy()
    base = d[[c for c in [time_col, value_col] if c is not None]].copy()
    if time_col is None:
        base["index"] = np.arange(len(d))
    return X, d


# ----------------------------
# Methods
# ----------------------------
def robust_rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Robust rolling z-score using rolling median and MAD.
    z = (x - median) / (1.4826 * MAD)
    """
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
    # Separate numeric and categorical for safety (time features are Int64, treat as numeric)
    Xc = X.copy()
    cat_cols = [c for c in Xc.columns if Xc[c].dtype == "object" or str(Xc[c].dtype).startswith("category")]
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


def run_isolation_forest(
    pipe: Pipeline,
    X: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    - anomaly_flag: 1 for anomaly, 0 for normal
    - anomaly_score: higher means more anomalous
    """
    # IsolationForest: predict -> -1 anomaly, 1 normal
    pred = pipe.predict(X)
    # decision_function: higher = more normal; invert for anomaly score
    normal_score = pipe.decision_function(X)
    anomaly_score = -normal_score
    anomaly_flag = (pred == -1).astype(int)
    return anomaly_flag, anomaly_score


def compute_threshold_tradeoff(scores: np.ndarray, higher_is_more_anomalous: bool = True) -> pd.DataFrame:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return pd.DataFrame(columns=["Threshold", "Anomalies", "Share"])

    # Use quantile-based thresholds for a stable tradeoff view
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
    """
    Simple model-driven explanation:
    standardized mean difference between anomaly vs normal for each feature.
    """
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
    out = out.sort_values("Separation score", ascending=False)
    return out


# ----------------------------
# Plotting
# ----------------------------
def plot_timeseries_with_anomalies(df_aligned: pd.DataFrame, time_col: Optional[str], value_col: str,
                                  flags: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    d = df_aligned.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[value_col]).reset_index(drop=True)

    flags = np.asarray(flags).astype(int)
    if len(flags) != len(d):
        flags = np.zeros(len(d), dtype=int)

    x = d[time_col] if time_col is not None else d.index
    y = d[value_col].astype(float)

    fig = plt.figure()
    plt.plot(x, y, linewidth=1.0)

    idx = np.where(flags == 1)[0]
    if idx.size > 0:
        plt.scatter(x.iloc[idx] if hasattr(x, "iloc") else x[idx], y.iloc[idx], s=10)

    plt.title(title)
    plt.xlabel("Time" if time_col is not None else "Index")
    plt.ylabel(value_col)
    st.pyplot(fig, clear_figure=True)


# ----------------------------
# Pages
# ----------------------------
def page_summary(df_raw: pd.DataFrame, meta: DataMeta, time_col: Optional[str], value_col: Optional[str]) -> None:
    st.title("Anomaly Detection Dashboard")
    st.caption(meta.source_label)

    df = standardize_timeseries(df_raw)

    if value_col is None:
        card(
            "Data check",
            "No numeric value column could be detected. Upload a dataset with a numeric measurement column.",
        )
        return

    # KPI cards (ONLY here)
    v = pd.to_numeric(df[value_col], errors="coerce")
    n = int(v.notna().sum())
    missing = int(v.isna().sum())

    start, end = None, None
    if time_col is not None and time_col in df.columns:
        t = pd.to_datetime(df[time_col], errors="coerce")
        if t.notna().any():
            start = t.min()
            end = t.max()

    # Default anomaly count using a lightweight robust z-score
    z = robust_rolling_zscore(v.fillna(method="ffill").fillna(method="bfill"), window=288)
    default_flags = (z.abs() >= 4.0).astype(int) if z.notna().any() else np.zeros(len(df), dtype=int)
    anom_cnt = int(default_flags.sum())
    anom_share = float(anom_cnt / max(1, len(df)))

    kpis_html = f"""
    <div class="kpi-grid">
      {kpi_card("Rows", f"{len(df):,}", "Records after basic sorting")}
      {kpi_card("Valid values", f"{n:,}", f"Missing: {missing:,}")}
      {kpi_card("Anomalies (default)", f"{anom_cnt:,}", f"Share: {anom_share:.2%}")}
      {kpi_card("Value range", f"{float(v.min()):.2f} to {float(v.max()):.2f}" if v.notna().any() else "Not available", "Min to max")}
    </div>
    """
    st.markdown(kpis_html, unsafe_allow_html=True)

    st.write("")

    left, right = st.columns([1.25, 1.0], gap="large")

    with left:
        date_text = "Not available"
        if start is not None and end is not None:
            date_text = f"{start} to {end}"
        card(
            "Dataset overview",
            f"""
            • Columns: {df.shape[1]:,}<br>
            • Time column: {time_col if time_col is not None else "Not detected"}<br>
            • Value column: {value_col}<br>
            • Time span: {date_text}
            """.strip(),
        )

    with right:
        # Quick descriptive stats (card)
        st.markdown('<div class="card"><h3>Value summary</h3>', unsafe_allow_html=True)
        if v.notna().any():
            s = v.describe()
            summary = pd.DataFrame(
                {
                    "Statistic": ["Mean", "Median", "Std", "P5", "P95"],
                    "Value": [
                        float(s["mean"]),
                        float(v.median()),
                        float(s["std"]) if np.isfinite(float(s["std"])) else np.nan,
                        float(v.quantile(0.05)),
                        float(v.quantile(0.95)),
                    ],
                }
            )
            st.dataframe(summary, use_container_width=True, hide_index=True)
        else:
            st.write("No valid numeric values found in the selected value column.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><h3>Signal preview</h3>', unsafe_allow_html=True)
    if time_col is not None and time_col in df.columns:
        # Align flags to cleaned series length by reusing build_features alignment logic
        X, aligned = build_features(df, time_col=time_col, value_col=value_col, rolling_window=288)
        v2 = pd.to_numeric(aligned[value_col], errors="coerce")
        z2 = robust_rolling_zscore(v2.fillna(method="ffill").fillna(method="bfill"), window=288)
        flags2 = (z2.abs() >= 4.0).astype(int).to_numpy()
        plot_timeseries_with_anomalies(aligned, time_col, value_col, flags2, "Time series with flagged points (default)")
    else:
        st.write("Time column was not detected, so a time-series plot is not available.")
    st.markdown("</div>", unsafe_allow_html=True)


def page_exploration(df_raw: pd.DataFrame, time_col: Optional[str], value_col: Optional[str]) -> None:
    st.title("Exploration")
    st.caption("Explore the signal, gaps, and stability patterns before running detection.")

    df = standardize_timeseries(df_raw)

    if value_col is None:
        card("Data check", "No numeric value column could be detected.")
        return

    v = pd.to_numeric(df[value_col], errors="coerce")
    df["_value_clean"] = v

    st.markdown('<div class="card"><h3>Filters</h3>', unsafe_allow_html=True)
    if time_col is not None and time_col in df.columns:
        t = pd.to_datetime(df[time_col], errors="coerce")
        df["_time_clean"] = t
        tmin, tmax = t.min(), t.max()
        if pd.notna(tmin) and pd.notna(tmax):
            start, end = st.slider(
                "Time range",
                min_value=tmin.to_pydatetime(),
                max_value=tmax.to_pydatetime(),
                value=(tmin.to_pydatetime(), tmax.to_pydatetime()),
            )
            mask = (df["_time_clean"] >= pd.Timestamp(start)) & (df["_time_clean"] <= pd.Timestamp(end))
            dff = df.loc[mask].copy()
        else:
            dff = df.copy()
            st.write("Time range filtering is not available due to invalid timestamps.")
    else:
        dff = df.copy()
        st.write("Time column was not detected, so time range filtering is not available.")
    st.markdown("</div>", unsafe_allow_html=True)

    if len(dff) == 0:
        st.warning("No rows match the current filters.")
        return

    st.write("")
    st.markdown('<div class="card"><h3>Distributions</h3>', unsafe_allow_html=True)
    import matplotlib.pyplot as plt

    v2 = dff["_value_clean"].dropna().astype(float)
    if len(v2) >= 10:
        fig = plt.figure()
        plt.hist(v2, bins=40)
        plt.title("Value distribution")
        plt.xlabel(value_col)
        plt.ylabel("Count")
        st.pyplot(fig, clear_figure=True)
    else:
        st.write("Not enough valid values to show a distribution.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><h3>Stability snapshot</h3>', unsafe_allow_html=True)
    # Largest absolute jumps in sequence (after sorting by time)
    seq = dff.dropna(subset=["_value_clean"]).copy()
    seq["_delta"] = seq["_value_clean"].astype(float).diff()
    top_jumps = seq.loc[seq["_delta"].abs().nlargest(10).index, [c for c in [time_col, value_col] if c is not None]].copy()
    if len(top_jumps) > 0:
        top_jumps["absolute_change"] = seq.loc[top_jumps.index, "_delta"].abs().values
        st.dataframe(top_jumps.sort_values("absolute_change", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.write("Jump analysis is not available due to insufficient data after cleaning.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><h3>Preview</h3>', unsafe_allow_html=True)
    cols_show = [c for c in [time_col, value_col] if c is not None] + [c for c in dff.columns if c not in (time_col, value_col)][:6]
    st.dataframe(dff[cols_show].head(200), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def page_detection(df_raw: pd.DataFrame, time_col: Optional[str], value_col: Optional[str]) -> None:
    st.title("Detection")
    st.caption("Run anomaly detection with configurable sensitivity and review the flagged points.")

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

    # Build features and align
    X, aligned = build_features(df, time_col=time_col, value_col=value_col, rolling_window=int(rolling_window))

    if len(X) < 50:
        st.warning("Not enough data after cleaning to run detection reliably.")
        return

    if method == "Isolation Forest":
        pipe = fit_isolation_forest(X, contamination=float(contamination), random_state=int(random_state))
        flags, scores = run_isolation_forest(pipe, X)
        score_name = "Anomaly score (higher is more anomalous)"
        higher_is_more_anomalous = True
    else:
        s = pd.to_numeric(aligned[value_col], errors="coerce").astype(float)
        z = robust_rolling_zscore(s.fillna(method="ffill").fillna(method="bfill"), window=int(rolling_window))
        scores = z.abs().to_numpy()
        flags = (z.abs() >= float(z_threshold)).astype(int).to_numpy()
        score_name = "Absolute robust z-score"
        higher_is_more_anomalous = True

    anom_cnt = int(flags.sum())
    anom_share = float(anom_cnt / len(flags))

    st.write("")
    st.markdown('<div class="card"><h3>Results</h3>', unsafe_allow_html=True)
    st.write(f"Flagged anomalies: {anom_cnt:,} ({anom_share:.2%})")
    plot_timeseries_with_anomalies(aligned, time_col, value_col, flags, "Signal with detected anomalies")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><h3>Top anomalies</h3>', unsafe_allow_html=True)
    out = aligned.copy()
    out["anomaly_flag"] = flags
    out["anomaly_score"] = scores

    cols = []
    if time_col is not None and time_col in out.columns:
        cols.append(time_col)
    cols.append(value_col)
    cols += ["anomaly_score", "anomaly_flag"]

    top = out.sort_values("anomaly_score", ascending=False).head(50)[cols]
    st.dataframe(top, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><h3>Threshold tradeoffs</h3>', unsafe_allow_html=True)
    trade = compute_threshold_tradeoff(scores, higher_is_more_anomalous=higher_is_more_anomalous)
    if len(trade) > 0:
        st.dataframe(trade, use_container_width=True, hide_index=True)
    else:
        st.write("Tradeoff view is not available for the current results.")
    st.markdown("</div>", unsafe_allow_html=True)


def page_insights(df_raw: pd.DataFrame, time_col: Optional[str], value_col: Optional[str]) -> None:
    st.title("Insights")
    st.caption("Key findings based on observed data patterns and detection behavior.")

    df = standardize_timeseries(df_raw)
    if value_col is None:
        card("Data check", "No numeric value column could be detected.")
        return

    # Create a consistent run for insights (fixed settings)
    rolling_window = 288
    X, aligned = build_features(df, time_col=time_col, value_col=value_col, rolling_window=rolling_window)

    if len(X) < 50:
        st.warning("Not enough data after cleaning to generate reliable insights.")
        return

    # Use Isolation Forest for model-driven insights baseline
    pipe = fit_isolation_forest(X, contamination=0.01, random_state=42)
    flags, scores = run_isolation_forest(pipe, X)

    # 1) Data-driven insights
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("1. Data-driven insights")

    v = pd.to_numeric(aligned[value_col], errors="coerce").astype(float)
    st.write(f"After cleaning, the dataset contains {len(v):,} measurements for '{value_col}'.")
    st.write(f"The central range (5th to 95th percentile) is {float(v.quantile(0.05)):.2f} to {float(v.quantile(0.95)):.2f}.")

    # Gaps (time span and large gaps if time available)
    if time_col is not None and time_col in aligned.columns:
        t = pd.to_datetime(aligned[time_col], errors="coerce")
        if t.notna().any():
            span = (t.max() - t.min())
            st.write(f"The observed time span is {span}.")
            gaps = t.diff().dt.total_seconds().div(60.0)
            if gaps.notna().any():
                large_gaps = gaps[gaps > gaps.quantile(0.99)]
                st.write(f"Large gaps (top 1% of time differences) indicate {int(large_gaps.size):,} intervals with unusually long spacing.")
    else:
        st.write("A time column was not detected, so time-gap analysis is not available.")

    # Descriptive stats table
    stats = pd.DataFrame(
        {
            "Statistic": ["Mean", "Median", "Std", "Min", "Max"],
            "Value": [float(v.mean()), float(v.median()), float(v.std()), float(v.min()), float(v.max())],
        }
    )
    st.dataframe(stats, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # 2) Model-driven insights
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("2. Model-driven insights")

    anom_cnt = int(flags.sum())
    anom_share = float(anom_cnt / len(flags))
    st.write(f"With the current configuration, {anom_cnt:,} points are flagged as anomalies ({anom_share:.2%}).")

    # Score bands
    s = pd.Series(scores, dtype=float)
    bands = pd.cut(s, bins=[-np.inf, s.quantile(0.50), s.quantile(0.75), s.quantile(0.90), s.quantile(0.97), np.inf])
    band_view = pd.DataFrame({"band": bands.astype(str), "count": 1}).groupby("band")["count"].sum().reset_index()
    band_view = band_view.rename(columns={"count": "Points"})
    st.subheader("Score bands")
    st.dataframe(band_view, use_container_width=True, hide_index=True)

    # Threshold tradeoffs
    st.subheader("Threshold tradeoffs")
    trade = compute_threshold_tradeoff(scores, higher_is_more_anomalous=True)
    st.dataframe(trade, use_container_width=True, hide_index=True)

    # Feature separation (simple explanation)
    st.subheader("Drivers of flagged anomalies")
    sep = separation_table(X, flags)
    if len(sep) > 0:
        st.dataframe(sep.head(12), use_container_width=True, hide_index=True)
        top = sep.iloc[0]
        st.write(
            f"The strongest separation between anomalies and normal points is observed on '{top['Feature']}'. "
            "This reflects a consistent difference in that feature between flagged and non-flagged records."
        )
    else:
        st.write("Not enough flagged points to compute stable feature separation.")

    st.markdown("</div>", unsafe_allow_html=True)


def page_data(df_raw: pd.DataFrame, meta: DataMeta) -> None:
    st.title("Data")
    st.caption(meta.source_label)

    df = df_raw.copy()

    st.markdown('<div class="card"><h3>Column health</h3>', unsafe_allow_html=True)
    missing = df.isna().mean().sort_values(ascending=False).reset_index()
    missing.columns = ["Column", "Missing share"]
    st.dataframe(missing, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><h3>Preview</h3>', unsafe_allow_html=True)
    st.dataframe(df.head(300), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# App
# ----------------------------
def main() -> None:
    st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide", initial_sidebar_state="expanded")
    inject_css()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Summary", "Exploration", "Detection", "Insights", "Data"], index=0)

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

    # Detect columns once and keep stable across pages
    time_col = detect_time_col(df_raw)
    value_col = detect_value_col(df_raw, time_col=time_col)

    if page == "Summary":
        page_summary(df_raw, meta, time_col, value_col)
    elif page == "Exploration":
        page_exploration(df_raw, time_col, value_col)
    elif page == "Detection":
        page_detection(df_raw, time_col, value_col)
    elif page == "Insights":
        page_insights(df_raw, time_col, value_col)
    elif page == "Data":
        page_data(df_raw, meta)


if __name__ == "__main__":
    main()
