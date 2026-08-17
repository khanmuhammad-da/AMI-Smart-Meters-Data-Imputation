"""
AMI Smart Meter — Streamlit Web Interface
V1 Tuned Random Forest + JMR Reconciliation

Run:
    streamlit run src/app.py
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl

mpl.rcParams["font.family"] = "Arial"
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import run_inference


# =============================================================================
# PAGE / BRANDING
# =============================================================================

st.set_page_config(
    page_title="AMI Smart Meter Imputation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ASSETS_DIR = PROJECT_ROOT / "assets"
NGCP_LOGO = ASSETS_DIR / "ngcp_logo.jpg"
GIKI_LOGO = ASSETS_DIR / "giki_logo.png"

st.markdown(
    """
<style>
.block-container {
    max-width: 1520px;
    padding-top: 0.75rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    border-right: 1px solid #E5E7EB;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}

html, body, [class*="css"] {
    font-family: Arial, Helvetica, sans-serif;
}

h1 {
    font-size: 2.0rem !important;
    font-weight: 750 !important;
    letter-spacing: -0.025em;
    margin-bottom: 0.15rem !important;
}

h2 {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    margin-top: 1.25rem !important;
}

h3 {
    font-size: 1.0rem !important;
}

.brand-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.25rem;
    padding: 0.35rem 0 0.8rem 0;
    border-bottom: 1px solid #E5E7EB;
    margin-bottom: 1rem;
}

.brand-logo-left,
.brand-logo-right {
    width: 92px;
    height: 92px;
    object-fit: contain;
}

.brand-center {
    flex: 1;
    text-align: center;
    min-width: 0;
}

.brand-kicker {
    color: #6B7280;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}

.brand-title {
    color: #111827;
    font-size: 1.65rem;
    font-weight: 750;
    line-height: 1.15;
}

.brand-subtitle {
    color: #4B5563;
    font-size: 0.82rem;
    margin-top: 0.35rem;
}

.brand-meta {
    display: inline-block;
    margin-top: 0.5rem;
    padding: 0.28rem 0.65rem;
    border: 1px solid #D1D5DB;
    border-radius: 999px;
    color: #374151;
    background: #F9FAFB;
    font-size: 0.68rem;
    font-weight: 650;
}

.hero-card {
    background: linear-gradient(135deg, #F8FAFC 0%, #FFFFFF 100%);
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 1rem 1.15rem;
    margin-bottom: 1rem;
}

.hero-title {
    font-size: 0.95rem;
    font-weight: 750;
    color: #111827;
    margin-bottom: 0.35rem;
}

.hero-text {
    color: #4B5563;
    font-size: 0.78rem;
    line-height: 1.55;
}

.workflow {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    flex-wrap: wrap;
    margin-top: 0.8rem;
}

.workflow-step {
    background: #FFFFFF;
    border: 1px solid #D1D5DB;
    border-radius: 8px;
    padding: 0.38rem 0.55rem;
    color: #1F2937;
    font-size: 0.68rem;
    font-weight: 650;
}

.workflow-arrow {
    color: #9CA3AF;
    font-size: 0.8rem;
}

.kpi {
    border-radius: 12px;
    padding: 13px 15px;
    min-height: 96px;
    border: 1px solid rgba(17,24,39,0.08);
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}

.kpi-label {
    font-size: 0.70rem;
    font-weight: 700;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.025em;
}

.kpi-value {
    font-size: 1.20rem;
    font-weight: 750;
    line-height: 1.1;
}

.kpi-sub {
    font-size: 0.64rem;
    opacity: 0.72;
    margin-top: 6px;
}

.kpi-blue   { background:#EEF5FF; color:#174EA6; }
.kpi-green  { background:#EEF9F1; color:#137333; }
.kpi-orange { background:#FFF7E6; color:#A85D00; }
.kpi-purple { background:#F5F0FF; color:#6B35C8; }
.kpi-red    { background:#FFF1F1; color:#B42318; }

.success-badge {
    background: #ECFDF3;
    border: 1px solid #A7F3D0;
    color: #047857;
    padding: 8px 12px;
    border-radius: 9px;
    font-size: 0.76rem;
    font-weight: 700;
    text-align: center;
}

.section-note {
    color: #6B7280;
    font-size: 0.72rem;
    margin-top: -0.55rem;
    margin-bottom: 0.65rem;
}

.sidebar-card {
    background: #F8FAFC;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 0.75rem 0.8rem;
    margin: 0.65rem 0;
}

.sidebar-card-title {
    font-size: 0.72rem;
    font-weight: 750;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.4rem;
}

.sidebar-item {
    font-size: 0.72rem;
    color: #4B5563;
    line-height: 1.65;
}

.footer-note {
    text-align: center;
    color: #9CA3AF;
    font-size: 0.66rem;
    padding-top: 18px;
    margin-top: 1.5rem;
    border-top: 1px solid #E5E7EB;
}
</style>
""",
    unsafe_allow_html=True,
)


def _logo_data_uri(path: Path) -> str:
    """Return a local image as a data URI for reliable Streamlit rendering."""
    import base64

    if not path.exists():
        return ""

    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


ngcp_uri = _logo_data_uri(NGCP_LOGO)
giki_uri = _logo_data_uri(GIKI_LOGO)

left_logo = (
    f'<img class="brand-logo-left" src="{ngcp_uri}" alt="NGCP logo">'
    if ngcp_uri
    else '<div class="brand-logo-left"></div>'
)

right_logo = (
    f'<img class="brand-logo-right" src="{giki_uri}" alt="GIKI logo">'
    if giki_uri
    else '<div class="brand-logo-right"></div>'
)

st.markdown(
    f"""
<div class="brand-header">
    <div>{left_logo}</div>
    <div class="brand-center">
        <div class="brand-kicker">AMI / Smart Metering • Machine Learning</div>
        <div class="brand-title">AMI Smart Meter Data Imputation</div>
        <div class="brand-subtitle">
            V1 Tuned Random Forest • Gap-specific prediction • JMR reconciliation
        </div>
        <div class="brand-meta">NGC × GIKI AI Bootcamp 2026</div>
    </div>
    <div>{right_logo}</div>
</div>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# HELPERS
# =============================================================================

def fmt_kwh(value) -> str:
    try:
        return f"{float(value):,.0f} kWh"
    except Exception:
        return "—"


def fmt_num(value) -> str:
    try:
        return f"{float(value):,.0f}"
    except Exception:
        return "—"


def read_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(path)
    raise RuntimeError(f"Unsupported file type: {suffix}")


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    lookup = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def find_time_column(df: pd.DataFrame) -> str:
    column = find_column(
        df,
        [
            "Time",
            "Timestamp",
            "DateTime",
            "Date Time",
            "datetime",
        ],
    )

    if column is None:
        raise RuntimeError(
            f"Could not identify timestamp column. Found: {list(df.columns)}"
        )

    return column


def find_lp_column(df: pd.DataFrame) -> str:
    column = find_column(
        df,
        [
            "A- [kWh]",
            "A-",
            "LP",
            "Load",
            "Load Profile",
            "load_profile",
            "Energy",
            "Value",
        ],
    )

    if column is not None:
        return column

    numeric = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    if numeric:
        return numeric[0]

    raise RuntimeError(
        f"Could not identify load-profile column. Found: {list(df.columns)}"
    )


def find_final_value_column(
    result_df: pd.DataFrame,
    original_lp_column: str,
) -> str:
    """
    Locate the final reconciled/imputed value in the inference output.
    """

    candidates = [
        "Reconciled LP",
        "Reconciled Value",
        "JMR Reconciled LP",
        "JMR Reconciled Value",
        "Final LP",
        "Final Value",
        "Imputed + JMR Reconciled LP",
        "imputed_jmr_reconciled",
        "reconciled_lp",
        "reconciled_value",
        "final_lp",
        "final_value",
    ]

    found = find_column(result_df, candidates)
    if found:
        return found

    for column in result_df.columns:
        text = str(column).lower()
        if "recon" in text and (
            "lp" in text or "value" in text or "kwh" in text
        ):
            return column

    # Some inference versions may use the original LP column for the final
    # reconciled series.
    if original_lp_column in result_df.columns:
        return original_lp_column

    raise RuntimeError(
        "Could not identify the final reconciled LP column in the "
        f"inference output. Found: {list(result_df.columns)}"
    )


def find_output_path(
    summary: dict,
    keys: list[str],
    output_dir: Path,
    stem: str,
    suffix: str,
) -> Path | None:

    outputs = summary.get("outputs", {})

    for key in keys:
        value = outputs.get(key)
        if value:
            path = Path(str(value))
            if path.exists():
                return path

        value = summary.get(key)
        if value:
            path = Path(str(value))
            if path.exists():
                return path

    files = sorted(
        output_dir.glob(f"{stem}*{suffix}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return files[0] if files else None


def parse_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series,
        errors="coerce",
        format="mixed",
        dayfirst=True,
    )


# =============================================================================
# FULL-MONTH PLOT
# =============================================================================

def build_full_month_plot(
    raw_df: pd.DataFrame,
    result_df: pd.DataFrame,
    time_column: str,
    lp_column: str,
    final_column: str,
    jmr_kwh: float,
):
    """
    Build the chart from the RAW meter grid.

    IMPORTANT:
    The raw file is the authoritative timeline. The result output is only
    used to replace the missing rows. This guarantees that:
        - the complete month is shown;
        - all 30-minute observations are represented;
        - observed values remain observed;
        - only missing rows become ML-imputed values;
        - no artificial straight lines are created between sparse result rows.
    """

    raw = raw_df.copy()
    result = result_df.copy()

    raw["_time"] = parse_time(raw[time_column])
    raw["_observed"] = pd.to_numeric(raw[lp_column], errors="coerce")

    result_time_column = (
        time_column
        if time_column in result.columns
        else find_time_column(result)
    )

    result["_time"] = parse_time(result[result_time_column])
    result["_final"] = pd.to_numeric(
        result[final_column],
        errors="coerce",
    )

    raw = raw.dropna(subset=["_time"]).sort_values("_time").reset_index(drop=True)
    result = result.dropna(subset=["_time"]).sort_values("_time")

    # -------------------------------------------------------------------------
    # Build a complete 30-minute master timeline from the RAW FILE.
    # -------------------------------------------------------------------------

    if raw.empty:
        raise RuntimeError("No valid timestamps found in input.")

    start = raw["_time"].min().floor("30min")
    end = raw["_time"].max().ceil("30min")

    master = pd.DataFrame(
        {
            "_time": pd.date_range(
                start=start,
                end=end,
                freq="30min",
            )
        }
    )

    raw_indexed = raw.set_index("_time")

    master["_observed"] = master["_time"].map(
        raw_indexed["_observed"]
    )

    # -------------------------------------------------------------------------
    # Get final predictions by timestamp.
    # -------------------------------------------------------------------------

    result_by_time = (
        result.drop_duplicates("_time", keep="last")
        .set_index("_time")["_final"]
    )

    master["_model"] = master["_time"].map(result_by_time)

    # -------------------------------------------------------------------------
    # IMPORTANT:
    # Final profile = observed value wherever it exists,
    #                 model value only where original value is missing.
    # -------------------------------------------------------------------------

    master["_missing"] = master["_observed"].isna()

    master["_final"] = master["_observed"]

    replace_mask = master["_missing"] & master["_model"].notna()

    master.loc[replace_mask, "_final"] = master.loc[
        replace_mask, "_model"
    ]

    # -------------------------------------------------------------------------
    # Office baseline:
    #
    # remaining JMR / number of missing LPs
    # -------------------------------------------------------------------------

    observed_total = float(master["_observed"].sum(skipna=True))
    missing_count = int(master["_missing"].sum())

    remaining = float(jmr_kwh - observed_total)

    baseline_mean = (
        remaining / missing_count
        if missing_count > 0
        else 0.0
    )

    master["_baseline"] = float("nan")
    master.loc[
        master["_missing"],
        "_baseline",
    ] = baseline_mean

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(17, 6.2), dpi=150)

    observed = master["_observed"].notna()
    imputed = master["_missing"] & master["_final"].notna()

    # -------------------------------------------------------------------------
    # STEP-PLOT DESIGN
    #
    # The meter is sampled every 30 minutes. A step plot is more appropriate
    # for a discrete load-profile series than a smooth-looking line plot.
    #
    # We use:
    #   BLUE  = observed values
    #   RED   = ML + JMR-reconciled values inside missing gaps
    #   GREY  = office mean-imputation baseline inside missing gaps
    #
    # Important boundary rule:
    # The red step for each gap includes the observed point immediately before
    # and immediately after the gap. This creates a visually continuous
    # transition:
    #
    #       observed BLUE ── RED IMPUTED ── observed BLUE
    #
    # while the red markers themselves are placed ONLY on the missing LPs.
    # Therefore the graph never shows a misleading blue line through a gap,
    # but it also does not look artificially broken at the boundaries.
    # -------------------------------------------------------------------------

    observed = master["_observed"].notna()
    imputed = master["_missing"] & master["_final"].notna()

    # Blue observed series: missing cells are NaN, so blue never runs through
    # an unavailable meter interval.
    observed_step = master["_observed"].copy()
    observed_step.loc[master["_missing"]] = float("nan")

    # Detect contiguous missing blocks.
    gap_indices = master.index[master["_missing"]].tolist()
    gap_blocks = []

    if gap_indices:
        block = [gap_indices[0]]

        for idx in gap_indices[1:]:
            if idx == block[-1] + 1:
                block.append(idx)
            else:
                gap_blocks.append(block)
                block = [idx]

        gap_blocks.append(block)

    # -------------------------------------------------------------------------
    # Base observed step profile.
    # -------------------------------------------------------------------------
    ax.step(
        master["_time"],
        observed_step,
        where="post",
        color="#2563EB",
        linewidth=1.45,
        label="Observed",
        zorder=3,
    )

    # -------------------------------------------------------------------------
    # Each missing block gets a red step bridge.
    # -------------------------------------------------------------------------
    for block_no, block in enumerate(gap_blocks, start=1):

        gap = master.loc[block]

        first_pos = master.index.get_loc(block[0])
        last_pos = master.index.get_loc(block[-1])

        # Include the last observed point before the gap and the first observed
        # point after the gap. These boundary points make the red step connect
        # cleanly with the blue observed profile.
        bridge_positions = []

        if first_pos > 0:
            bridge_positions.append(first_pos - 1)

        bridge_positions.extend(range(first_pos, last_pos + 1))

        if last_pos < len(master) - 1:
            bridge_positions.append(last_pos + 1)

        bridge = master.iloc[bridge_positions].copy()

        bridge["_display_value"] = bridge["_observed"]

        missing_bridge = bridge["_missing"]

        bridge.loc[
            missing_bridge,
            "_display_value",
        ] = bridge.loc[
            missing_bridge,
            "_final",
        ]

        # Shade the exact missing period.
        ax.axvspan(
            gap["_time"].iloc[0] - pd.Timedelta(minutes=15),
            gap["_time"].iloc[-1] + pd.Timedelta(minutes=15),
            color="#FCA5A5",
            alpha=0.16,
            zorder=0,
        )

        # Red step bridge. This is what visually connects observed → imputed
        # → observed without drawing a false observed line through the gap.
        ax.step(
            bridge["_time"],
            bridge["_display_value"],
            where="post",
            color="#DC2626",
            linewidth=2.35,
            zorder=5,
            label=(
                "ML Imputed + JMR Reconciled"
                if block_no == 1
                else None
            ),
        )

        # Highlight ONLY the actual imputed LPs with small markers.
        ax.scatter(
            gap["_time"],
            gap["_final"],
            color="#DC2626",
            s=12,
            zorder=6,
        )

        # Office baseline: only across the missing LP interval.
        ax.plot(
            gap["_time"],
            gap["_baseline"],
            color="#6B7280",
            linestyle="--",
            linewidth=1.25,
            label=(
                "Office Mean Baseline"
                if block_no == 1
                else None
            ),
            zorder=4,
        )

        # Gap label.
        y_top = float(
            pd.concat(
                [
                    master["_observed"],
                    master["_final"],
                ]
            ).max()
        )

        gap_length = len(block)

        ax.text(
            gap["_time"].iloc[len(gap) // 2],
            y_top * 0.985,
            f"Gap {block_no} ({gap_length} LP)",
            ha="center",
            va="top",
            fontsize=8,
            fontname="Arial",
            fontweight="bold",
            color="#B91C1C",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "#FCA5A5",
                "linewidth": 0.8,
                "alpha": 0.92,
            },
            zorder=7,
        )

    # -------------------------------------------------------------------------
    # X axis: explicit HALF-HOURLY timeline, daily labels.
    #
    # We don't label all 1,440 points because that would be unreadable.
    # Every data point is still plotted at its actual 30-minute timestamp.
    # -------------------------------------------------------------------------

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    ax.xaxis.set_minor_locator(
        mdates.HourLocator(
            byhour=[0, 6, 12, 18]
        )
    )

    ax.tick_params(
        axis="x",
        which="major",
        labelsize=8,
        rotation=45,
    )

    ax.tick_params(
        axis="x",
        which="minor",
        length=3,
    )

    for tick in ax.get_xticklabels():
        tick.set_fontname("Arial")
        tick.set_fontsize(8)

    for tick in ax.get_yticklabels():
        tick.set_fontname("Arial")
        tick.set_fontsize(8)

    ax.grid(
        axis="y",
        alpha=0.18,
        linewidth=0.65,
    )

    ax.grid(
        axis="x",
        which="major",
        alpha=0.10,
        linewidth=0.65,
    )

    ax.set_xlim(
        master["_time"].min() - pd.Timedelta(hours=2),
        master["_time"].max() + pd.Timedelta(hours=2),
    )

    ax.set_xlabel(
        "Date — 30-minute load-profile intervals",
        fontsize=9,
        fontname="Arial",
    )

    ax.set_ylabel(
        "Energy (kWh / 30 min)",
        fontsize=9,
        fontname="Arial",
    )

    ax.set_title(
        f"AMI Load Profile — Observed vs ML Imputation\n"
        f"JMR = {jmr_kwh:,.0f} kWh",
        fontsize=12,
        fontweight="bold",
        fontname="Arial",
        pad=12,
    )

    ax.legend(
        loc="upper left",
        fontsize=8,
        frameon=True,
        framealpha=0.94,
        edgecolor="#D1D5DB",
        prop={"family": "Arial", "size": 8},
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=160,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    buffer.seek(0)

    return buffer, master, baseline_mean


# =============================================================================
# GAP / BASELINE TABLE
# =============================================================================

def build_gap_comparison(
    master: pd.DataFrame,
    baseline_mean: float,
) -> pd.DataFrame:

    missing = master["_missing"] & master["_final"].notna()

    indices = master.index[missing].tolist()

    if not indices:
        return pd.DataFrame()

    blocks = []
    block = [indices[0]]

    for idx in indices[1:]:
        if idx == block[-1] + 1:
            block.append(idx)
        else:
            blocks.append(block)
            block = [idx]

    blocks.append(block)

    rows = []

    for gap_id, block in enumerate(blocks, start=1):
        gap = master.loc[block]

        model_total = float(gap["_final"].sum())
        baseline_total = float(
            baseline_mean * len(block)
        )

        rows.append(
            {
                "Gap": f"Gap {gap_id}",
                "Start": gap["_time"].iloc[0].strftime(
                    "%Y-%m-%d %H:%M"
                ),
                "End": gap["_time"].iloc[-1].strftime(
                    "%Y-%m-%d %H:%M"
                ),
                "Missing LPs": len(block),
                "ML Reconciled (kWh)": round(model_total),
                "Office Baseline (kWh)": round(baseline_total),
                "ML − Baseline (kWh)": round(
                    model_total - baseline_total
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# SIDEBAR / INFERENCE CONTROLS
# =============================================================================

with st.sidebar:
    st.markdown("### ⚡ Inference Controls")
    st.caption("One meter at a time • Half-hourly load-profile data")

    uploaded_file = st.file_uploader(
        "Upload raw meter file",
        type=["csv", "xlsx", "xlsm"],
        help="Upload the raw meter load-profile file.",
    )

    jmr = st.number_input(
        "Full-month JMR (kWh)",
        min_value=1.0,
        value=69_190_000.0,
        step=1_000.0,
        format="%.0f",
        help="Enter the authoritative full-month JMR energy.",
    )

    run_button = st.button(
        "▶  Run Imputation",
        type="primary",
        use_container_width=True,
    )

    st.markdown(
        """
<div class="sidebar-card">
    <div class="sidebar-card-title">Gap-specific models</div>
    <div class="sidebar-item">1 LP → RF-1</div>
    <div class="sidebar-item">6 LP → RF-6</div>
    <div class="sidebar-item">24 LP → RF-24</div>
    <div class="sidebar-item">48 LP → RF-48</div>
</div>

<div class="sidebar-card">
    <div class="sidebar-card-title">JMR reconciliation</div>
    <div class="sidebar-item">• Predicted values only</div>
    <div class="sidebar-item">• Observed values unchanged</div>
    <div class="sidebar-item">• Proportional scaling</div>
</div>

<div class="sidebar-card">
    <div class="sidebar-card-title">Output</div>
    <div class="sidebar-item">CSV + Excel reconciliation</div>
    <div class="sidebar-item">Full-month comparison graph</div>
    <div class="sidebar-item">Gap-level diagnostics</div>
</div>
""",
        unsafe_allow_html=True,
    )


# =============================================================================
# LANDING / EMPTY STATE
# =============================================================================

if uploaded_file is None:
    st.markdown(
        """
<div class="hero-card">
    <div class="hero-title">Ready for inference</div>
    <div class="hero-text">
        Upload a raw meter CSV/XLSX file, enter the authoritative full-month
        JMR, then run the tuned Random Forest imputation pipeline.
    </div>

    <div class="workflow">
        <span class="workflow-step">1. Upload meter file</span>
        <span class="workflow-arrow">→</span>
        <span class="workflow-step">2. Detect gaps</span>
        <span class="workflow-arrow">→</span>
        <span class="workflow-step">3. RF prediction</span>
        <span class="workflow-arrow">→</span>
        <span class="workflow-step">4. JMR reconciliation</span>
        <span class="workflow-arrow">→</span>
        <span class="workflow-step">5. Export results</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.subheader("What the application provides")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
**Gap-specific prediction**

Uses RF-1, RF-6, RF-24 and RF-48 according to gap length.
"""
        )

    with c2:
        st.markdown(
            """
**JMR reconciliation**

Observed LPs remain unchanged; only missing values are reconciled.
"""
        )

    with c3:
        st.markdown(
            """
**Engineering diagnostics**

Full-month profile, gap-level comparison and downloadable outputs.
"""
        )

    st.caption(
        "Graph convention: blue = observed, red = ML + JMR reconciled, "
        "dashed gray = office mean baseline, shaded areas = missing LP gaps."
    )

    st.stop()


# =============================================================================
# EXECUTION
# =============================================================================

if run_button:

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="ami_streamlit_"
        )
    )

    input_path = temp_dir / uploaded_file.name

    try:

        input_path.write_bytes(
            uploaded_file.getbuffer()
        )

        with st.spinner(
            "Running tuned models and JMR reconciliation..."
        ):

            summary = run_inference(
                input_path=input_path,
                jmr_kwh=float(jmr),
            )

        # ---------------------------------------------------------------------
        # Locate inference outputs.
        # ---------------------------------------------------------------------

        output_dir = PROJECT_ROOT / "outputs" / "inference"

        stem = input_path.stem

        csv_path = find_output_path(
            summary,
            ["csv", "csv_output", "csv_path"],
            output_dir,
            stem,
            ".csv",
        )

        xlsx_path = find_output_path(
            summary,
            [
                "xlsx",
                "excel",
                "excel_output",
                "xlsx_output",
                "xlsx_path",
            ],
            output_dir,
            stem,
            ".xlsx",
        )

        if csv_path is None:
            raise RuntimeError(
                "Inference completed but reconciled CSV was not found."
            )

        # ---------------------------------------------------------------------
        # Read raw and final data.
        # ---------------------------------------------------------------------

        raw_df = read_file(input_path)

        result_df = pd.read_csv(csv_path)

        time_column = find_time_column(raw_df)
        lp_column = find_lp_column(raw_df)

        final_column = find_final_value_column(
            result_df,
            lp_column,
        )

        # ---------------------------------------------------------------------
        # FULL-MONTH GRAPH DATA.
        # ---------------------------------------------------------------------

        plot_buffer, master, baseline_mean = build_full_month_plot(
            raw_df=raw_df,
            result_df=result_df,
            time_column=time_column,
            lp_column=lp_column,
            final_column=final_column,
            jmr_kwh=float(jmr),
        )

        missing_mask = master["_missing"]
        imputed_mask = (
            master["_missing"]
            & master["_final"].notna()
        )

        observed_total = float(
            master["_observed"].sum(
                skipna=True
            )
        )

        ml_missing_total = float(
            master.loc[
                imputed_mask,
                "_final"
            ].sum()
        )

        final_total = observed_total + ml_missing_total

        reconciliation_error = (
            final_total - float(jmr)
        )

        missing_count = int(
            missing_mask.sum()
        )

        gap_df = build_gap_comparison(
            master,
            baseline_mean,
        )

        baseline_total = (
            baseline_mean * missing_count
        )

        improvement = (
            ml_missing_total
            - baseline_total
        )

        # ---------------------------------------------------------------------
        # KPI CARDS
        # ---------------------------------------------------------------------

        st.subheader("Inference Summary")
        st.markdown('<div class="section-note">Monthly energy reconciliation and missing-load diagnostics</div>', unsafe_allow_html=True)

        k1, k2, k3, k4, k5 = st.columns(5)

        with k1:
            st.markdown(
                f"""
<div class="kpi kpi-blue">
<div class="kpi-label">JMR (Monthly)</div>
<div class="kpi-value">{fmt_kwh(jmr)}</div>
<div class="kpi-sub">Target energy</div>
</div>
""",
                unsafe_allow_html=True,
            )

        with k2:
            st.markdown(
                f"""
<div class="kpi kpi-green">
<div class="kpi-label">Observed Energy</div>
<div class="kpi-value">{fmt_kwh(observed_total)}</div>
<div class="kpi-sub">Available meter data</div>
</div>
""",
                unsafe_allow_html=True,
            )

        with k3:
            st.markdown(
                f"""
<div class="kpi kpi-orange">
<div class="kpi-label">Missing Energy Required</div>
<div class="kpi-value">{fmt_kwh(jmr - observed_total)}</div>
<div class="kpi-sub">Energy required in missing LPs</div>
</div>
""",
                unsafe_allow_html=True,
            )

        with k4:
            st.markdown(
                f"""
<div class="kpi kpi-purple">
<div class="kpi-label">Final Total</div>
<div class="kpi-value">{fmt_kwh(final_total)}</div>
<div class="kpi-sub">JMR reconciled</div>
</div>
""",
                unsafe_allow_html=True,
            )

        with k5:
            st.markdown(
                f"""
<div class="kpi kpi-red">
<div class="kpi-label">Missing LPs</div>
<div class="kpi-value">{missing_count}</div>
<div class="kpi-sub">Across {len(gap_df)} detected gaps</div>
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        if abs(reconciliation_error) < 1e-6:
            st.markdown(
                '<div class="success-badge">'
                "✓ Reconciliation Error: 0 kWh"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning(
                f"Reconciliation error: "
                f"{reconciliation_error:,.0f} kWh"
            )

        # ---------------------------------------------------------------------
        # MAIN FULL-MONTH GRAPH
        # ---------------------------------------------------------------------

        st.subheader("Load Profile Comparison")
        st.markdown('<div class="section-note">Complete 30-minute timeline from the raw meter grid</div>', unsafe_allow_html=True)

        st.image(
            plot_buffer,
            use_container_width=True,
        )

        st.caption(
            "Blue = observed. Red = ML + JMR reconciled missing LPs. "
            "Dashed gray = office mean baseline inside gaps. "
            "All available half-hourly observations are plotted."
        )

        # ---------------------------------------------------------------------
        # GAP DETAILS
        # ---------------------------------------------------------------------

        st.subheader("Gap Details")
        st.markdown('<div class="section-note">Comparison of ML-reconciled energy against the office mean baseline</div>', unsafe_allow_html=True)

        if not gap_df.empty:

            st.dataframe(
                gap_df,
                use_container_width=True,
                hide_index=True,
            )

            if improvement >= 0:

                st.success(
                    f"ML vs office mean baseline: "
                    f"**{improvement:,.0f} kWh** difference "
                    f"across the missing LPs."
                )

            else:

                st.warning(
                    f"ML vs office mean baseline: "
                    f"{improvement:,.0f} kWh difference."
                )

            st.caption(
                "Office baseline = "
                "(JMR − observed monthly energy) ÷ "
                "missing LP count = "
                f"{baseline_mean:,.0f} kWh per missing LP."
            )

        else:

            st.info(
                "No missing LPs were detected."
            )

        # ---------------------------------------------------------------------
        # DOWNLOADS
        # ---------------------------------------------------------------------

        st.subheader("Download Results")
        st.markdown('<div class="section-note">Export the reconciled dataset and comparison visualization</div>', unsafe_allow_html=True)

        d1, d2, d3 = st.columns(3)

        with d1:

            st.download_button(
                "⬇ Download Reconciled CSV",
                data=csv_path.read_bytes(),
                file_name=csv_path.name,
                mime="text/csv",
                use_container_width=True,
            )

        with d2:

            if xlsx_path:

                st.download_button(
                    "⬇ Download Reconciled Excel",
                    data=xlsx_path.read_bytes(),
                    file_name=xlsx_path.name,
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    use_container_width=True,
                )

            else:

                st.info(
                    "Excel output not found."
                )

        with d3:

            st.download_button(
                "⬇ Download Comparison Plot",
                data=plot_buffer.getvalue(),
                file_name=f"{stem}_comparison.png",
                mime="image/png",
                use_container_width=True,
            )

        # ---------------------------------------------------------------------
        # TECHNICAL DETAILS
        # ---------------------------------------------------------------------

        with st.expander("Technical Details"):

            st.write(
                {
                    "Input file": uploaded_file.name,
                    "Input rows": len(raw_df),
                    "Timestamp column": time_column,
                    "LP column": lp_column,
                    "Final output column": final_column,
                    "Timeline frequency": "30 minutes",
                    "Missing LPs": missing_count,
                    "Baseline mean (kWh/LP)": round(
                        baseline_mean
                    ),
                    "ML missing energy (kWh)": round(
                        ml_missing_total
                    ),
                    "Baseline missing energy (kWh)": round(
                        baseline_total
                    ),
                    "Final total (kWh)": round(
                        final_total
                    ),
                    "Reconciliation error (kWh)": round(
                        reconciliation_error
                    ),
                    "MLflow run": summary.get(
                        "mlflow_run_id",
                        summary.get("run_id", "—"),
                    ),
                }
            )

            st.json(summary)

    except Exception as exc:

        st.error(
            f"Inference failed: "
            f"{type(exc).__name__}: {exc}"
        )

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )


st.markdown(
    '<div class="footer-note">'
    "AMI Smart Meter Data Imputation • V1 Tuned Random Forest • JMR Reconciliation"
    " • NGC × GIKI AI Bootcamp 2026"
    "</div>",
    unsafe_allow_html=True,
)