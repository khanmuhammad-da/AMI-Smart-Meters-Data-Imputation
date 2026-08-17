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
# PAGE
# =============================================================================

st.set_page_config(
    page_title="AMI Smart Meter Imputation",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container {
    max-width: 1500px;
    padding-top: 1.25rem;
    padding-bottom: 2rem;
}

h1 {
    font-size: 1.85rem !important;
    margin-bottom: 0.1rem !important;
}

h2 {
    font-size: 1.20rem !important;
    margin-top: 1.15rem !important;
}

h3 {
    font-size: 1.0rem !important;
}

.subtitle {
    color: #9ca3af;
    font-size: 0.82rem;
    margin-bottom: 1rem;
}

.kpi {
    border-radius: 10px;
    padding: 12px 14px;
    min-height: 92px;
    border: 1px solid rgba(255,255,255,0.08);
}

.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    margin-bottom: 5px;
}

.kpi-value {
    font-size: 1.22rem;
    font-weight: 700;
    line-height: 1.1;
}

.kpi-sub {
    font-size: 0.65rem;
    opacity: 0.70;
    margin-top: 5px;
}

.kpi-blue   { background:#eaf2ff; color:#1758b5; }
.kpi-green  { background:#edf9f1; color:#137333; }
.kpi-orange { background:#fff7e3; color:#b45309; }
.kpi-purple { background:#f4efff; color:#6d28d9; }
.kpi-red    { background:#fff0f0; color:#b91c1c; }

.success-badge {
    background: rgba(16,185,129,0.10);
    border: 1px solid rgba(16,185,129,0.25);
    color: #10b981;
    padding: 7px 11px;
    border-radius: 8px;
    font-size: 0.76rem;
    font-weight: 600;
    text-align: center;
}

.footer-note {
    text-align: center;
    color: #7d8794;
    font-size: 0.68rem;
    padding-top: 15px;
}
</style>
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
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.header("⚡ Inference")

    uploaded_file = st.file_uploader(
        "Upload raw meter file",
        type=["csv", "xlsx", "xlsm"],
        help="One meter at a time. Half-hourly load-profile data.",
    )

    jmr = st.number_input(
        "Full-month JMR (kWh)",
        min_value=1.0,
        value=69_190_000.0,
        step=1_000.0,
        format="%.0f",
    )

    run_button = st.button(
        "Run Imputation",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.caption("Gap-specific models")

    st.write("1 LP → RF-1")
    st.write("6 LP → RF-6")
    st.write("24 LP → RF-24")
    st.write("48 LP → RF-48")

    st.divider()

    st.caption("JMR reconciliation")
    st.write("Predicted values only")
    st.write("Observed values unchanged")
    st.write("Proportional scaling")


# =============================================================================
# HEADER
# =============================================================================

st.title("AMI Smart Meter Data Imputation")

st.markdown(
    '<div class="subtitle">'
    "V1 Tuned Random Forest • Gap-specific prediction • "
    "JMR reconciliation"
    "</div>",
    unsafe_allow_html=True,
)


if uploaded_file is None:
    st.info(
        "Upload a raw CSV/XLSX meter file, enter the full-month JMR, "
        "and click **Run Imputation**."
    )

    st.markdown(
        """
### Workflow

**Raw meter file → Gap detection → RF prediction → JMR reconciliation
→ Reconciled CSV/XLSX → Full-month comparison graph**

The graph compares:

- **Blue:** actual observed meter data
- **Red:** ML-imputed + JMR-reconciled values
- **Dashed gray:** office mean-imputation baseline
- **Shaded red areas:** missing LP gaps

Every available 30-minute LP is plotted at its real timestamp.
"""
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
    "AMI Smart Meter Data Imputation • V1 Tuned Random Forest • Local Inference"
    "</div>",
    unsafe_allow_html=True,
)