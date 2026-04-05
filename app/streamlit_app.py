"""
app/streamlit_app.py
====================

F1 Strategy Intelligence Dashboard — Phase 3b.

Tabs:
  1. Pre-Race Strategy  — optimizer with pit windows, degradation curves, deltas
  2. Safety Car What-If  — SC scenario analysis (pit vs stay out)
  3. Race Analysis       — actual historical lap data per driver per race
  4. Season Form         — current season driver/team performance
"""

import os
import sys

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dashboard.api_client import F1StrategyAPIClient

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
try:
    API_URL = os.getenv("API_URL") or st.secrets.get("API_URL", "http://localhost:8000")
except Exception:
    API_URL = os.getenv("API_URL", "http://localhost:8000")

api = F1StrategyAPIClient(API_URL)

st.set_page_config(
    page_title="F1 Strategy Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Quick API connectivity check (fail fast instead of hanging for 90s)
# ---------------------------------------------------------------------------
import requests as _req

@st.cache_data(ttl=60, show_spinner=False)
def _check_api(url: str) -> bool:
    try:
        _req.get(f"{url}/data/races", params={"year": 2024}, timeout=5)
        return True
    except Exception:
        return False

if not _check_api(API_URL):
    _is_localhost = "localhost" in API_URL or "127.0.0.1" in API_URL
    if _is_localhost:
        st.error(
            "**Cannot reach the API** — currently pointing at `localhost`.\n\n"
            "If you're on **Streamlit Cloud**, add this secret "
            "(Settings → Secrets):\n"
            "```toml\n"
            'API_URL = "https://f1-strategy-simulator-xlev.onrender.com"\n'
            "```\n"
            "Then **Reboot app** (top-right menu)."
        )
    else:
        st.warning(
            f"**API is not responding** (`{API_URL}`). "
            "The Render free tier takes ~30s to wake up — try refreshing in a minute."
        )
    st.caption(f"Configured API URL: `{API_URL}`")

# ---------------------------------------------------------------------------
# Compound colour palette
# ---------------------------------------------------------------------------
COMPOUND_COLORS = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFC300",
    "HARD": "#CCCCCC",
    "INTERMEDIATE": "#39B54A",
    "WET": "#0072CE",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt_time(seconds: float) -> str:
    """Format seconds as m:ss.fff"""
    if seconds is None:
        return "—"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:06.3f}"


def _pit_laps_from_stints(stints: list) -> list:
    """Derive pit lap numbers from stint lengths. [20, 32] -> [20]"""
    pit_laps = []
    cumulative = 0
    for s in stints[:-1]:
        cumulative += s
        pit_laps.append(cumulative)
    return pit_laps


def _strategy_label(compounds: list, stints: list) -> str:
    """e.g. 'MEDIUM(20) → HARD(32)' """
    parts = []
    for c, s in zip(compounds, stints):
        parts.append(f"{c}({s})")
    return " → ".join(parts)


def _delta_str(delta: float) -> str:
    """Format delta: +1.234s or -0.567s"""
    if delta == 0:
        return "BEST"
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.1f}s"


@st.cache_data(ttl=300)
def _load_races(year: int):
    try:
        return api.list_races(year=year)
    except Exception:
        return []


@st.cache_data(ttl=300)
def _load_drivers(year: int):
    try:
        return api.list_drivers(year=year)
    except Exception:
        return []


@st.cache_data(ttl=600)
def _load_track_metrics():
    try:
        return api.list_track_metrics()
    except Exception:
        return []


# ===========================================================================
# SIDEBAR
# ===========================================================================
st.sidebar.title("F1 Strategy Intelligence")

year = st.sidebar.number_input("Season", min_value=2020, max_value=2030, value=2024)

races = _load_races(year)
race_options = {f"R{r['round']}: {r['event_name']}": r for r in races}

if race_options:
    selected_race_label = st.sidebar.selectbox("Grand Prix", list(race_options.keys()))
    selected_race = race_options[selected_race_label]
    round_number = selected_race["round"]
    event_name = selected_race["event_name"]
    race_laps = selected_race.get("total_laps") or 57
else:
    st.sidebar.warning("No races found. Enter manually.")
    round_number = st.sidebar.number_input("Round", 1, 30, 1)
    event_name = st.sidebar.text_input("Event Name", "")
    race_laps = 57

race_laps = st.sidebar.number_input("Race Laps", 30, 80, race_laps, key="laps_override")

drivers = _load_drivers(year)
if drivers:
    driver_id = st.sidebar.selectbox("Driver", drivers)
else:
    driver_id = st.sidebar.text_input("Driver ID", "VER")

is_wet = st.sidebar.checkbox("Wet Race")

with st.sidebar.expander("Simulation Params"):
    num_sims = st.slider("MC Simulations", 100, 2000, 300, 100)
    risk_penalty = st.slider("Risk Penalty", 0.0, 3.0, 1.0, 0.1)

tab_selection = st.sidebar.radio(
    "View",
    ["Pre-Race Strategy", "Safety Car What-If", "Race Analysis", "Season Form"],
)

# ===========================================================================
st.title("F1 Strategy Intelligence Dashboard")


# ===========================================================================
# TAB 1: PRE-RACE STRATEGY
# ===========================================================================
if tab_selection == "Pre-Race Strategy":
    st.header(f"Pre-Race Strategy — {event_name} {year}")

    # GP info bar
    track_metrics = _load_track_metrics()
    pit_loss_val = next(
        (tm["avg_pit_loss"] for tm in track_metrics if tm["event_name"] == event_name),
        None,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Circuit", event_name)
    c2.metric("Laps", race_laps)
    c3.metric("Driver", driver_id)
    c4.metric("Pit Loss", f"{pit_loss_val:.1f}s" if pit_loss_val else "~22s (default)")

    st.divider()

    if st.button("Run Strategy Optimization", type="primary", use_container_width=True):
        payload = {
            "race_laps": race_laps,
            "is_wet_race": is_wet,
            "year": year,
            "round": round_number,
            "driver_id": driver_id,
            "event_name": event_name,
            "num_simulations": num_sims,
            "risk_penalty": risk_penalty,
        }
        with st.spinner("Running Bayesian optimization..."):
            try:
                result = api.optimize_strategy(payload)
            except Exception as e:
                st.error(f"Optimization failed: {e}")
                st.stop()
        st.session_state["opt_result"] = result

    if "opt_result" not in st.session_state:
        st.info("Click **Run Strategy Optimization** to find the best race strategy.")
        st.stop()

    result = st.session_state["opt_result"]
    all_strats = result.get("all_strategies", [])
    if not all_strats:
        st.warning("No strategies evaluated.")
        st.stop()

    best = result["best_strategy"]
    best_p50 = result["p50"]
    best_stints = best["stints"]
    best_compounds = best["compounds"]
    best_pit_laps = _pit_laps_from_stints(best_stints)

    # ── Best Strategy Hero Card ──
    st.subheader("Recommended Strategy")
    hero1, hero2, hero3 = st.columns([3, 2, 2])
    with hero1:
        st.markdown(f"### {_strategy_label(best_compounds, best_stints)}")
        pit_str = ", ".join([f"Lap {p}" for p in best_pit_laps]) if best_pit_laps else "No pit"
        st.markdown(f"**Pit stops:** {pit_str}")
    with hero2:
        st.metric("Predicted Race Time", _fmt_time(best_p50))
    with hero3:
        st.metric("Consistency (Std)", f"{result['std_time']:.1f}s")

    st.divider()

    # ── Degradation Curve for Best Strategy ──
    st.subheader("Tire Degradation Curve — Best Strategy")
    try:
        deg_payload = {
            "year": year,
            "round": round_number,
            "driver_id": driver_id,
            "stints": best_stints,
            "compounds": best_compounds,
        }
        deg = api.get_degradation_curve(deg_payload)

        deg_df = pd.DataFrame({
            "Lap": deg["laps"],
            "p10": deg["p10"],
            "p50 (Predicted)": deg["p50"],
            "p90": deg["p90"],
            "Compound": deg["compound"],
            "Stint": deg["stint_number"],
        })

        fig_deg = go.Figure()

        # Shaded uncertainty band
        fig_deg.add_trace(go.Scatter(
            x=deg_df["Lap"], y=deg_df["p90"],
            mode="lines", line=dict(width=0),
            showlegend=False,
        ))
        fig_deg.add_trace(go.Scatter(
            x=deg_df["Lap"], y=deg_df["p10"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(100,150,255,0.15)",
            name="p10–p90 range",
        ))

        # Colored p50 line per compound
        for compound in deg_df["Compound"].unique():
            mask = deg_df["Compound"] == compound
            fig_deg.add_trace(go.Scatter(
                x=deg_df.loc[mask, "Lap"],
                y=deg_df.loc[mask, "p50 (Predicted)"],
                mode="lines+markers",
                name=compound,
                line=dict(color=COMPOUND_COLORS.get(compound, "#888"), width=3),
                marker=dict(size=3),
            ))

        # Pit stop vertical lines
        for pl in deg["pit_laps"]:
            fig_deg.add_vline(
                x=pl, line_dash="dash", line_color="white",
                annotation_text=f"PIT L{pl}", annotation_position="top",
            )

        fig_deg.update_layout(
            yaxis_title="Lap Time (seconds)",
            xaxis_title="Lap Number",
            height=420,
            legend=dict(orientation="h", y=1.08),
            margin=dict(l=10, r=10, t=50, b=40),
        )
        st.plotly_chart(fig_deg, use_container_width=True)
        st.caption("Band shows p10–p90 uncertainty. Dashed lines = pit stops. Rising curves = tire degradation.")

    except Exception as e:
        st.warning(f"Could not load degradation curve: {e}")

    # ── Strategy Comparison Table (Deltas) ──
    st.subheader("All Strategies — Delta from Best")

    rows = []
    for s in all_strats:
        stints = s["strategy"]["stints"]
        compounds = s["strategy"]["compounds"]
        pit_laps = _pit_laps_from_stints(stints)
        delta = s["p50"] - best_p50

        rows.append({
            "Strategy": _strategy_label(compounds, stints),
            "Pit Laps": ", ".join([f"L{p}" for p in pit_laps]) if pit_laps else "—",
            "Stops": s["strategy"]["num_stops"],
            "Race Time": _fmt_time(s["p50"]),
            "Delta": _delta_str(delta),
            "Delta (s)": round(delta, 2),
            "Std Dev": round(s["std_time"], 1),
            "Objective": round(s["objective_score"], 1),
        })

    strat_df = pd.DataFrame(rows).sort_values("Delta (s)")
    strat_df.index = range(1, len(strat_df) + 1)
    strat_df.index.name = "#"

    st.dataframe(
        strat_df[["Strategy", "Pit Laps", "Stops", "Race Time", "Delta", "Std Dev"]].style.map(
            lambda v: "color: #2ecc71" if v == "BEST" else
                      "color: #e74c3c" if isinstance(v, str) and v.startswith("+") and float(v.replace("s","").replace("+","")) > 5 else
                      "",
            subset=["Delta"],
        ),
        use_container_width=True,
        height=400,
    )

    # ── Pit Window Analysis ──
    st.subheader("Pit Window Flexibility")

    # Show range of pit laps across top strategies
    top_n = min(15, len(all_strats))
    pit_window_data = []
    for s in sorted(all_strats, key=lambda x: x["p50"])[:top_n]:
        stints = s["strategy"]["stints"]
        compounds = s["strategy"]["compounds"]
        pit_laps = _pit_laps_from_stints(stints)
        for i, pl in enumerate(pit_laps):
            pit_window_data.append({
                "Strategy": _strategy_label(compounds, stints),
                "Stop": i + 1,
                "Pit Lap": pl,
                "p50": s["p50"],
            })

    if pit_window_data:
        pw_df = pd.DataFrame(pit_window_data)
        fig_pw = px.strip(
            pw_df, x="Pit Lap", y="Stop", color="Stop",
            hover_data=["Strategy", "p50"],
            title="Pit Stop Windows (Top 15 Strategies)",
        )
        fig_pw.update_layout(
            xaxis_title="Lap Number",
            yaxis_title="Stop #",
            height=300,
            margin=dict(l=10, r=10, t=40, b=40),
        )
        st.plotly_chart(fig_pw, use_container_width=True)
        st.caption("Each dot is a pit stop from a top strategy. Clusters show the optimal pit window.")


# ===========================================================================
# TAB 2: SAFETY CAR WHAT-IF
# ===========================================================================
elif tab_selection == "Safety Car What-If":
    st.header(f"Safety Car Scenario — {event_name} {year}")

    st.markdown("""
    Simulate what happens if a Safety Car is deployed during the race.
    Compare **pit under SC** (reduced pit loss ~12s) vs **stay out** on current strategy.
    """)

    # Strategy input
    col_a, col_b = st.columns(2)
    with col_a:
        stints_input = st.text_input("Current Strategy — Stints", "20,37")
        compounds_input = st.text_input("Current Strategy — Compounds", "MEDIUM,HARD")
    with col_b:
        sc_lap = st.slider("Safety Car Deployed on Lap", 1, race_laps - 5, race_laps // 3)
        sc_sims = st.number_input("Simulations", 100, 1000, 300, 100)

    try:
        stints_list = [int(x.strip()) for x in stints_input.split(",")]
        compounds_list = [x.strip().upper() for x in compounds_input.split(",")]
    except ValueError:
        st.error("Invalid format. Use comma-separated values.")
        st.stop()

    if sum(stints_list) != race_laps:
        st.warning(f"Stints sum to {sum(stints_list)}, but race is {race_laps} laps. Adjust stints to match.")

    if st.button("Analyze Safety Car Scenario", type="primary", use_container_width=True):
        payload = {
            "year": year,
            "round": round_number,
            "driver_id": driver_id,
            "race_laps": race_laps,
            "stints": stints_list,
            "compounds": compounds_list,
            "sc_lap": sc_lap,
            "num_simulations": sc_sims,
            "event_name": event_name,
        }
        with st.spinner("Simulating SC scenario..."):
            try:
                sc_result = api.simulate_safety_car(payload)
            except Exception as e:
                st.error(f"SC simulation failed: {e}")
                st.stop()
        st.session_state["sc_result"] = sc_result

    if "sc_result" in st.session_state:
        sc = st.session_state["sc_result"]

        # Recommendation banner
        rec = sc["recommendation"]
        delta = sc["time_delta"]
        if rec == "PIT":
            st.success(f"**RECOMMENDATION: PIT UNDER SAFETY CAR** — saves {abs(delta):.1f}s")
        else:
            st.warning(f"**RECOMMENDATION: STAY OUT** — pitting costs {abs(delta):.1f}s")

        st.divider()

        # Side-by-side comparison
        col_stay, col_pit = st.columns(2)

        stay = sc["stay_out"]
        pit = sc["pit_under_sc"]

        with col_stay:
            st.markdown("### Stay Out")
            stay_strat = stay["strategy"]
            st.markdown(f"**Strategy:** {_strategy_label(stay_strat['compounds'], stay_strat['stints'])}")
            st.metric("Race Time (p50)", _fmt_time(stay["p50"]))
            st.metric("Consistency", f"±{stay['std_time']:.1f}s")

        with col_pit:
            st.markdown("### Pit Under SC")
            pit_strat = pit["strategy"]
            st.markdown(f"**Strategy:** {_strategy_label(pit_strat['compounds'], pit_strat['stints'])}")
            st.metric("Race Time (p50)", _fmt_time(pit["p50"]), delta=f"{-delta:.1f}s" if delta > 0 else f"+{abs(delta):.1f}s")
            st.metric("Consistency", f"±{pit['std_time']:.1f}s")

        # Visual comparison
        fig_sc = go.Figure()
        for label, data, color in [
            ("Stay Out", stay, "#e74c3c"),
            ("Pit Under SC", pit, "#2ecc71"),
        ]:
            fig_sc.add_trace(go.Bar(
                name=label,
                x=[label],
                y=[data["p50"]],
                marker_color=color,
                text=[_fmt_time(data["p50"])],
                textposition="outside",
                width=0.4,
            ))

        fig_sc.update_layout(
            yaxis_title="Race Time (seconds)",
            height=350,
            showlegend=False,
            margin=dict(l=10, r=10, t=30, b=40),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        # SC lap context
        st.info(f"SC deployed on **Lap {sc['sc_lap']}** of {race_laps}. "
                f"Remaining: {race_laps - sc['sc_lap']} laps. "
                f"Pit loss under SC: ~12s (vs ~22s normal).")


# ===========================================================================
# TAB 3: RACE ANALYSIS
# ===========================================================================
elif tab_selection == "Race Analysis":
    st.header(f"Race Analysis — {event_name or f'Round {round_number}'} {year}")

    try:
        analysis = api.get_race_analysis(year, round_number)
    except Exception as e:
        st.error(f"Could not load race analysis: {e}")
        st.stop()

    race_info = analysis["race"]
    tm = analysis.get("track_metric")

    c1, c2, c3 = st.columns(3)
    c1.metric("Event", race_info["event_name"])
    c2.metric("Laps", race_info.get("total_laps", "—"))
    c3.metric("Pit Loss", f"{tm['avg_pit_loss']:.1f}s" if tm and tm.get("avg_pit_loss") else "—")

    all_drivers = analysis.get("drivers", [])
    if not all_drivers:
        st.info("No lap data found for this race.")
        st.stop()

    # Driver selector for analysis
    driver_codes = [d["driver_code"] for d in all_drivers]
    selected_drivers = st.multiselect(
        "Select drivers to analyze",
        driver_codes,
        default=driver_codes[:min(5, len(driver_codes))],
    )

    st.divider()

    # ── Lap Time Overlay ──
    st.subheader("Lap Time Comparison")

    fig_laps = go.Figure()
    colors = px.colors.qualitative.Bold
    for i, drv_data in enumerate(all_drivers):
        if drv_data["driver_code"] not in selected_drivers:
            continue
        laps = drv_data["laps"]
        valid = [l for l in laps if l["lap_time_seconds"] and l["lap_time_seconds"] > 0
                 and not l.get("is_pit_out_lap")]
        if not valid:
            continue

        fig_laps.add_trace(go.Scatter(
            x=[l["lap_number"] for l in valid],
            y=[l["lap_time_seconds"] for l in valid],
            mode="lines",
            name=drv_data["driver_code"],
            line=dict(width=1.5, color=colors[i % len(colors)]),
        ))

    fig_laps.update_layout(
        yaxis_title="Lap Time (seconds)",
        xaxis_title="Lap Number",
        height=450,
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=10, r=10, t=40, b=40),
    )
    st.plotly_chart(fig_laps, use_container_width=True)

    # ── Per-Driver Stint Analysis ──
    st.subheader("Stint Breakdown")

    for drv_data in all_drivers:
        if drv_data["driver_code"] not in selected_drivers:
            continue

        with st.expander(
            f"{drv_data['driver_code']} — Best: {_fmt_time(drv_data['best_lap'])} | "
            f"Avg: {_fmt_time(drv_data['avg_lap'])} | {drv_data['total_laps']} laps",
            expanded=(drv_data["driver_code"] == selected_drivers[0] if selected_drivers else False),
        ):
            stint_data = drv_data.get("stints", [])
            if stint_data:
                stint_df = pd.DataFrame(stint_data)
                # Color-code by compound
                st.dataframe(
                    stint_df.style.format({"avg_time": "{:.3f}"}).map(
                        lambda v: f"background-color: {COMPOUND_COLORS.get(v, '')}" if isinstance(v, str) and v in COMPOUND_COLORS else "",
                        subset=["compound"] if "compound" in stint_df.columns else [],
                    ),
                    use_container_width=True,
                )

            # Lap-by-lap scatter colored by compound
            laps = drv_data["laps"]
            valid = [l for l in laps if l["lap_time_seconds"] and l["lap_time_seconds"] > 0
                     and not l.get("is_pit_out_lap")]
            if valid:
                lap_df = pd.DataFrame(valid)
                fig_drv = px.scatter(
                    lap_df,
                    x="lap_number",
                    y="lap_time_seconds",
                    color="compound",
                    color_discrete_map=COMPOUND_COLORS,
                    hover_data=["tyre_life"],
                    labels={"lap_time_seconds": "Lap Time (s)", "lap_number": "Lap"},
                )
                fig_drv.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=20, b=30),
                    legend=dict(orientation="h", y=1.08),
                )
                st.plotly_chart(fig_drv, use_container_width=True)


# ===========================================================================
# TAB 4: SEASON FORM
# ===========================================================================
elif tab_selection == "Season Form":
    st.header(f"Season Form — {year}")

    # Use first available event name if none selected (Season Form works for any event)
    form_event = event_name
    if not form_event and races:
        form_event = races[0]["event_name"]
    if not form_event:
        form_event = "Unknown"

    try:
        intel = api.get_pre_race_intelligence(year, form_event)
    except Exception as e:
        st.error(f"Could not load pre-race intelligence: {e}")
        st.stop()

    # ── Driver Season Form (Normalized Per-Race) ──
    st.subheader(f"Driver Performance — {year} Season (Normalized)")
    st.caption(
        "Pace is normalized per-race: for each GP we compute each driver's gap "
        "to the fastest driver at that event, then average across all races. "
        "This eliminates track-length bias (Monza vs Spa vs Monaco)."
    )

    form_data = intel.get("driver_form", [])
    if form_data:
        form_df = pd.DataFrame(form_data)

        # Format the delta column for display
        if "avg_delta_to_race_best" in form_df.columns:
            form_df["Avg Gap to Leader"] = form_df["avg_delta_to_race_best"].apply(
                lambda x: f"+{x:.3f}s" if x is not None else "—"
            )
            form_df["Best Race Gap"] = form_df["best_delta_to_race_best"].apply(
                lambda x: f"+{x:.3f}s" if x is not None else "—"
            )

        display_cols = ["driver_code", "races_completed", "Avg Gap to Leader", "Best Race Gap"]
        available = [c for c in display_cols if c in form_df.columns]

        st.dataframe(
            form_df[available].rename(columns={
                "driver_code": "Driver",
                "races_completed": "Races",
            }),
            use_container_width=True,
            height=500,
        )

        # Bar chart of normalized pace deltas
        if "avg_delta_to_race_best" in form_df.columns and form_df["avg_delta_to_race_best"].notna().any():
            chart_df = form_df.dropna(subset=["avg_delta_to_race_best"]).head(20)
            fig_form = px.bar(
                chart_df,
                x="driver_code",
                y="avg_delta_to_race_best",
                color="avg_delta_to_race_best",
                color_continuous_scale="RdYlGn_r",
                labels={
                    "driver_code": "Driver",
                    "avg_delta_to_race_best": "Avg Gap to Race Best (s)",
                },
                title="Season Pace — Avg Gap to Race Winner's Pace (lower = faster)",
            )
            fig_form.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=40, b=40),
            )
            st.plotly_chart(fig_form, use_container_width=True)
    else:
        st.info("No season form data available.")

    st.divider()

    # ── Track History ──
    st.subheader(f"Track History — {form_event}")

    track_hist = intel.get("track_history", [])
    if track_hist:
        hist_df = pd.DataFrame(track_hist)
        st.dataframe(
            hist_df.rename(columns={
                "year": "Year",
                "round": "Round",
                "event_name": "Event",
                "total_laps": "Laps",
                "avg_pit_loss": "Pit Loss (s)",
            }),
            use_container_width=True,
        )
    else:
        st.info(f"No historical data for {form_event}.")
