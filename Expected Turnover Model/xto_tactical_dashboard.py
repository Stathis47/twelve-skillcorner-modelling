"""
xTO Tactical Dashboard — Ajax Defensive Metrics
=================================================
Professional Streamlit application for interactive analysis of
Expected Turnover (xTO) defensive metrics from SkillCorner tracking data.

Execution from Repository Root:
    streamlit run "Expected Turnover Model/xto_tactical_dashboard.py"

Tabs:
  1. Chain Visualizer  — "Film Room": replay pressing chains as GIFs with Shapley attribution
  2. Player Comparison — "Scouting Room": positional-percentile radar charts (6 OOP Pillars)

Author: Defensive Football Analyst
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mplsoccer import Pitch
from scipy.stats import percentileofscore
from PIL import Image
import json
import os
import glob
import io
import warnings
import base64

warnings.filterwarnings("ignore")

# =====================================================================
# PATHS
# =====================================================================
from pathlib import Path
import sys
import os
from dotenv import load_dotenv

# Standardize path resolutions using Pathlib
WORKSPACE = Path(__file__).resolve().parent

load_dotenv(WORKSPACE / ".env")
env_path = os.getenv("SKILLCORNER_DATA_DIR")

candidate_paths = [
    Path("C:/SkillcornerData/1/2024"),              # Local absolute path (User's PC)
    WORKSPACE.parent / "SkillcornerData/1/2024"     # Relative to repo (Cloned/Supervisor's PC)
]
if env_path:
    candidate_paths.insert(0, Path(env_path))

ROOT_DIR = None
for path in candidate_paths:
    if path.exists():
        ROOT_DIR = path
        break

if ROOT_DIR is None:
    st.error(f"❌ Target Data Directory could not be found.\n\nPlease define `SKILLCORNER_DATA_DIR` in a `.env` file or place the proprietary dataset in one of these locations:\n1. `{candidate_paths[0]}`\n2. `{candidate_paths[1]}`")
    st.stop()

SUBMETRICS_PATH = WORKSPACE / "player_xTurnover_submetrics.parquet"
SHAPLEY_PATH = WORKSPACE / "xTurnover_marginal_contributions_Shapley.parquet"
CHAINS_PATH = WORKSPACE / "xTurnover_chains.parquet"
PHYS_CSV_PATH = WORKSPACE / "physical_explore_output" / "season_physical_summary.csv"

PHYS_DIR = ROOT_DIR / "physical"
TRACKING_DIR = ROOT_DIR / "tracking_parquets"
DYNAMIC_DIR = ROOT_DIR / "dynamic"
META_DIR = ROOT_DIR / "meta"

# =====================================================================
# AJAX THEME
# =====================================================================
AJAX_RED = "#D2122E"
AJAX_BLACK = "#1A1A2E"
AJAX_WHITE = "#FAFAFA"
AJAX_GOLD = "#C5A258"
AJAX_DARK_BG = "#0F0F1A"
AJAX_CARD_BG = "#16213E"

# =====================================================================
# PAGE CONFIG
# =====================================================================
# Your custom base64 icon
ICON_B64 = "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"
icon_bytes = base64.b64decode(ICON_B64)
custom_icon = Image.open(io.BytesIO(icon_bytes))

st.set_page_config(
    page_title="xTO Tactical Dashboard",
    page_icon=custom_icon,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject Ajax-themed CSS
st.markdown(f"""
<style>
    /* Main background */
    .stApp {{
        background-color: {AJAX_DARK_BG};
        color: {AJAX_WHITE};
    }}
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {AJAX_CARD_BG};
    }}
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {AJAX_CARD_BG};
        color: {AJAX_WHITE};
        border-radius: 6px 6px 0 0;
        padding: 10px 24px;
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {AJAX_RED};
        color: white;
    }}
    /* Metric cards */
    [data-testid="stMetric"] {{
        background-color: {AJAX_CARD_BG};
        border: 1px solid {AJAX_RED}33;
        border-radius: 8px;
        padding: 12px 16px;
    }}
    [data-testid="stMetricLabel"] {{
        color: {AJAX_GOLD};
    }}
    [data-testid="stMetricValue"] {{
        color: {AJAX_WHITE};
    }}
    /* Selectboxes */
    .stSelectbox label, .stMultiSelect label {{
        color: {AJAX_GOLD} !important;
        font-weight: 600;
    }}
    /* Buttons */
    .stButton > button {{
        background-color: {AJAX_RED};
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 2rem;
    }}
    .stButton > button:hover {{
        background-color: #B01025;
        color: white;
    }}
    /* Headers */
    h1, h2, h3 {{
        color: {AJAX_WHITE} !important;
    }}
    /* Dividers */
    hr {{
        border-color: {AJAX_RED}44;
    }}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# DATA LOADING (cached)
# =====================================================================
@st.cache_data(show_spinner="Loading xTO sub-metrics…")
def load_submetrics():
    if not SUBMETRICS_PATH.exists():
        return None
    return pd.read_parquet(SUBMETRICS_PATH)


@st.cache_data(show_spinner="Loading Shapley marginal contributions…")
def load_shapley_data():
    if not SHAPLEY_PATH.exists():
        return None
    df = pd.read_parquet(SHAPLEY_PATH)
    df["match_id"] = df["match_id"].astype(str)
    df["pressing_chain_index"] = df["pressing_chain_index"].astype(str)
    return df


@st.cache_data(show_spinner="Loading chain-level data…")
def load_chains():
    if not CHAINS_PATH.exists():
        return None
    df = pd.read_parquet(CHAINS_PATH)
    df["match_id"] = df["match_id"].astype(str)
    df["pressing_chain_index"] = df["pressing_chain_index"].astype(str)
    return df


@st.cache_data(show_spinner="Recalculating player positions...")
def load_physical():
    phys_files = list(PHYS_DIR.glob("*.parquet"))
    if not phys_files:
        return None
        
    # 1. Load ALL matches for ALL players
    all_phys = pd.concat(
        [pd.read_parquet(f, columns=["player_id", "position_group"]) for f in phys_files],
        ignore_index=True,
    )
    
    # 2. Find the most frequent position for each player (The Mode)
    primary_positions = (
        all_phys.groupby("player_id")["position_group"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )
    
    return primary_positions


@st.cache_data(show_spinner=False)
def get_available_match_ids():
    """Discover matches that have tracking + dynamic + meta."""
    track_files = list(TRACKING_DIR.glob("*.parquet"))
    ids = []
    for f in track_files:
        mid = f.stem
        if (DYNAMIC_DIR / f"{mid}.parquet").exists() and (META_DIR / f"{mid}.json").exists():
            ids.append(mid)
    return sorted(ids)


@st.cache_data(show_spinner="Loading match tracking data…")
def load_match_tracking(match_id: str):
    path = TRACKING_DIR / f"{match_id}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading match dynamic data…")
def load_match_dynamic(match_id: str):
    path = DYNAMIC_DIR / f"{match_id}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading match metadata…")
def load_match_meta(match_id: str):
    path = META_DIR / f"{match_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================================
# GIF GENERATOR  (ported from visualize_marginal_xTurnover_gif.py)
# =====================================================================
MAX_SPEED_VIS = 7.0
BUFFER_FRAMES = 25
GIF_FPS = 10


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert a CSS hex color string (e.g. '#c10021') to a (R, G, B) tuple in [0,1] range."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def _kit_colors_from_meta(meta: dict, home_team_id, away_team_id) -> dict:
    """
    Returns {team_id: {'fill': hex, 'number': hex}} for both teams.
    Falls back to neutral defaults if the kit field is missing.
    """
    result = {}
    for side, team_id, fallback_fill, fallback_num in [
        ("home", home_team_id, "#4169E1", "#FFFFFF"),
        ("away", away_team_id, "#FF4444", "#FFFFFF"),
    ]:
        kit = meta.get(f"{side}_team_kit", {})
        result[team_id] = {
            "fill": kit.get("jersey_color", fallback_fill),
            "number": kit.get("number_color", fallback_num),
        }
    return result


def generate_chain_gif(chain_players_df, tracking_df, dynamic_df, meta,
                       downsample: int = 2) -> bytes | None:
    """
    Render a pressing-chain GIF and return raw bytes.

    Optimisation summary (vs. naive per-frame approach):
      1. Pre-index tracking data by frame (O(1) lookup vs. O(n) scan each frame).
      2. Pre-classify player colour categories outside the frame loop.
      3. Create figure & draw pitch ONCE; reuse across frames via dynamic-artist tracking.
      4. Vectorised scatter — one ax.scatter() call per colour group, not per player.
      5. Single ax.quiver() call for all velocity arrows (replaces N FancyArrowPatch).
      6. numpy buffer capture (fig.canvas.tostring_rgb) instead of PNG encode/decode cycle.
    """
    chain_idx = str(chain_players_df["pressing_chain_index"].iloc[0])
    xT_pred = chain_players_df["xTurnover_full_calibrated"].iloc[0]
    chain_success = int(chain_players_df["chain_success"].iloc[0])

    player_marginal = {
        int(pid): val for pid, val in
        zip(chain_players_df["player_id"], chain_players_df["marginal_xTurnover_calibrated"])
    }
    player_names = {
        int(pid): name for pid, name in
        zip(chain_players_df["player_id"], chain_players_df["player_name"])
    }
    player_share = (
        {int(pid): val for pid, val in
         zip(chain_players_df["player_id"], chain_players_df["contribution_share"])}
        if "contribution_share" in chain_players_df.columns else None
    )

    # ── Frame range ──────────────────────────────────────────────────────────
    # 1. Replicate notebook logic: Fill missing chain IDs with the event_id for solo presses
    dynamic_df["clean_chain_id"] = dynamic_df["pressing_chain_index"].fillna(dynamic_df["event_id"]).astype(str)

    chain_events = dynamic_df[
        (dynamic_df["event_type"] == "on_ball_engagement")
        & (dynamic_df["clean_chain_id"] == chain_idx)
    ]
    
    if chain_events.empty:
        return None

    # 2. Get exact start of first action and end of last action
    first_start = int(chain_events["frame_start"].min())
    last_end = int(chain_events["frame_end"].max())
    raw_peak = int(chain_events["frame_start"].max()) # Used for finding the active pressing team

    # 3. Add BUFFER_FRAMES seconds (BUFFER_FRAMES frames) before the start and AFTER the end
    avail = tracking_df["frame"].unique()
    f_start = max(first_start - BUFFER_FRAMES, int(avail.min()))
    f_end   = min(last_end + BUFFER_FRAMES, int(avail.max()))

    all_frames: list[int] = list(range(f_start, f_end + 1))
    if downsample > 1:
        all_frames = all_frames[::downsample]
    # Snap peak to the nearest rendered frame
    peak_frame = min(all_frames, key=lambda f: abs(f - raw_peak))

    # ── Metadata helpers ─────────────────────────────────────────────────────
    player_jerseys: dict[int, str] = {}
    for p in meta.get("players", []):
        if p.get("number") is not None:
            player_jerseys[p["id"]] = str(p["number"])

    pressing_events = dynamic_df[
        (dynamic_df["frame_start"] <= raw_peak)
        & (dynamic_df["frame_end"]   >= raw_peak)
        & (dynamic_df["clean_chain_id"] == chain_idx)
    ]
    pressing_team = (
        int(pressing_events["team_id"].iloc[0]) if not pressing_events.empty else None
    )
    
    # Get IDs and Names from metadata
    home_id = meta.get("home_team", {}).get("id")
    away_id = meta.get("away_team", {}).get("id")
    home_name = meta.get("home_team", {}).get("name", "Home Team")
    away_name = meta.get("away_team", {}).get("name", "Away Team")
    
    # Assign actual names based on who is pressing
    pressing_team_name = home_name if pressing_team == home_id else away_name
    opponent_team_name = away_name if pressing_team == home_id else home_name

    kit_colors   = _kit_colors_from_meta(meta, home_id, away_id)
    opponent_team = away_id if pressing_team == home_id else home_id
    pressing_kit  = kit_colors.get(pressing_team, {"fill": "#4169E1", "number": "#FFFFFF"})
    opp_kit       = kit_colors.get(opponent_team, {"fill": "#FF4444", "number": "#FFFFFF"})

    # ── OPTIMISATION 1: Pre-filter & index frames ────────────────────────────
    frames_needed = set(all_frames)
    tracking_sub = tracking_df[tracking_df["frame"].isin(frames_needed)].copy()
    tracking_sub = tracking_sub.sort_values(["player_id", "frame"])
    # Calculate exact time gap between rows (solves the giant arrow bug in Fast/Normal mode)
    frame_diff = tracking_sub.groupby("player_id")["frame"].diff().fillna(downsample)
    dt = frame_diff / 10.0  # Convert frames to seconds (10 Hz tracking)
    tracking_sub["vx"] = tracking_sub.groupby("player_id")["x"].diff().fillna(0) / dt
    tracking_sub["vy"] = tracking_sub.groupby("player_id")["y"].diff().fillna(0) / dt
    _spd = np.sqrt(tracking_sub["vx"] ** 2 + tracking_sub["vy"] ** 2)
    _over = _spd > 12.0
    if _over.any():
        _sc = 12.0 / _spd[_over]
        tracking_sub.loc[_over, "vx"] *= _sc
        tracking_sub.loc[_over, "vy"] *= _sc
    frame_dict: dict[int, pd.DataFrame] = {
        int(f): grp for f, grp in tracking_sub.groupby("frame")
    }

    # ── OPTIMISATION 2: Pre-classify player colour categories ────────────────
    player_cat: dict[int, tuple] = {}
    pid_tid_map = (
        tracking_sub[~tracking_sub["is_ball"]][["player_id", "team_id"]]
        .drop_duplicates("player_id")
        .set_index("player_id")["team_id"]
        .to_dict()
    )
    for _pid, _tid in pid_tid_map.items():
        _pid = int(_pid)
        is_p = _pid in player_marginal
        if pressing_team is not None and pd.notna(_tid) and int(_tid) == pressing_team:
            # Chain players get a larger dot; colour is always the team kit
            sz = 650 if is_p else 450
            color, edge, txt_c = pressing_kit["fill"], "white" if is_p else "black", pressing_kit["number"]
        else:
            color, edge, txt_c, sz = opp_kit["fill"], "black", opp_kit["number"], 450
        player_cat[_pid] = (color, edge, txt_c, sz)

    # ── OPTIMISATION 3: Create figure & draw pitch ONCE ──────────────────────
    p_len  = meta.get("pitch_length", 105)
    p_wid  = meta.get("pitch_width",  68)
    fig_w  = 14
    fig_h  = fig_w * (p_wid / p_len)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=90)
    fig.patch.set_facecolor("#1a1a1a")
    pitch = Pitch(
        pitch_type="skillcorner", pitch_length=p_len, pitch_width=p_wid,
        pitch_color="#22312b", line_color="white", linewidth=2,
        goal_type="box", corner_arcs=True,
    )
    pitch.draw(ax=ax)

    # NEW: Add a placeholder title to force Matplotlib to reserve the top margin space!
    ax.set_title("Placeholder Title Reserve Space", fontsize=20, fontweight="bold", pad=20)

    fig.tight_layout()          # called once; layout does not change between frames

    legend_els = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=pressing_kit["fill"],
                   markersize=13, markeredgecolor="white", markeredgewidth=1.5,
                   label="Chain Player(s)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=pressing_kit["fill"],
                   markersize=9, markeredgecolor="black", markeredgewidth=1,
                   label=f"{pressing_team_name}"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=opp_kit["fill"],
                   markersize=9, markeredgecolor="black", markeredgewidth=1,
                   label=f"{opponent_team_name}"),
    ]
    outcome = "Success" if chain_success == 1 else "Failure"

    frames_images: list[Image.Image] = []
    _dyn: list = []     # dynamic artists to remove before next frame

    for frame_num in all_frames:
        # Remove previous frame's artists (pitch lines/patches remain untouched)
        for h in _dyn:
            try:
                h.remove()
            except Exception:
                pass
        _dyn = []

        fdata = frame_dict.get(frame_num)
        if fdata is None or len(fdata) < 10:
            continue

        # Ball
        ball = fdata[fdata["is_ball"]]
        if not ball.empty:
            bx, by = float(ball.iloc[0]["x"]), float(ball.iloc[0]["y"])
            _dyn.append(ax.scatter(bx, by, s=300, c="white", edgecolors="black",
                                   linewidths=2, zorder=10))

        # ── OPTIMISATION 4+5: Build per-category arrays + per-player text/arrow data
        players_fdata = fdata[~fdata["is_ball"]]
        cat_arrays: dict[tuple, dict] = {}
        per_player: list[tuple] = []   # (x, y, pid, txt_c, mxt, is_presser, vx, vy)

        for _, row in players_fdata.iterrows():
            pid = int(row["player_id"])
            cat = player_cat.get(pid)
            if cat is None:
                _tid = row.get("team_id")
                if pressing_team is not None and pd.notna(_tid) and int(_tid) == pressing_team:
                    is_p = pid in player_marginal
                    sz = 650 if is_p else 450
                    edge = "white" if is_p else "black"
                    cat = (pressing_kit["fill"], edge, pressing_kit["number"], sz)
                else:
                    cat = (opp_kit["fill"], "black", opp_kit["number"], 500)
                player_cat[pid] = cat
            color, edge, txt_c, sz = cat
            key = (color, edge, sz)
            if key not in cat_arrays:
                cat_arrays[key] = {"x": [], "y": [], "txt_c": txt_c}
            cat_arrays[key]["x"].append(float(row["x"]))
            cat_arrays[key]["y"].append(float(row["y"]))
            per_player.append((
                float(row["x"]), float(row["y"]), pid, txt_c,
                player_marginal.get(pid, 0.0), pid in player_marginal,
                float(row.get("vx", 0)), float(row.get("vy", 0)),
            ))

        # One scatter call per colour group (replaces N individual scatter calls)
        for (color, edge, sz), arrays in cat_arrays.items():
            _dyn.append(ax.scatter(arrays["x"], arrays["y"], s=sz, c=color,
                                   edgecolors=edge, linewidths=2, zorder=5, alpha=1.0))

        # Per-player jersey number + xTO label (text must be per-item)
        for px, py, pid, txt_c, mxt, is_presser, vx, vy in per_player:
            jn = player_jerseys.get(pid, "")
            if jn:
                _dyn.append(ax.text(px, py, jn, fontsize=10, fontweight="bold",
                                    color=txt_c, ha="center", va="center", zorder=6))
            if is_presser and mxt > 0:
                _dyn.append(ax.text(
                    px, py + 3.5, f"{mxt:.3f}", fontsize=9, fontweight="bold",
                    color="white", ha="center", va="bottom", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black",
                              edgecolor="white", alpha=0.8, linewidth=1.5),
                ))

        # OPTIMISATION 5: single ax.quiver() for all velocity arrows
        arr_x, arr_y, arr_u, arr_v = [], [], [], []
        for px, py, pid, txt_c, mxt, is_presser, vx, vy in per_player:
            spd = np.sqrt(vx ** 2 + vy ** 2)
            if spd > 0.5:
                sc_f = 0.8 * (spd / MAX_SPEED_VIS)
                arr_x.append(px);       arr_y.append(py)
                arr_u.append(vx * sc_f); arr_v.append(vy * sc_f)
        if arr_x:
            _dyn.append(ax.quiver(
                arr_x, arr_y, arr_u, arr_v,
                color="white", alpha=0.5, zorder=4,
                angles="xy", scale_units="xy", scale=1,
                width=0.003, headwidth=4, headlength=5,
            ))

        # Timestamp
        period_info = next(
            (p for p in meta.get("match_periods", [])
             if p["start_frame"] <= frame_num <= p["end_frame"]), None
        )
        if period_info:
            pn      = period_info.get("period", 1)
            base_m  = {1: 0, 2: 45, 3: 90, 4: 105}.get(pn, (pn - 1) * 45)
            total_s = base_m * 60 + (frame_num - period_info["start_frame"]) / 10.0
            ts = f"{int(total_s // 60):02d}:{int(total_s % 60):02d}"
        else:
            ts = f"{int(frame_num / 10.0 // 60):02d}:{int(frame_num / 10.0 % 60):02d}"

        # set_title replaces in-place — no handle needed
        ax.set_title(f"xTO: {xT_pred:.3f}  |  {outcome}  |  {ts}",
                     fontsize=16, fontweight="bold", color="white", pad=14)

        _dyn.append(ax.legend(handles=legend_els, loc="upper left", fontsize=8,
                              framealpha=0.2, labelcolor="white", facecolor="black"))

        if frame_num == peak_frame:
            txt = "Shapley Contributions:\n"
            for pid_, v_ in sorted(player_marginal.items(), key=lambda x: x[1], reverse=True)[:5]:
                txt += f"  {player_names.get(pid_, '?')}: {v_:.3f}\n"
            _dyn.append(ax.text(
                0.98, 0.02, txt, transform=ax.transAxes, fontsize=8, color="white",
                ha="right", va="bottom", zorder=8,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="black",
                          edgecolor="white", alpha=0.7),
            ))

        # OPTIMISATION 6: numpy buffer capture (eliminates PNG encode + PIL decode)
        # buffer_rgba() is the correct API for matplotlib >= 3.8 (tostring_rgb removed)
        fig.canvas.draw()
        fw, fh = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fh, fw, 4)
        frames_images.append(Image.fromarray(buf[:, :, :3], mode="RGB").copy())

    # Tear down
    for h in _dyn:
        try:
            h.remove()
        except Exception:
            pass
    plt.close(fig)

    if not frames_images:
        return None

    out = io.BytesIO()
    frame_duration = (1000 // GIF_FPS) * downsample
    frames_images[0].save(
        out, format="GIF", save_all=True, append_images=frames_images[1:],
        duration=frame_duration, loop=0,
    )
    return out.getvalue()


# =====================================================================
# RADAR CHART HELPER  (ported from slide2_pressing_efficiency_dominance)
# =====================================================================
PILLARS = [
    ("chains_per_90",          "Volume\n(Chains/90)"),
    ("xTurnover_per_chain",          "Efficiency\n(xTO/Chain)"),
    ("solo_xTurnover_ratio",         "Self-Sufficiency\n(Solo Ratio)"),
    ("avg_contribution_share", "Dominance\n(Contribution %)"),
    ("avg_chain_xTurnover_full",      "Tactical IQ\n(Chain Danger)"),
    ("negative_impact_per_90",    "Shape Discipline\n(Negative xTO/90)"),
    ("pressing_risk",    "Pressing Risk Ratio\n(xT conceded per 1000 xTO generated)"),
    #("defensive_penalty_per_90", "Pressing Risk\n(Defensive Penalty/90)"),
]
N_PILLARS = len(PILLARS)
ANGLES = np.linspace(0, 2 * np.pi, N_PILLARS, endpoint=False).tolist()
ANGLES_CLOSED = ANGLES + ANGLES[:1]


def _draw_radar(ax, values, color, label, alpha_fill=0.20):
    v = values + values[:1]
    ax.plot(ANGLES_CLOSED, v, color=color, lw=0.5, zorder=4, label=label)
    ax.fill(ANGLES_CLOSED, v, color=color, alpha=alpha_fill, zorder=3)
    ax.scatter(ANGLES, values, s=5, color=color, zorder=5, edgecolors="white", linewidths=0.3)


def _style_radar(ax, title, title_color):
    # 1. Match the dark background of the app
    ax.set_facecolor(AJAX_DARK_BG)  
    ax.set_ylim(0, 105)
    ax.set_xticks(ANGLES)
    
    # 2. Change text color to white so it's visible on the dark background
    ax.set_xticklabels([p[1] for p in PILLARS], fontsize=6, fontweight="bold", color=AJAX_WHITE)
    ax.tick_params(axis='x', pad=8)
    
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25th", "50th", "75th", "100th"], fontsize=3.5, color="#AAA")
    ax.set_rlabel_position(0)
    
    # 3. Dim the grid rings and spokes to a subtle dark grey
    for lvl in [25, 50, 75, 100]:
        ax.plot(ANGLES_CLOSED, [lvl] * (N_PILLARS + 1), color="#444444", lw=0.8, ls="--", zorder=1)
    for a in ANGLES:
        ax.plot([a, a], [0, 100], color="#444444", lw=0.8, zorder=1)
        
    ax.set_title(title, fontsize=13, fontweight="bold", color=title_color, pad=20, va="bottom")
    ax.spines["polar"].set_visible(False)
    ax.grid(False)


def build_radar_figure(players_info, peer_df):
    """
    Build a single radar chart with positional percentiles for up to 3 players.
    ``players_info`` is a list of tuples: (row, name, color).
    ``peer_df`` is the position-filtered DataFrame used as the percentile reference pool.
    Returns a matplotlib Figure.
    """
    pillar_cols = [p[0] for p in PILLARS]

    # 1. Define which metrics need to be inverted (Lower = Better)
    lower_is_better = ["pressing_risk", "defensive_penalty_per_90"]

    def pct_vals(row):
        out = []
        for col in pillar_cols:
            ref = peer_df[col].dropna().values
            v = row[col] if pd.notna(row[col]) else 0

            # Calculate standard percentile
            pct = percentileofscore(ref, v, kind="rank")
            # 2. Invert the percentile if the metric is in our list
            if col in lower_is_better:
                pct = 100 - pct

            out.append(pct)
        return out

    fig = plt.figure(figsize=(5.5, 4.8), dpi=90)
    fig.patch.set_facecolor(AJAX_DARK_BG)

    ax = fig.add_subplot(111, projection="polar")

    vals_list = []
    legend_handles = []
    
    for row, name, color in players_info:
        vals = pct_vals(row)
        vals_list.append(vals)
        
        _draw_radar(ax, vals, color, name)
        
        xto = row.get("xTurnover_per_90", 0) if "xTurnover_per_90" in row.index else row.get("marginal_xTurnover_per_90", 0)
        team = row.get("team_name", "")
        
        # create proxy artist for legend
        line = plt.Line2D([0], [0], color=color, lw=1, label=f"{name} ({xto:.3f} xTO/90, {team})")
        legend_handles.append(line)

    _style_radar(ax, "", AJAX_WHITE)

    fig.text(0.5, 0.95,
             f"The {len(PILLARS)} Pillars of Out-of-Possession Impact — Pressing DNA",
             ha="center", va="top", fontsize=12, fontweight="bold", color=AJAX_WHITE)

    # Create a legend with text colors matching the lines
    leg = ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=1, frameon=False, fontsize=6)
    for text, handle in zip(leg.get_texts(), leg.legend_handles):
        text.set_color(handle.get_color())
        text.set_fontweight("bold")

    fig.subplots_adjust(left=0.08, right=0.92, top=0.80, bottom=0.28)
    return fig, vals_list


# =====================================================================
# HEADER
# =====================================================================
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:14px;padding-bottom:6px;">
        <div style="font-size:42px;font-weight:800;color:{AJAX_RED};letter-spacing:-1px;">
            xTO Tactical Dashboard
        </div>
        <div style="font-size:14px;color:#888;padding-top:12px;">
            Ajax Defensive Metrics &mdash; SkillCorner 2024
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# =====================================================================
# LOAD ALL DATA
# =====================================================================
submetrics = load_submetrics()
shapley_df = load_shapley_data()
chains_df = load_chains()
phys_df = load_physical()

missing = []
if submetrics is None:
    missing.append("player_xTurnover_submetrics.parquet")
if shapley_df is None:
    missing.append("xTurnover_marginal_contributions_Shapley.parquet")
if chains_df is None:
    missing.append("xTurnover_chains.parquet")
if missing:
    st.error(f"Missing data files: {', '.join(missing)}. Run the xTurnover pipeline first.")
    st.stop()


# =====================================================================
# TABS
# =====================================================================
tab_film, tab_scout = st.tabs(["🎬  Chain Visualizer  (Film Room)", "🔍  Player Comparison  (Scouting Room)"])


# =====================================================================
# TAB 1 — CHAIN VISUALIZER
# =====================================================================
with tab_film:
    st.markdown(f"### 🎬 Pressing Chain Film Room")
    st.caption("Select a match and pressing chain to replay the Shapley-attributed pressing sequence as a GIF.")

    all_match_ids = get_available_match_ids()
    # Filter to matches that are also in Shapley data
    shapley_matches = set(shapley_df["match_id"].unique())
    available_matches = [m for m in all_match_ids if m in shapley_matches]

    if not available_matches:
        st.warning("No matches with both tracking data and Shapley contributions found.")
    else:
        # Match selector — build display labels from metadata
        @st.cache_data(show_spinner=False)
        def build_match_labels(match_ids):
            labels = {}
            for mid in match_ids:
                meta = load_match_meta(mid)
                if meta:
                    home = meta.get("home_team", {}).get("short_name", "?")
                    away = meta.get("away_team", {}).get("short_name", "?")
                    date = meta.get("date_time", "")[:10]
                    labels[mid] = f"{home} vs {away}  ({date})"
                else:
                    labels[mid] = mid
            return labels

        match_labels = build_match_labels(tuple(available_matches))

        col_match, col_chain = st.columns([1, 1])

        with col_match:
            selected_label = st.selectbox(
                "Match",
                options=available_matches,
                format_func=lambda x: match_labels.get(x, x),
            )
        selected_match = selected_label

        # Chains for this match
        match_shapley = shapley_df[shapley_df["match_id"] == selected_match]
        chain_ids = match_shapley["global_chain_id"].unique().tolist()

        # Build chain display labels with xTO and size
        chain_summaries = (
            match_shapley.groupby("global_chain_id")
            .agg(
                pressing_chain_index=("pressing_chain_index", "first"),
                xTO=("xTurnover_full_calibrated", "first"),
                success=("chain_success", "first"),
                size=("chain_size", "first"),
            )
            .sort_values("xTO", ascending=False)
        )

        chain_options = chain_summaries.index.tolist()
        chain_label_map = {}
        for cid in chain_options:
            r = chain_summaries.loc[cid]
            outcome = "✓" if r["success"] == 1 else "✗"
            orig_cid = r["pressing_chain_index"]
            chain_label_map[cid] = f"Chain {orig_cid}  |  xTO: {r['xTO']:.3f}  |  {int(r['size'])}P  |  {outcome}"

        with col_chain:
            selected_chain = st.selectbox(
                "Pressing Chain (sorted by xTO ↓)",
                options=chain_options,
                format_func=lambda x: chain_label_map.get(x, x),
            )

        # Chain summary metrics
        if selected_chain:
            meta = load_match_meta(selected_match)
            player_jerseys = {}
            if meta:
                for p in meta.get("players", []):
                    if p.get("number") is not None:
                        player_jerseys[int(p["id"])] = str(p["number"])

            chain_row = chain_summaries.loc[selected_chain]
            chain_players = match_shapley[match_shapley["global_chain_id"] == selected_chain].copy()
            chain_players["jersey"] = chain_players["player_id"].astype(int).map(player_jerseys).fillna("-")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("xTO (Model Prediction)", f"{chain_row['xTO']:.4f}")
            c2.metric("Outcome", "TURNOVER" if chain_row["success"] == 1 else "No Turnover")
            c3.metric("Chain Size", f"{int(chain_row['size'])} players")
            
            # Defensive Line Height from chains_df if available
            ch_detail = chains_df[
                chains_df["global_chain_id"] == selected_chain
            ]
            dl_height = ch_detail["defensive_line_height"].iloc[0] * 105.0 if (not ch_detail.empty and "defensive_line_height" in ch_detail.columns) else "—"
            dl_height = abs(dl_height) if isinstance(dl_height, (float, np.floating)) else dl_height # Guard catch for pre-existing flipped data
            c4.metric("Def. Line Height", f"{dl_height:.1f} m" if isinstance(dl_height, (float, np.floating)) else dl_height)

            # Player contribution table
            st.markdown("##### Shapley Marginal Contributions")
            disp = chain_players[
                ["jersey", "player_name", "marginal_xTurnover_calibrated", "contribution_share", "temporal_weight"]
            ].copy()
            disp.columns = ["No.", "Player", "Marginal xTO", "Share", "Temporal Weight"]
            disp = disp.sort_values("Marginal xTO", ascending=False).reset_index(drop=True)
            disp.index += 1
            disp["Share"] = disp["Share"].apply(lambda v: f"{v * 100:.1f}%")
            disp["Marginal xTO"] = disp["Marginal xTO"].apply(lambda v: f"{v:.4f}")
            disp["Temporal Weight"] = disp["Temporal Weight"].apply(lambda v: f"{v:.3f}")
            st.dataframe(disp, use_container_width=True, height=min(280, 60 + len(disp) * 38))

            # GIF generation
            st.markdown("---")
            ds_option = st.radio("Frame quality", ["Fast (every 3rd frame)", "Normal (every 2nd frame)", "Full (all frames)"],
                                 horizontal=True, index=0)
            ds_map = {"Fast (every 3rd frame)": 3, "Normal (every 2nd frame)": 2, "Full (all frames)": 1}
            downsample = ds_map[ds_option]

            if st.button("▶  Generate Visualization", type="primary", use_container_width=True):
                with st.spinner("Loading match data & rendering frames… This may take 15–60 seconds."):
                    tracking = load_match_tracking(selected_match)
                    dynamic = load_match_dynamic(selected_match)

                    if tracking is None or dynamic is None or meta is None:
                        st.warning(f"Tracking/dynamic/meta data missing for match {selected_match}.")
                    else:
                        gif_bytes = generate_chain_gif(
                            chain_players, tracking, dynamic, meta,
                            downsample=downsample,
                        )
                        if gif_bytes:
                            st.image(gif_bytes, caption=f"Chain {selected_chain}  |  xTO: {chain_row['xTO']:.3f}",
                                     use_container_width=True)
                        else:
                            st.warning("Could not generate GIF — no valid frames found for this chain.")


# =====================================================================
# TAB 2 — PLAYER COMPARISON
# =====================================================================
with tab_scout:
    st.markdown("### 🔍 Player Pressing DNA Comparison")
    st.caption(f"Compare two players' {len(PILLARS)}-pillar OOP profiles. Percentiles are computed within their shared position group.")

    # Merge position_group onto submetrics
    if phys_df is not None and "position_group" in phys_df.columns:
        sub_with_pos = submetrics.merge(
            phys_df[["player_id", "position_group"]].drop_duplicates("player_id"),
            on="player_id", how="left",
        )
    else:
        sub_with_pos = submetrics.copy()
        sub_with_pos["position_group"] = "Unknown"

    pos_groups = sorted(sub_with_pos["position_group"].dropna().unique().tolist())
    all_teams = sorted(sub_with_pos["team_name"].dropna().unique().tolist())

    st.markdown("##### Filter Players <span style='font-size: 14px; font-weight: normal; color: #888;'> (Optional)</span>", unsafe_allow_html=True)
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        sel_positions = st.multiselect("Filter by Position", pos_groups, placeholder="Select one or more positions...")
    with col_f2:
        sel_teams = st.multiselect("Filter by Team", all_teams, placeholder="Select one or more teams...")

    pool = sub_with_pos.copy()
    if sel_positions:
        pool = pool[pool["position_group"].isin(sel_positions)]
    if sel_teams:
        pool = pool[pool["team_name"].isin(sel_teams)]

    available_players = sorted(pool["player_name"].dropna().unique().tolist())
    
    st.markdown("##### Select Players <span style='font-size: 14px; font-weight: normal; color: #888;'> (Max 5)</span>", unsafe_allow_html=True)
    
    selected_player_names = st.multiselect(
        "Search and select players (up to 5):",
        options=available_players,
        max_selections=5,
        placeholder="Type to search for players..."
    )

    st.markdown("---")

    players_info = []
    pools = []
    pos_labels = []
    COLORS = ["#4169E1", AJAX_RED, "#2E8B57", AJAX_GOLD, "#9370DB"]

    for idx, p_name in enumerate(selected_player_names):
        # Identify player row
        row = pool[pool["player_name"] == p_name].iloc[0]
        
        # Find percentile reference pool specifically for this player's position
        p_pos = row["position_group"]
        p_pool = sub_with_pos[sub_with_pos["position_group"] == p_pos]
        
        players_info.append((row, p_name, COLORS[idx % len(COLORS)]))
        pools.append(p_pool)
        pos_labels.append(p_pos)

    if len(players_info) > 0:
        # Determine shared peer pool
        peer = pd.concat(pools).drop_duplicates("player_id")
        unique_pos = list(dict.fromkeys(pos_labels))
        peer_label = " + ".join(unique_pos)

        st.caption(f"Percentiles relative to **{peer_label}** ({len(peer)} players, ≥900 min)")

        fig, vals_list = build_radar_figure(players_info, peer)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Stats table with raw values + percentiles
        st.markdown("##### Raw Metric Values & Percentiles")
        compare_rows = []
        for i, (col_name, label) in enumerate(PILLARS):
            row_data = {"Pillar": label.replace("\n", " ")}
            
            for p_idx, (row_data_p, name_p, _) in enumerate(players_info):
                v_p = row_data_p[col_name] if pd.notna(row_data_p[col_name]) else 0
                if col_name == "chains_per_90":
                    fmt = "{:.1f}" 
                elif col_name == "pressing_risk":
                    fmt = "{:.6f}"
                else:
                    fmt = "{:.4f}"
                row_data[name_p] = fmt.format(v_p)
                row_data[f"{name_p} Pctile"] = f"{vals_list[p_idx][i]:.0f}th"
            
            compare_rows.append(row_data)
            
        st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)
    
st.markdown("<br><br>", unsafe_allow_html=True)
