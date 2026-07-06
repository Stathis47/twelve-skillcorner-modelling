"""
xTO Tactical Dashboard — Ajax Defensive Metrics
=================================================
Professional Streamlit application for interactive analysis of
Expected Turnover (xTO) defensive metrics from SkillCorner tracking data.

Execution from Repository Root:
    streamlit run "Week 8/xto_tactical_dashboard.py"

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
from matplotlib.patches import FancyArrowPatch, Circle
from mplsoccer import Pitch
from scipy.stats import percentileofscore
from PIL import Image
import json
import base64
import os
import glob
import io
import warnings
import re

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

load_dotenv(WORKSPACE.parent / ".env")
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
# THEME (DEFAULT STREAMLIT LOOK)
# =====================================================================
DEFAULT_BG = "#FFFFFF"
DEFAULT_TEXT = "#000000"

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(page_title="xTO Tactical Dashboard", layout="wide")


# =====================================================================
# DATA LOADING (cached)
# =====================================================================
minutes_threshold = 900
chains_threshold = 40

@st.cache_data(show_spinner="Loading xTO sub-metrics…")
def load_submetrics():
    if not SUBMETRICS_PATH.exists():
        return None
    df = pd.read_parquet(SUBMETRICS_PATH)
    if {"minutes_played", "chains_participated"}.issubset(df.columns):
        df = df[(df["minutes_played"] >= minutes_threshold) & (df["chains_participated"] >= chains_threshold)].copy()
    return df


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
RING_RADIUS = 1.8
RING_TAIL_FRAMES = 5


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

    engagement_windows = (
        chain_events[["player_id", "frame_start", "frame_end"]]
        .dropna()
        .astype({"player_id": int, "frame_start": int, "frame_end": int})
        .itertuples(index=False, name=None)
    )
    engagement_windows = list(engagement_windows)

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
            # Chain players are distinguished by circumference color; size is uniform.
            sz = 450
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
                   markersize=9, markeredgecolor="white", markeredgewidth=1.5,
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
                    sz = 450
                    edge = "white" if is_p else "black"
                    cat = (pressing_kit["fill"], edge, pressing_kit["number"], sz)
                else:
                    cat = (opp_kit["fill"], "black", opp_kit["number"], 450)
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

        # Highlight active engagements with a dotted green ring (1s tail after engagement ends).
        active_pids = {
            pid for pid, start, end in engagement_windows
            if start <= frame_num <= end + RING_TAIL_FRAMES
        }
        for px, py, pid, txt_c, mxt, is_presser, vx, vy in per_player:
            if pid in active_pids:
                ring = Circle(
                    (px, py),
                    RING_RADIUS,
                    fill=False,
                    edgecolor="#00FF00",
                    linewidth=2,
                    linestyle="--",
                    zorder=6,
                )
                ax.add_patch(ring)
                _dyn.append(ring)

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

        # Status bar (mirrors risk/reward scripts)
        status_switch_frame = last_end + 20
        if frame_num < first_start:
            status_txt, status_color = "PRE-PRESS", "#AAAAAA"
        elif frame_num <= last_end:
            status_txt, status_color = "PRESS ACTIVE", "#FFA500"
        else:
            if chain_success == 1:
                status_txt, status_color = "PRESS WON", "#2E8B57"
            else:
                status_txt, status_color = "PRESS FAILED", "#FF4444"

        _dyn.append(ax.text(
            0.5, 0.95, f"STATUS: {status_txt}", transform=ax.transAxes,
            fontsize=14, fontweight="bold", color=status_color,
            ha="center", va="top", zorder=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a",
                      edgecolor=status_color, linewidth=2, alpha=0.8),
        ))

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
# RADAR CHART HELPER  
# =====================================================================
PILLARS = [
    ("chains_per_90",          "Volume\n(Chains/90)"),
    ("xTurnover_per_chain",          "Efficiency\n(xTO/Chain)"),
    ("xTurnover_per_90",             "Quality\n(xTO/90)"),
    #("solo_xTurnover_ratio",         "Self-Sufficiency\n(Solo xTO / Total xTO)"),
    ("median_contribution_share_coord", "Dominance\n(Median \nmarginal xTO / Chain xTO)"),
    ("avg_chain_xTurnover_full",      "Tactical IQ\n(Avg full-chain xTO)"),
    #("negative_impact_per_chain",    "Shape Discipline\n(Negative xTO/Chain)"),
    #("pressing_risk",    "Pressing Risk-Reward Ratio\n(xT conceded/xTO)"),
    ("defensive_penalty_per_100_chains", "Exposure Penalty\n(xT conceded per 100 Chains)"),
    ("xT_generated_per_100_chains", "Attacking Reward\n(xT generated per 100 Chains)"),
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
    ax.set_facecolor(DEFAULT_BG)
    ax.set_ylim(0, 105)
    ax.set_xticks(ANGLES)
    
    # 2. Change text color to white so it's visible on the dark background
    ax.set_xticklabels([p[1] for p in PILLARS], fontsize=6, fontweight="bold", color=DEFAULT_TEXT)
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
    lower_is_better = ["pressing_risk", "defensive_penalty_per_100_chains"]

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

    fig = plt.figure(figsize=(5.04, 4.32), dpi=90)
    fig.patch.set_facecolor(DEFAULT_BG)

    ax = fig.add_subplot(111, projection="polar")

    vals_list = []
    legend_handles = []
    
    for row, name, color in players_info:
        vals = pct_vals(row)
        vals_list.append(vals)
        
        _draw_radar(ax, vals, color, name)
        
        team = row.get("team_name", "")

        # create proxy artist for legend
        line = plt.Line2D([0], [0], color=color, lw=1, label=f"{name} ({team})")
        legend_handles.append(line)

    _style_radar(ax, "", DEFAULT_TEXT)

    fig.text(0.5, 0.95,
             f"Defensive Metrics Radar Chart",
             ha="center", va="top", fontsize=12, fontweight="bold", color=DEFAULT_TEXT)

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
st.title("xTO Tactical Dashboard")
st.caption("Premier League 2024/25")
st.divider()

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

        col_filters, col_content = st.columns([1, 3], gap="large")

        with col_filters:
            st.markdown("##### Filters")
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

            selected_chain = st.selectbox(
                "Pressing Chain (sorted by xTO ↓)",
                options=chain_options,
                format_func=lambda x: chain_label_map.get(x, x),
            )

        with col_content:
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

                c1, c2, c3 = st.columns(3)
                c1.metric("xTO (Model Prediction)", f"{chain_row['xTO']:.4f}")
                c2.metric("Outcome", "TURNOVER" if chain_row["success"] == 1 else "No Turnover")
                c3.metric("Chain Size", f"{chain_row['size']}")

                ch_detail = chains_df[
                    chains_df["global_chain_id"] == selected_chain
                ]

                # Chain feature table (11 model features)
                st.markdown("##### Chain Feature Profile")
                feature_specs = [
                    ("distance_to_goal", "Mean Dist to Goal (m)", "{:.2f}", None),
                    ("defensive_line_height", "Defensive Line Height (m)", "{:.1f}", lambda v: abs(v) * 105.0),
                    ("chain_duration", "Chain Duration (s)", "{:.2f}", None),
                    ("proximity_to_sideline", "Proximity to Sideline (m)", "{:.2f}", None),
                    ("possession_chain_length", "Possession Chain Length", "{:.0f}", None),
                    ("max_radial_velocity", "Max Radial Closing Velocity (m/s)", "{:.3f}", None),
                    ("forward_pressing_ratio", "Forward Pressing Ratio", "{:.2f}", None),                   
                    ("delta_n_passing_options", "Mean Delta Passing Options", "{:.2f}", None),
                    ("local_numerical_superiority", "Mean LNS", "{:.3f}", None),
                    ("defensive_proximity", "Mean Defensive Proximity (m)", "{:.3f}", None),                   
                ]

                chain_feature_rows = []
                if not ch_detail.empty:
                    ch_row = ch_detail.iloc[0]
                    for col, label, fmt, transform in feature_specs:
                        if col not in ch_detail.columns or pd.isna(ch_row.get(col)):
                            value_str = "—"
                        else:
                            val = ch_row[col]
                            if transform is not None:
                                val = transform(val)
                            value_str = fmt.format(val)
                        chain_feature_rows.append({"Feature": label, "Value": value_str})
                else:
                    chain_feature_rows = [{"Feature": label, "Value": "—"} for _, label, _, _ in feature_specs]

                st.dataframe(
                    pd.DataFrame(chain_feature_rows),
                    use_container_width=True,
                    hide_index=True,
                    height=420,
                )

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
                                gif_b64 = base64.b64encode(gif_bytes).decode("ascii")
                                st.markdown(
                                    f"<img src=\"data:image/gif;base64,{gif_b64}\" style=\"width:100%; height:auto;\" />",
                                    unsafe_allow_html=True,
                                )
                                st.caption(f"Chain {selected_chain}  |  xTO: {chain_row['xTO']:.3f}")
                                st.download_button(
                                    label="Download GIF",
                                    data=gif_bytes,
                                    file_name=f"chain_{selected_chain}.gif",
                                    mime="image/gif",
                                    icon=":material/download:",
                                    use_container_width=False,
                                    icon_position="right",
                                )
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

    col_filters, col_content = st.columns([1, 3], gap="large")

    with col_filters:
        st.markdown("##### Filters")
        sel_positions = st.multiselect("Filter by Position", pos_groups, placeholder="Select one or more positions...")
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

    players_info = []
    pools = []
    pos_labels = []
    COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, p_name in enumerate(selected_player_names):
        row = pool[pool["player_name"] == p_name].iloc[0]
        p_pos = row["position_group"]
        p_pool = sub_with_pos[sub_with_pos["position_group"] == p_pos]

        players_info.append((row, p_name, COLORS[idx % len(COLORS)]))
        pools.append(p_pool)
        pos_labels.append(p_pos)

    with col_content:
        if len(players_info) > 0:
            peer = pd.concat(pools).drop_duplicates("player_id")
            unique_pos = list(dict.fromkeys(pos_labels))
            peer_label = " + ".join(unique_pos)

            st.caption(f"Percentiles relative to **{peer_label}** ({len(peer)} players, ≥{minutes_threshold} min and ≥{chains_threshold} chains)")

            fig, vals_list = build_radar_figure(players_info, peer)
            st.pyplot(fig, use_container_width=False)
            radar_buffer = io.BytesIO()
            fig.savefig(
                radar_buffer,
                format="png",
                dpi=150,
                bbox_inches="tight",
                facecolor=DEFAULT_BG,
            )
            radar_bytes = radar_buffer.getvalue()

            def _safe_filename_part(name: str) -> str:
                ascii_name = name.encode("ascii", "ignore").decode("ascii")
                ascii_name = re.sub(r"[^A-Za-z0-9]+", "_", ascii_name).strip("_")
                return ascii_name or "player"

            display_names = selected_player_names
            name_part = "_vs_".join(_safe_filename_part(n) for n in display_names)
            file_name = f"pressing_dna_radar_{name_part}.png"

            st.download_button(
                label="Download Radar",
                data=radar_bytes,
                file_name=file_name,
                mime="image/png",
                icon=":material/download:",
                icon_position="right",
            )
            plt.close(fig)

            st.markdown("##### Raw Metric Values & Percentiles")
            compare_rows = []
            for i, (col_name, label) in enumerate(PILLARS):
                row_data = {"Pillar": label.replace("\n", " ")}

                for p_idx, (row_data_p, name_p, _) in enumerate(players_info):
                    v_p = row_data_p[col_name] if pd.notna(row_data_p[col_name]) else 0
                    if col_name == "chains_per_90":
                        fmt = "{:.1f}"
                    elif col_name == "negative_impact_per_chain" or col_name == "pressing_risk" or col_name == "defensive_penalty_per_100_chains" or col_name == "xT_generated_per_100_chains":
                        fmt = "{:.4f}"
                    else:
                        fmt = "{:.3f}"
                    row_data[name_p] = fmt.format(v_p)
                    row_data[f"{name_p} Pctile"] = f"{vals_list[p_idx][i]:.0f}th"

                compare_rows.append(row_data)

            st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Select players on the left to generate the radar and comparison table.")
    
st.markdown("<br><br>", unsafe_allow_html=True)
