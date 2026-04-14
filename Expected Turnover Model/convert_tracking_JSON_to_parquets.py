# -*- coding: utf-8 -*-
"""
Convert each tracking JSON into its own Parquet file.

Author: Pegah & Gustimorth
"""

import os
import json
import pandas as pd
from pathlib import Path

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
import os
from dotenv import load_dotenv

WORKSPACE = Path(__file__).resolve().parent

load_dotenv(WORKSPACE / ".env")
env_path = os.getenv("SKILLCORNER_DATA_DIR")

candidate_paths = [
    Path("C:/SkillcornerData/1/2024"),              # Local absolute path (User's PC)
    WORKSPACE.parent / "SkillcornerData/1/2024",    # Relative to repo, deep structure                
]
if env_path:
    candidate_paths.insert(0, Path(env_path))

BASE_DIR = None
for path in candidate_paths:
    if path.exists():
        BASE_DIR = path
        break

if BASE_DIR is None:
    print(f"❌ ERROR: Target Data Directory could not be found.")
    print("   Ensure the SkillCorner tracking dataset is defined in `.env` as `SKILLCORNER_DATA_DIR`")
    print("   Or placed in one of these locations:")
    for path in candidate_paths:
        print(f"   - {path}")
    import sys
    sys.exit(1)

TRACKING_DIR = BASE_DIR / "tracking"
META_DIR = BASE_DIR / "meta"
PARQUET_DIR = BASE_DIR / "tracking_parquets"

# Create output folder if it doesn't exist
PARQUET_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# HELPER: GET PLAYER MAPPING
# --------------------------------------------------------------------------
def get_player_team_map(match_id):
    """
    Reads the meta JSON and returns a dictionary: {player_id: team_id}
    """
    meta_path = META_DIR / f"{match_id}.json"
    if not meta_path.exists():
        return {}
    
    with open(meta_path, "r", encoding='utf-8') as f:
        data = json.load(f)
        
    # Create mapping: {1234: 44, 5678: 44, ...}
    pid_to_tid = {}
    if 'players' in data:
        for p in data['players']:
            pid_to_tid[p['id']] = p['team_id']
            
    return pid_to_tid

# --------------------------------------------------------------------------
# LOADER FUNCTION
# --------------------------------------------------------------------------
def load_tracking_full(match_id: int, sort_rows: bool = False, require_ball_detected: bool = True) -> pd.DataFrame:
    fpath = TRACKING_DIR / f"{match_id}.json"
    if not fpath.exists():
        raise FileNotFoundError(f"No tracking for match {match_id}")
    
    # Get player-to-team mapping from metadata
    player_map = get_player_team_map(match_id)
    
    with open(fpath, "r") as f:
        raw = json.load(f)

    rows = []
    for d in raw:
        frame = int(d.get("frame"))
        ts = d.get("timestamp")
        period = d.get("period")

        # Players
        for p in d.get("player_data") or []:
            p_id = p.get("player_id")
            rows.append({
                "match_id": match_id,
                "time": ts,
                "frame": frame,
                "period": period,
                "player_id": p_id,
                "team_id": player_map.get(p_id, -1),
                "is_detected": bool(p.get("is_detected", False)),
                "is_ball": False,
                "x": p.get("x"),
                "y": p.get("y"),
                "z": p.get("z", 0.0),
            })

        # Ball
        ball = d.get("ball_data")
        if ball is not None:
            if (not require_ball_detected) or (ball.get("is_detected") is not None):
                rows.append({
                    "match_id": match_id,
                    "time": ts,
                    "frame": frame,
                    "period": period,
                    "player_id": -1,
                    "team_id": -1,
                    "is_detected": bool(ball.get("is_detected", False)),
                    "is_ball": True,
                    "x": ball.get("x"),
                    "y": ball.get("y"),
                    "z": ball.get("z", 0.0),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if sort_rows:
        df = df.sort_values(["match_id", "frame", "player_id"]).reset_index(drop=True)
    return df

# --------------------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import glob

    match_ids = []
    matches_path = BASE_DIR / "matches.parquet"
    if matches_path.exists():
        df_matches = pd.read_parquet(matches_path)
        match_ids = df_matches["id"].values.tolist()
        print(f"Found {len(match_ids)} matches via matches.parquet")
    else:
        print(f"matches.parquet not found. Falling back to globbing tracking JSON files...")
        json_files = glob.glob(str(TRACKING_DIR / "*.json"))
        match_ids = [int(Path(f).stem) for f in json_files]
        print(f"Found {len(match_ids)} matching JSON files to process.")

    if not match_ids:
        print("❌ ERROR: No matches found to convert.")
        import sys
        sys.exit(1)

    for match_id in match_ids:
        try:
            df_tracking = load_tracking_full(match_id)
            if df_tracking is None or df_tracking.empty:
                print(f"⚠️ Empty tracking for {match_id}, skipping")
                continue

            # Save each match as its own parquet
            output_path = PARQUET_DIR / f"{match_id}.parquet"
            df_tracking.to_parquet(output_path)
            print(f"✅ Saved {output_path} ({len(df_tracking)} rows)")

        except FileNotFoundError:
            print(f"⚠️ No tracking for match {match_id}")
        except Exception as e:
            print(f"❌ Error with match {match_id}: {e}")

    print("✅ Finished converting all JSON files to Parquet.")
