# %%
"""
CHAIN-LEVEL xTURNOVER MODEL with Exact Game-Theoretic Shapley Values

Execution from Repository Root:
    python "Expected Turnover Model/xTO_pipeline_refactored.py"

Architecture:
1. Model predicts turnover probability for ENTIRE pressing chains (not individual engagements)
2. Features are aggregated from all players in the chain  
3. Target: chain_success (binary - did the chain cause a turnover?)
4. Shapley Attribution: Compute combinations $v(S)$ via $2^N$ combinatorial evaluation
5. Marginal Value: Derived from properly weighted factorial permutations of subset inclusions

Key Features:
- Chain-level aggregates: AUC pressure, spatial extent, coordination metrics
- Exact Game-Theoretic Shapley attribution for margin
- Game state context (passing options, defensive line height)
- Pure 2D Geometric Spatial proxies (LNS, AZMT)
- Isotonic Regression calibration
- Total Value Recovery & Temporal Proximity Decay

Author: Efstathios Papadopoulos
Date: April 2026
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import gc
import warnings
from scipy.signal import savgol_filter
from dataclasses import dataclass
from pathlib import Path
import sys
import math
from itertools import combinations
from scipy.spatial.distance import cdist
import os
from dotenv import load_dotenv

# Optional: SHAP for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available. Install with: pip install shap")

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# ==========================================
# 0. CONFIGURATION
# ==========================================

@dataclass
class Config:
    """Global project configuration and physics constants."""
    BASE_DIR: Path
    ROOT_DIR: Path
    OUTPUT_DIR: Path
    FPS: float = 10.0

    # Physics Constants (Bekkers Model)
    MAX_SPEED: float = 7.0        
    REACTION_TIME: float = 0.7    
    TARGET_WINDOW: float = 1.5    
    LOGISTIC_SIGMA: float = 0.45  
    EPSILON: float = 1e-5

def resolve_paths() -> Config:
    """Intelligently resolves data and output paths across devices."""
    base_dir = Path(__file__).resolve().parent
    
    load_dotenv(base_dir / ".env")
    env_path = os.getenv("SKILLCORNER_DATA_DIR")
    
    candidate_paths = [
        Path("C:/SkillcornerData/1/2024"),       
        base_dir.parent / "SkillcornerData/1/2024"      
    ]
    if env_path:
        candidate_paths.insert(0, Path(env_path))

    root_dir = None
    for path in candidate_paths:
        if path.exists():
            root_dir = path
            break

    out_dir = Path(__file__).resolve().parent  

    if root_dir is None:
        print(f"❌ ERROR: Data directory not found.")
        print("   Please define `SKILLCORNER_DATA_DIR` in a `.env` file at the root of the project,")
        print("   or place the data in one of these locations:")
        for path in candidate_paths:
            print(f"   - {path}")
        sys.exit(1)

    # Pre-execution check missing parquet structures
    parquet_dir = root_dir / "tracking_parquets"
    if not parquet_dir.exists() or not any(parquet_dir.iterdir()):
        print(f"⚠️ Parquet data not found in {parquet_dir}.")
        print("   Executing conversion script before spinning up pipeline...")
        import subprocess
        subprocess.run(["python", str(Path(__file__).resolve().parent / "convert_tracking_JSON_to_parquets.py")], check=True)
        print("✅ Parquet Conversion complete. Resuming pipeline execution...")

    return Config(BASE_DIR=base_dir, ROOT_DIR=root_dir, OUTPUT_DIR=out_dir)

CONFIG = resolve_paths()

# %%
# ==========================================
# 1. PHYSICS ENGINE
# ==========================================

class PhysicsEngine:
    @staticmethod
    def calculate_bekkers_pressure(df: pd.DataFrame, config: Config = CONFIG) -> np.ndarray:
        p_future_x = df['x'].values + (df['vx'].values * config.REACTION_TIME)
        p_future_y = df['y'].values + (df['vy'].values * config.REACTION_TIME)
        b_future_x = df['ball_x'].values + (df['ball_vx'].values * config.REACTION_TIME)
        b_future_y = df['ball_y'].values + (df['ball_vy'].values * config.REACTION_TIME)
        
        dist = np.sqrt((b_future_x - p_future_x)**2 + (b_future_y - p_future_y)**2)
        tau_dist = dist / config.MAX_SPEED
        
        u_x, u_y = df['vx'].values, df['vy'].values
        v_x, v_y = (b_future_x - df['x'].values), (b_future_y - df['y'].values)
        dot = (u_x * v_x) + (u_y * v_y)
        norm = np.sqrt(u_x**2 + u_y**2) * np.sqrt(v_x**2 + v_y**2)
        beta = np.arccos(np.clip(dot / (norm + config.EPSILON), -1, 1))
        tau_beta = (np.sqrt(u_x**2 + u_y**2) * beta) / np.pi
        
        T_ij = config.REACTION_TIME + tau_dist + tau_beta
        exponent = np.clip((-np.pi / (np.sqrt(3) * config.LOGISTIC_SIGMA)) * (config.TARGET_WINDOW - T_ij), -100, 100)
        p_ij = 1 / (1 + np.exp(exponent))
        
        speed = np.sqrt(df['vx'].values**2 + df['vy'].values**2)
        p_ij = np.where(speed < 2.0, 0.0, p_ij)
        return p_ij

    @staticmethod
    def compute_all_player_velocities(players_df: pd.DataFrame, fps: float = 10.0) -> pd.DataFrame:
        players_df = players_df.sort_values(['player_id', 'frame']).reset_index(drop=True)
        frames_all = players_df['frame'].to_numpy(dtype=np.float64)
        x_all      = players_df['x'].to_numpy(dtype=np.float64)
        y_all      = players_df['y'].to_numpy(dtype=np.float64)

        vx_all = np.zeros(len(players_df), dtype=np.float64)
        vy_all = np.zeros(len(players_df), dtype=np.float64)

        for idx in players_df.groupby('player_id', sort=False).indices.values():
            sort_order = np.argsort(frames_all[idx])
            s_idx      = idx[sort_order]

            frames = frames_all[s_idx]
            xs     = x_all[s_idx]
            ys     = y_all[s_idx]

            fdiff     = np.empty(len(frames), dtype=np.float64)
            fdiff[0]  = 1.0
            fdiff[1:] = frames[1:] - frames[:-1]
            dt = fdiff / fps

            dx = np.empty_like(xs); dx[0] = 0.0; dx[1:] = xs[1:] - xs[:-1]
            dy = np.empty_like(ys); dy[0] = 0.0; dy[1:] = ys[1:] - ys[:-1]

            valid = (fdiff >= 1) & (fdiff <= 4) & (dt > 0)
            vx = np.where(valid, dx / dt, np.nan)
            vy = np.where(valid, dy / dt, np.nan)

            fin = np.isfinite(vx)
            if fin.sum() >= 5:
                ia = np.arange(len(vx))
                if (~fin).any():
                    vx = np.interp(ia, ia[fin], vx[fin])
                    fin2 = np.isfinite(vy)
                    vy = np.interp(ia, ia[fin2], vy[fin2])
                vx = savgol_filter(vx, window_length=5, polyorder=2)
                vy = savgol_filter(vy, window_length=5, polyorder=2)
            else:
                np.nan_to_num(vx, copy=False, nan=0.0)
                np.nan_to_num(vy, copy=False, nan=0.0)

            vx_all[s_idx] = vx
            vy_all[s_idx] = vy

        players_df = players_df.copy()
        players_df['vx'] = vx_all
        players_df['vy'] = vy_all
        return players_df

    @staticmethod
    def compute_velocities_per_player(df_player: pd.DataFrame, fps: float = 10.0) -> pd.DataFrame:
        df_out = df_player.sort_values("frame").reset_index(drop=True).copy()

        frame_arr = df_out["frame"].to_numpy(dtype=np.float64)
        x_arr     = df_out["x"].to_numpy(dtype=np.float64)
        y_arr     = df_out["y"].to_numpy(dtype=np.float64)

        fdiff     = np.empty(len(frame_arr), dtype=np.float64)
        fdiff[0]  = 1.0
        fdiff[1:] = frame_arr[1:] - frame_arr[:-1]
        dt = fdiff / fps

        dx = np.empty_like(x_arr); dx[0] = 0.0; dx[1:] = x_arr[1:] - x_arr[:-1]
        dy = np.empty_like(y_arr); dy[0] = 0.0; dy[1:] = y_arr[1:] - y_arr[:-1]

        valid = (fdiff >= 1) & (fdiff <= 4) & (dt > 0)
        vx = np.where(valid, dx / dt, np.nan)
        vy = np.where(valid, dy / dt, np.nan)

        fin = np.isfinite(vx)
        if fin.sum() >= 5:
            ia = np.arange(len(vx))
            if (~fin).any():
                vx = np.interp(ia, ia[fin], vx[fin])
                fin2 = np.isfinite(vy)
                vy = np.interp(ia, ia[fin2], vy[fin2])
            vx = savgol_filter(vx, window_length=5, polyorder=2)
            vy = savgol_filter(vy, window_length=5, polyorder=2)
        else:
            np.nan_to_num(vx, copy=False, nan=0.0)
            np.nan_to_num(vy, copy=False, nan=0.0)

        df_out["vx"] = vx
        df_out["vy"] = vy
        return df_out

    @staticmethod
    def compute_velocities_ball(ball_df: pd.DataFrame, fps: float = 10.0) -> pd.DataFrame:
        ball_sorted = ball_df[['frame', 'ball_x', 'ball_y']].sort_values('frame').copy()
        tmp = PhysicsEngine.compute_velocities_per_player(
            ball_sorted.rename(columns={'ball_x': 'x', 'ball_y': 'y'}), fps=fps
        )
        ball_sorted['ball_vx'] = tmp['vx'].values
        ball_sorted['ball_vy'] = tmp['vy'].values
        return ball_sorted

# %%
# ==========================================
# 2. CHAIN-LEVEL FEATURE ENGINEERING
# ==========================================

class ChainFeatureEngine:
    @staticmethod
    def extract_phase_outcomes(dynamic_data: pd.DataFrame) -> pd.DataFrame:
        phase_outcomes = dynamic_data.groupby('phase_index').agg(
            phase_end_frame=('frame_end', 'max'),
            is_loss=('team_possession_loss_in_phase', 'max')
        ).reset_index()
        phase_outcomes['is_loss'] = phase_outcomes['is_loss'].fillna(0).astype(int)
        return phase_outcomes

    @staticmethod
    def normalize_attacking_direction(engagements_df: pd.DataFrame) -> pd.DataFrame:
        if 'attacking_side' in engagements_df.columns:
            flip_mask = engagements_df['attacking_side'] == 'right_to_left'
            if flip_mask.any():
                position_cols = ['x_start', 'y_start']
                if 'x_end' in engagements_df.columns: position_cols.append('x_end')
                if 'y_end' in engagements_df.columns: position_cols.append('y_end')
                
                for col in position_cols:
                    if col in engagements_df.columns:
                        engagements_df.loc[flip_mask, col] = -engagements_df.loc[flip_mask, col]
        return engagements_df

    @staticmethod
    def add_passing_options_context(engagements_df: pd.DataFrame, dynamic_data: pd.DataFrame) -> pd.DataFrame:
        poss_lookup = dynamic_data[dynamic_data['event_type'] == 'player_possession'].sort_values('frame_start')
        if not poss_lookup.empty:
            engagements_df = engagements_df.sort_values('frame_start')
            engagements_df = pd.merge_asof(engagements_df, poss_lookup[['frame_start', 'n_passing_options']], 
                               on='frame_start', direction='backward', suffixes=('', '_before'))
            engagements_df = engagements_df.sort_values('frame_end')
            engagements_df = pd.merge_asof(engagements_df, poss_lookup[['frame_start', 'n_passing_options']], 
                               left_on='frame_end', right_on='frame_start', direction='forward', 
                               suffixes=('', '_after'))
        else:
            engagements_df['n_passing_options_before'] = 0
            engagements_df['n_passing_options_after'] = 0
        return engagements_df

    @staticmethod
    def calculate_engagement_pressure_auc(engagements_df: pd.DataFrame, player_pressures: pd.DataFrame, fps: float = 10.0) -> pd.DataFrame:
        if 'frame_end' in engagements_df.columns:
            eng_windows = engagements_df[['event_id', 'player_id', 'frame_start', 'frame_end']].copy()
            eng_windows['frame_end']   = eng_windows['frame_end'].fillna(eng_windows['frame_start']).astype(int)
            eng_windows['frame_start'] = eng_windows['frame_start'].astype(int)
            auc_merge = player_pressures.merge(eng_windows, on='player_id', how='inner')
            in_window = ((auc_merge['frame'] >= auc_merge['frame_start']) &
                         (auc_merge['frame'] <= auc_merge['frame_end']))
            auc_by_eng = (auc_merge[in_window]
                          .groupby('event_id')['pressure_prob']
                          .sum()
                          .div(fps)
                          .rename('engagement_pressure_auc'))
            engagements_df = engagements_df.merge(auc_by_eng, on='event_id', how='left')
        else:
            engagements_df['engagement_pressure_auc'] = 0.0
        engagements_df['engagement_pressure_auc'] = engagements_df['engagement_pressure_auc'].fillna(0.0)
        return engagements_df

    @staticmethod
    def calculate_spatial_metrics(engagement_frames: np.ndarray, df_track: pd.DataFrame) -> pd.DataFrame:
        track_eng = df_track[df_track['frame'].isin(engagement_frames)]
        
        frame_metrics = []
        for f in engagement_frames:
            f_track = track_eng[track_eng['frame'] == f]
            if f_track.empty: continue
                
            bx = f_track['ball_x'].iloc[0]
            by = f_track['ball_y'].iloc[0]
            poss_team = f_track['possession_team_id'].iloc[0]
            
            attackers = f_track[f_track['team_id'] == poss_team]
            defenders = f_track[f_track['team_id'] != poss_team]
            
            dist_att_ball = np.hypot(attackers['x'] - bx, attackers['y'] - by)
            dist_def_ball = np.hypot(defenders['x'] - bx, defenders['y'] - by)
            
            att_15m = attackers[dist_att_ball <= 15.0]
            def_15m = defenders[dist_def_ball <= 15.0]
            
            lns = len(def_15m) - len(att_15m)
            def_ids = def_15m['player_id'].tolist()
            
            azmt = 0.0
            azmt_loo_player = {}
            
            if len(att_15m) > 0 and len(def_15m) > 0:
                att_coords = att_15m[['x', 'y']].values
                def_coords = def_15m[['x', 'y']].values
                dists = cdist(att_coords, def_coords)
                min_dists = dists.min(axis=1)
                azmt = min_dists.mean()
                
                for j, def_id in enumerate(def_ids):
                    if len(def_15m) == 1:
                        azmt_loo_player[def_id] = 0.0
                    else:
                        dists_loo = np.delete(dists, j, axis=1)
                        azmt_loo_player[def_id] = dists_loo.min(axis=1).mean()
                        
            frame_metrics.append({
                'frame_start': f, 'LNS': lns, 'AZMT': azmt,
                'defenders_15m': def_ids, 'azmt_loo_dict': azmt_loo_player
            })
            
        metrics_df = pd.DataFrame(frame_metrics)
        return metrics_df

    @staticmethod
    def calculate_base_features(engagements_df: pd.DataFrame, frame_pressure: pd.DataFrame, fps: float = 10.0):
        engagements_df['dist_to_goal'] = np.sqrt((52.5 - engagements_df['x_start'])**2 + (0 - engagements_df['y_start'])**2)
        
        if 'last_defensive_line_height_start' in engagements_df.columns:
            engagements_df['defensive_line_height'] = engagements_df['last_defensive_line_height_start'].fillna(35.0) / 105.0
        else:
            engagements_df['defensive_line_height'] = 0.33
        
        if 'x_end' in engagements_df.columns and 'x_start' in engagements_df.columns:
            engagements_df['pressing_distance'] = np.sqrt((engagements_df['x_end'] - engagements_df['x_start'])**2 + 
                                                           (engagements_df['y_end'] - engagements_df['y_start'])**2)
            engagements_df['forward_pressing'] = (engagements_df['x_end'] - engagements_df['x_start'] > 0.5).astype(int)
        else:
            engagements_df['pressing_distance'] = 0
            engagements_df['forward_pressing'] = 0
        
        engagements_df['pressing_chain_length'] = engagements_df['pressing_chain_length'].fillna(1) if 'pressing_chain_length' in engagements_df.columns else 1
        
        if 'engagement_pressure_auc' not in engagements_df.columns:
            engagements_df['engagement_pressure_auc'] = 0.0

        engagements_df['engagement_delta_options'] = (
            engagements_df['n_passing_options_after'].fillna(0) - 
            engagements_df['n_passing_options_before'].fillna(0)
        )

        engagements_df = pd.merge(engagements_df, frame_pressure, left_on='frame_start', right_on='frame', how='left')
        engagements_df['frame_max_pressure'] = engagements_df['frame_max_pressure'].fillna(0)
        
        if 'frame_end' not in engagements_df.columns: engagements_df['frame_end'] = engagements_df['frame_start']
        engagements_df = engagements_df.sort_values(['pressing_chain_index', 'frame_start'])

        chain_features = engagements_df.groupby('pressing_chain_index').agg(
            chain_size=('player_id', 'nunique'),
            chain_start_frame=('frame_start', 'min'),
            chain_end_frame=('frame_end', 'max'),
            chain_mean_y=('y_start', 'mean'),
            mean_press_pressure=('frame_max_pressure', 'mean'),
            mean_dist_to_goal=('dist_to_goal', 'mean'),
            mean_pressing_distance=('pressing_distance', 'mean'),
            forward_pressing_ratio=('forward_pressing', 'mean'),
            defensive_line_height=('defensive_line_height', 'mean'),
            chain_success=('chain_success', 'max'),
            conceded_xT=('conceded_xT', 'first'),
            mean_delta_options=('engagement_delta_options', 'mean'),
            game_state=('game_state', 'first'),
            third_start=('third_start', 'first'),
            match_id=('match_id', 'first'),
            chain_length_sc=('pressing_chain_length', 'max'),
            chain_mean_LNS=('LNS', 'mean'),
            chain_mean_AZMT=('AZMT', 'mean'),
            chain_mean_auc=('engagement_pressure_auc', 'mean')
        ).reset_index()
        
        chain_features['chain_duration'] = (chain_features['chain_end_frame'] - chain_features['chain_start_frame']) / fps
        chain_features['engagement_density'] = chain_features['chain_size'] / (chain_features['chain_duration'] + 0.1) 
        chain_features = chain_features.drop(columns=['chain_start_frame', 'chain_end_frame'])
        chain_features['chain_proximity_to_sideline'] = np.abs(chain_features['chain_mean_y'])
        chain_features['is_coordinated_press'] = (chain_features['chain_size'] > 1).astype(int)
        
        player_chain_map = (
            engagements_df.drop_duplicates(subset=['pressing_chain_index', 'player_id'])
            .groupby('pressing_chain_index')
            .agg(player_ids=('player_id', list), player_names=('player_name', list))
            .to_dict('index')
        )
        return chain_features, player_chain_map, engagements_df

    @staticmethod
    def prepare_subset_lookups(engagements_df: pd.DataFrame) -> dict:
        lookups = {}
        for global_chain_id, group in tqdm(engagements_df.groupby('global_chain_id'), desc="Preparing Subset Lookups"):
            engs = group.to_dict('records')
            original_players = set(group['player_id'].unique())
            lookups[global_chain_id] = {
                'engagements': engs,
                'original_players': original_players,
                'chain_length_sc': len(engs)
            }
        return lookups

    @staticmethod
    def calculate_subset_features(lookup_data: dict, subset_player_ids: list, fps: float = 10.0) -> dict:
        original_players = lookup_data['original_players']
        subset_set = set(subset_player_ids)
        excluded_players = original_players - subset_set
        
        active_engs = [e for e in lookup_data['engagements'] if e['player_id'] in subset_set]
        if not active_engs: return None

        count = len(active_engs)
        mean_delta_options = sum(e.get('engagement_delta_options', 0) for e in active_engs) / count
        mean_y = sum(e['y_start'] for e in active_engs) / count
        chain_mean_auc = sum(e.get('engagement_pressure_auc', 0) for e in active_engs) / count
        
        mean_press = sum(e.get('frame_max_pressure', 0) for e in active_engs) / count
        mean_dist_to_goal = sum(e['dist_to_goal'] for e in active_engs) / count
        mean_pressing_dist = sum(e['pressing_distance'] for e in active_engs) / count
        forward_pressing_ratio = sum(e['forward_pressing'] for e in active_engs) / count
        defensive_line_height = sum(e.get('defensive_line_height', 0) for e in active_engs) / count
        
        lns_sum = 0
        azmt_sum = 0
        for e in active_engs:
            base_lns = e.get('LNS', 0)
            defs_15m = e.get('defenders_15m', [])
            rem_lns = len([p for p in excluded_players if p in defs_15m])
            lns_sum += (base_lns - rem_lns)
            
            if len(excluded_players) == 0:
                azmt_sum += e.get('AZMT', 0)
            elif len(excluded_players) == 1:
                p_exc = list(excluded_players)[0]
                loo_dict = e.get('azmt_loo_dict', {})
                azmt_sum += loo_dict.get(p_exc, e.get('AZMT', 0))
            else:
                azmt_sum += e.get('AZMT', 0)
                
        mean_lns = lns_sum / count
        mean_azmt = azmt_sum / count

        start_frame = min(e['frame_start'] for e in active_engs)
        end_frame = max(e.get('frame_end', e['frame_start']) for e in active_engs)
        sub_duration = (end_frame - start_frame) / fps
        sub_density = count / (sub_duration + 0.1)

        features = {
            'chain_size': len(subset_set),
            'chain_length_sc': count,
            'chain_mean_y': mean_y,
            'chain_proximity_to_sideline': abs(mean_y),
            'chain_duration': sub_duration,
            'engagement_density': sub_density,
            'mean_press_pressure': mean_press,
            'mean_dist_to_goal': mean_dist_to_goal,
            'mean_pressing_distance': mean_pressing_dist,
            'forward_pressing_ratio': forward_pressing_ratio,
            'defensive_line_height': defensive_line_height,
            'mean_delta_options': mean_delta_options,
            'chain_mean_LNS': mean_lns,
            'chain_mean_AZMT': mean_azmt,
            'chain_mean_auc': chain_mean_auc,
            'is_coordinated_press': 1 if len(subset_set) > 1 else 0
        }
        return features

class MatchProcessor:
    @staticmethod
    def process_single_match(match_id: str, track_path: str, dyn_path: str, meta_path: str, phys_path: str, fps: float = 10.0) -> dict:
        if not os.path.exists(dyn_path) or not os.path.exists(meta_path):
            return None

        tracking_df = pd.read_parquet(track_path, columns=['frame', 'player_id', 'x', 'y', 'is_ball', 'team_id'])
        dynamic_df = pd.read_parquet(dyn_path)
        
        physical_df = None
        if os.path.exists(phys_path):
            physical_df = pd.read_parquet(phys_path, columns=['player_id', 'minutes_full_all', 'team_id'])
            physical_df['match_id'] = match_id

        teams_dict = {}
        if 'team_shortname' in dynamic_df.columns and 'team_id' in dynamic_df.columns:
            teams_in_match = dynamic_df[['team_id', 'team_shortname']].dropna().drop_duplicates()
            teams_dict = dict(zip(teams_in_match['team_id'].astype(int), teams_in_match['team_shortname']))

        ball = tracking_df[tracking_df['is_ball']].copy()
        ball = ball.rename(columns={'x': 'ball_x', 'y': 'ball_y'})
        ball = PhysicsEngine.compute_velocities_ball(ball, fps=fps)

        players = tracking_df[~tracking_df['is_ball']].copy()
        p_team = players[['player_id', 'team_id']].drop_duplicates()
        players = PhysicsEngine.compute_all_player_velocities(players, fps=fps)

        df_track = pd.merge(players, ball, on='frame', how='inner')
        df_track['pressure_prob'] = PhysicsEngine.calculate_bekkers_pressure(df_track, config=CONFIG)
        
        _poss = (dynamic_df.dropna(subset=['team_id'])
                 [['frame_start', 'frame_end', 'team_id']]
                 .astype({'frame_start': int, 'frame_end': int}))
        _n = (_poss['frame_end'] - _poss['frame_start'] + 1).values
        _frames = np.concatenate([np.arange(r.frame_start, r.frame_end + 1)
                                  for r in _poss.itertuples(index=False)])
        _teams  = np.repeat(_poss['team_id'].values, _n)
        frame_to_possession = dict(zip(_frames.tolist(), _teams.tolist()))
        df_track['possession_team_id'] = df_track['frame'].map(frame_to_possession)
        df_track.loc[df_track['team_id'] == df_track['possession_team_id'], 'pressure_prob'] = 0

        player_pressure_series = df_track[['frame', 'player_id', 'pressure_prob']].copy()
        frame_pressure = df_track.groupby('frame')['pressure_prob'].max().reset_index()
        frame_pressure.columns = ['frame', 'frame_max_pressure']

        phase_outcomes = ChainFeatureEngine.extract_phase_outcomes(dynamic_df)

        engagement_data = dynamic_df[dynamic_df['event_type'] == 'on_ball_engagement'].copy()
        if engagement_data.empty: return None
        
        if 'team_out_of_possession_phase_type' in engagement_data.columns:
            engagement_data = engagement_data[engagement_data['team_out_of_possession_phase_type'] == 'high_block'].copy()
        if engagement_data.empty: return None
        
        engagement_data = ChainFeatureEngine.normalize_attacking_direction(engagement_data)
        engagement_data = ChainFeatureEngine.add_passing_options_context(engagement_data, dynamic_df)
        
        engagement_frames = engagement_data['frame_start'].unique()
        metrics_df = ChainFeatureEngine.calculate_spatial_metrics(engagement_frames, df_track)
        
        if not metrics_df.empty:
            engagement_data = pd.merge(engagement_data, metrics_df, on='frame_start', how='left')
        else:
            engagement_data['LNS'] = 0
            engagement_data['AZMT'] = 0.0
            engagement_data['defenders_15m'] = [[] for _ in range(len(engagement_data))]
            engagement_data['azmt_loo_dict'] = [{} for _ in range(len(engagement_data))]
            
        engagement_data['LNS'] = engagement_data['LNS'].fillna(0)
        engagement_data['AZMT'] = engagement_data['AZMT'].fillna(0.0)
        engagement_data['defenders_15m'] = engagement_data['defenders_15m'].apply(lambda x: x if isinstance(x, list) else [])
        engagement_data['azmt_loo_dict'] = engagement_data['azmt_loo_dict'].apply(lambda x: x if isinstance(x, dict) else {})
        
        engagement_data = pd.merge(engagement_data, phase_outcomes, on='phase_index', how='left')
        engagement_data['time_to_end'] = engagement_data['phase_end_frame'] - engagement_data['frame_start']
        engagement_data['chain_success'] = np.where((engagement_data['is_loss'] == 1) & (engagement_data['time_to_end'] <= 50), 1, 0)
        engagement_data['pressing_chain_index'] = engagement_data['pressing_chain_index'].fillna(engagement_data['event_id']).astype(str)
        engagement_data['match_id'] = match_id

        has_xthreat = 'xthreat' in dynamic_df.columns
        conceded_xT_map = {}
        if has_xthreat:
            for chain_id, chain_group in engagement_data.groupby('pressing_chain_index'):
                if chain_group['chain_success'].iloc[0] == 1:
                    conceded_xT_map[chain_id] = 0.0
                else:
                    phase_idx = chain_group['phase_index'].iloc[0]
                    chain_end_frame = chain_group['frame_end'].max()
                    phase_events = dynamic_df[dynamic_df['phase_index'] == phase_idx]
                    poss_team = phase_events[phase_events['event_type'] == 'player_possession']['team_id'].mode()
                    if not poss_team.empty:
                        poss_team_id = poss_team.iloc[0]
                        next_poss = phase_events[
                            (phase_events['event_type'] == 'player_possession') & 
                            (phase_events['team_id'] == poss_team_id) & 
                            (phase_events['frame_start'] >= chain_end_frame)
                        ].sort_values('frame_start')
                        
                        if not next_poss.empty:
                            receiver_id = next_poss.iloc[0]['player_id']
                            recv_frame = next_poss.iloc[0]['frame_start']
                            xT_events = phase_events[
                                (phase_events['player_id'] == receiver_id) & 
                                (phase_events['event_type'].isin(['passing_option', 'off_ball_run'])) & 
                                (phase_events['frame_end'] <= recv_frame + 20)
                            ].sort_values('frame_end', ascending=False)
                            
                            if not xT_events.empty:
                                raw_xt = xT_events.iloc[0]['xthreat']
                                conceded_xT_map[chain_id] = float(raw_xt) if pd.notna(raw_xt) else 0.0
                            else: conceded_xT_map[chain_id] = 0.0
                        else: conceded_xT_map[chain_id] = 0.0
                    else: conceded_xT_map[chain_id] = 0.0
        
        engagement_data['conceded_xT'] = engagement_data['pressing_chain_index'].map(conceded_xT_map).fillna(0.0)
        engagement_data = ChainFeatureEngine.calculate_engagement_pressure_auc(engagement_data, player_pressure_series, fps=fps)

        chain_df, player_chain_map, eng_with_features = ChainFeatureEngine.calculate_base_features(engagement_data, frame_pressure, fps=fps)

        del tracking_df, dynamic_df, df_track, players, ball, engagement_data, player_pressure_series
        gc.collect()

        return {
            'chain_features': chain_df,
            'engagement_store': eng_with_features,
            'player_team_map': p_team,
            'team_names': teams_dict,
            'physical_stats': physical_df,
            'player_chain_map': player_chain_map
        }

# %%
# ==========================================
# 3. MODEL TRAINING AND EXPLAINABILITY
# ==========================================

class TurnoverModel:
    def __init__(self, feature_cols: list, weight_ratio: float):
        self.feature_cols = feature_cols
        self.weight_ratio = weight_ratio
        
        self.base_model = xgb.XGBClassifier(
            n_estimators=400, 
            learning_rate=0.02, 
            max_depth=5, 
            min_child_weight=3,
            subsample=0.8, 
            colsample_bytree=0.8, 
            scale_pos_weight=self.weight_ratio,
            max_delta_step=2,
            gamma=0.5,
            reg_alpha=0.1,
            eval_metric='logloss', 
            random_state=42, 
            n_jobs=-1
        )
        self.model = None
        self.optimal_threshold = 0.5
        
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        print("\n🔄 Running 5-fold Cross-Validation...")
        cv_scores = cross_val_score(self.base_model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
        print(f"   Mean CV ROC-AUC: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")
        
        print("\n🎯 Training final model with calibration...")
        self.model = CalibratedClassifierCV(self.base_model, method='isotonic', cv=3, n_jobs=-1)
        self.model.fit(X_train, y_train)
        print(f"✅ Model trained successfully!")
        
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        print("\n📊 Generating Evaluation Metrics...")
        y_probs = self.model.predict_proba(X_test)[:, 1]
        
        precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
        best_f1_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[best_f1_idx]
        
        y_pred = (y_probs >= self.optimal_threshold).astype(int)
        
        print(f"   ⚙️  Using optimal threshold: {self.optimal_threshold:.4f} (Best F1)")
        print("\nCLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred, target_names=['No Turnover', 'Turnover']))
        
        pr_auc = auc(recall, precision)
        print(f"   PR AUC: {pr_auc:.3f}")
        
        # Plot ROC and PR curves
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC Curve
        ax1.plot(fpr, tpr, color='orange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic')
        ax1.legend(loc="lower right")
        ax1.grid(alpha=0.3)
        
        # PR Curve
        ax2.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def explain_with_shap(self, X_test: pd.DataFrame):
        if not SHAP_AVAILABLE:
            print("\n⚠️  SHAP not available. Skipping explainability.")
            return
            
        print("\n🔍 GENERATING SHAP EXPLAINABILITY ANALYSIS...")
        base_estimator = self.model.calibrated_classifiers_[0].estimator if hasattr(self.model, 'calibrated_classifiers_') else self.model
        
        sample_size = min(500, len(X_test))
        X_shap_sample = X_test.sample(n=sample_size, random_state=42)
        
        print(f"   Computing SHAP values for {sample_size} test samples...")
        explainer = shap.TreeExplainer(base_estimator)
        shap_values = explainer.shap_values(X_shap_sample)
        
        # Summary Plot
        print("   Generating SHAP summary plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_shap_sample, 
                          max_display=20, 
                          show=False)
        plt.title('SHAP Feature Impact Analysis\n(How features influence xTurnover predictions)', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        # Bar Plot
        print("   Generating SHAP feature importance bar plot...")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_shap_sample, 
                          plot_type="bar",
                          max_display=15,
                          show=False)
        plt.title('Mean Absolute SHAP Values\n(Average feature contribution magnitude)', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        # Waterfall Plot
        print("   Generating example waterfall plot for high xTurnover chain...")
        y_probs_sample = self.model.predict_proba(X_shap_sample)[:, 1]
        high_prob_idx = y_probs_sample.argmax()
        
        shap_explanation = shap.Explanation(
            values=shap_values[high_prob_idx],
            base_values=explainer.expected_value,
            data=X_shap_sample.iloc[high_prob_idx].values,
            feature_names=self.feature_cols
        )
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap_explanation, max_display=15, show=False)
        plt.title(f'SHAP Waterfall: Highest Probability Chain (xT = {y_probs_sample[high_prob_idx]:.3f})\n'
                  f'Shows how each feature pushes prediction from baseline to final value',
                  fontsize=12, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        # Dependence Plot
        feature_to_select = "defensive_line_height"
        print(f"   Generating dependence plot for selected feature: {feature_to_select}...")
        if feature_to_select in self.feature_cols:
            feature_idx = self.feature_cols.index(feature_to_select)
            plt.figure(figsize=(8, 6))
            feature_values = X_shap_sample.iloc[:, feature_idx].values
            shap_vals = shap_values[:, feature_idx]
            
            plt.scatter(feature_values, shap_vals, color='black', alpha=1, s=30)
            plt.xlabel(f"{feature_to_select} (normalized)", fontsize=11)
            plt.ylabel('SHAP value', fontsize=11)
            plt.title(f'SHAP Dependence: {feature_to_select}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.show()
        else:
            print(f"⚠️ Feature '{feature_to_select}' not found. Cannot plot dependence plot.")
            
        print("✅ SHAP analysis complete!")
        print("\n📊 SHAP Interpretation Guide:")
        print("   • Summary Plot: Red = high feature value, Blue = low feature value")
        print("   • Horizontal spread = impact on model output (right = increases xTurnover)")
        print("   • Bar Plot: Shows average absolute contribution of each feature")
        print("   • Waterfall: Shows how individual features add up to final prediction")
        print("   • Dependence: Shows relationship between feature value and its SHAP impact")

# %%
# ==========================================
# 4. COUNTERFACTUAL SHAPLEY ATTRIBUTION (BATCH INFERENCE)
# ==========================================

class ShapleyAttributionEngine:
    def __init__(self, temporal_decay_tau: int = 50, temporal_weight_floor: float = 0.4):
        self.temporal_decay_tau = temporal_decay_tau
        self.temporal_weight_floor = temporal_weight_floor
        
    def calculate_exact_shapley(self, model, master_chain_df: pd.DataFrame, 
                                master_engagement_df: pd.DataFrame, 
                                player_chain_mapping: dict, feature_cols: list) -> pd.DataFrame:
        print("\n🏆 CALCULATING GAME-THEORETIC SHAPLEY MAPS (BATCH INFERENCE)...")
        
        master_chain_df['xTurnover_full'] = model.predict_proba(master_chain_df[feature_cols])[:, 1]
        lookups = ChainFeatureEngine.prepare_subset_lookups(master_engagement_df)
        
        subset_features_list = []
        subset_mapping = []
        marginal_contributions = []
        
        for idx, chain_row in tqdm(master_chain_df.iterrows(), total=len(master_chain_df), desc="Subset Generation"):
            match_id     = str(chain_row['match_id'])
            chain_id     = str(chain_row['pressing_chain_index'])
            global_chain_id = str(chain_row['global_chain_id'])
            chain_key    = f"{match_id}_{chain_id}"
            
            if chain_key not in player_chain_mapping or global_chain_id not in lookups: continue
            
            player_ids = player_chain_mapping[chain_key]['player_ids']
            player_names = player_chain_mapping[chain_key]['player_names']
            N = len(player_ids)
            
            if N == 1:
                marginal_contributions.append({
                    'match_id': match_id, 'pressing_chain_index': chain_id, 'global_chain_id': global_chain_id,
                    'player_id': player_ids[0], 'player_name': player_names[0],
                    'xTurnover_full': chain_row['xTurnover_full'], 'xTurnover_without_player': 0.0,
                    'raw_shapley_value': chain_row['xTurnover_full'],
                    'marginal_xTurnover': chain_row['xTurnover_full'], 'chain_success': chain_row['chain_success'],
                    'conceded_xT': chain_row.get('conceded_xT', 0.0), 'chain_size': 1, 'temporal_weight': 1.0
                })
                continue
                
            all_subsets = []
            for r in range(1, N + 1):
                all_subsets.extend(combinations(player_ids, r))
                
            for subset in all_subsets:
                if len(subset) == N: continue  
                feats = ChainFeatureEngine.calculate_subset_features(lookups[global_chain_id], list(subset), fps=CONFIG.FPS)
                if feats is not None:
                    subset_features_list.append(feats)
                    subset_mapping.append((global_chain_id, subset))

        v_S = {}
        if subset_features_list:
            print("\n🔮 Running batch predict_proba() horizontally...")
            batch_df = pd.DataFrame(subset_features_list)[feature_cols].fillna(0)
            batch_probs = model.predict_proba(batch_df)[:, 1]
            
            for (g_id, sub_tuple), prob in zip(subset_mapping, batch_probs):
                if g_id not in v_S: v_S[g_id] = {}
                v_S[g_id][sub_tuple] = prob
        
        print("\n🔀 Mapping Factorials to Marginals...")
        for idx, chain_row in tqdm(master_chain_df.iterrows(), total=len(master_chain_df), desc="Marginal Calculation"):
            chain_key = f"{chain_row['match_id']}_{chain_row['pressing_chain_index']}"
            g_id = str(chain_row['global_chain_id'])
            
            if chain_key not in player_chain_mapping: continue
            
            player_ids = player_chain_mapping[chain_key]['player_ids']
            player_names = player_chain_mapping[chain_key]['player_names']
            N = len(player_ids)
            if N == 1: continue 
            
            xTurnover_full = chain_row['xTurnover_full']
            chain_engs = lookups.get(g_id, {}).get('engagements', [])
            chain_last_frame = max((e.get('frame_end', e['frame_start']) for e in chain_engs), default=None) if chain_engs else None
            
            local_v_S = v_S.get(g_id, {})
            local_v_S[()] = 0.0
            local_v_S[tuple(player_ids)] = xTurnover_full
            
            for p_id, p_name in zip(player_ids, player_names):
                shapley_value = 0.0
                subsets_without_i = [s for s in local_v_S.keys() if p_id not in s and len(s) < N]
                
                for S in subsets_without_i:
                    S_union_i = tuple(sorted(list(S) + [p_id], key=lambda x: player_ids.index(x))) if len(S)+1 < N else tuple(player_ids)
                    val_S = local_v_S.get(S, 0.0)
                    val_S_union_i = local_v_S.get(S_union_i, 0.0)
                    
                    marginal = val_S_union_i - val_S
                    weight = (math.factorial(len(S)) * math.factorial(N - len(S) - 1)) / math.factorial(N)
                    shapley_value += weight * marginal
                
                loo_subset = tuple([p for p in player_ids if p != p_id])
                xT_without = local_v_S.get(loo_subset, 0.0)
                
                p_engs = [e for e in chain_engs if e['player_id'] == p_id]
                if p_engs and chain_last_frame:
                    p_last = max(e.get('frame_end', e['frame_start']) for e in p_engs)
                    time_gap = max(0, chain_last_frame - p_last)
                    temporal_weight = max(self.temporal_weight_floor, np.exp(-time_gap / self.temporal_decay_tau))
                else: temporal_weight = 1.0
                
                marginal_contributions.append({
                    'match_id': chain_row['match_id'], 'pressing_chain_index': chain_row['pressing_chain_index'],
                    'global_chain_id': g_id, 'player_id': p_id, 'player_name': p_name,
                    'xTurnover_full': xTurnover_full, 'xTurnover_without_player': xT_without,
                    'raw_shapley_value': shapley_value, 'marginal_xTurnover': max(0.0, shapley_value),
                    'chain_success': chain_row['chain_success'], 'conceded_xT': chain_row.get('conceded_xT', 0.0),
                    'chain_size': N, 'temporal_weight': temporal_weight
                })
                
        marginal_df = pd.DataFrame(marginal_contributions)
        return self._distribute_residual(marginal_df)

    def _distribute_residual(self, marginal_df: pd.DataFrame) -> pd.DataFrame:
        def _apply(grp):
            xT_full = grp['xTurnover_full'].iloc[0]
            clipped = grp['marginal_xTurnover'].values.copy()
            s = clipped.sum()
            residual = xT_full - s
            grp = grp.copy()
            if s > 0:
                grp['marginal_xTurnover'] = clipped + residual * (clipped / s)
            else:
                grp['marginal_xTurnover'] = xT_full / len(grp)
            return grp
        
        out_df = marginal_df.groupby('global_chain_id', group_keys=False).apply(_apply)
        
        out_df['weighted_marginal_xTurnover'] = out_df['marginal_xTurnover'] * out_df['temporal_weight']
        sum_by_chain = out_df.groupby('global_chain_id')['weighted_marginal_xTurnover'].sum().reset_index()
        sum_by_chain = sum_by_chain.rename(columns={'weighted_marginal_xTurnover': 'sum_weighted_marginal'})
        
        out_df = pd.merge(out_df, sum_by_chain, on='global_chain_id', how='left')
        
        out_df['contribution_share'] = np.where(
            out_df['sum_weighted_marginal'] > 0,
            out_df['weighted_marginal_xTurnover'] / out_df['sum_weighted_marginal'],
            1.0 / out_df['chain_size']
        )
        
        out_df['attributed_xTurnover'] = out_df['contribution_share'] * out_df['xTurnover_full']
        out_df['attributed_turnover']  = out_df['chain_success'] * out_df['contribution_share']
        out_df['defensive_penalty'] = out_df['contribution_share'] * (1 - out_df['xTurnover_full']) * out_df['conceded_xT']
        
        # Lock in calibrated column names for ranking consistency
        out_df['marginal_xTurnover_calibrated'] = out_df['attributed_xTurnover']
        out_df['xTurnover_full_calibrated'] = out_df['xTurnover_full']
        out_df['xTurnover_without_player_calibrated'] = out_df['xTurnover_without_player']
        
        return out_df

# %%
# ==========================================
# 5. METRICS AND EXPORT ENGINE
# ==========================================

class MetricsAndExportEngine:
    @staticmethod
    def calculate_sub_metrics(marginal_df: pd.DataFrame, ranking_df: pd.DataFrame, physical_stats_df: pd.DataFrame, team_name_map: dict, player_team_map_list: list) -> pd.DataFrame:
        print("\n📈 Calculating Nuanced xTurnover Sub-Metrics...")
        metrics_df = ranking_df.copy().set_index('player_id')

        # Map Teams
        if player_team_map_list:
            master_player_team = pd.concat(player_team_map_list, ignore_index=True).drop_duplicates('player_id')
            master_player_team['team_name'] = master_player_team['team_id'].map(team_name_map)
            metrics_df = metrics_df.join(master_player_team.set_index('player_id')[['team_name']])
        else:
            metrics_df['team_name'] = 'Unknown'

        # Minutes Played
        if physical_stats_df is not None and not physical_stats_df.empty:
            minutes_df = physical_stats_df.groupby('player_id').agg(
                minutes_played=('minutes_full_all', 'sum'),
                matches_played=('match_id', 'nunique')
            )
            metrics_df = metrics_df.join(minutes_df)
        else:
            metrics_df['minutes_played'] = 90.0 * 30
            metrics_df['matches_played'] = 30
            
        metrics_df['minutes_played'] = metrics_df['minutes_played'].fillna(0).replace(0, 1)
        per_90_factor = 90.0 / metrics_df['minutes_played']

        # Base / P90 Stats
        metrics_df['xTurnover_per_90'] = metrics_df['total_marginal_xTurnover_calibrated'] * per_90_factor
        metrics_df['chains_per_90'] = metrics_df['chains_participated'] * per_90_factor
        metrics_df['chains_per_match'] = metrics_df['chains_participated'] / metrics_df['matches_played'].replace(0, 1)
        metrics_df['xTurnover_per_chain'] = metrics_df['total_marginal_xTurnover_calibrated'] / metrics_df['chains_participated'].replace(0, 1)
        
        metrics_df['defensive_penalty_per_90'] = metrics_df['defensive_penalty'] * per_90_factor
        metrics_df['net_defensive_value_per_90'] = (metrics_df['total_marginal_xTurnover_calibrated'] - metrics_df['defensive_penalty']) * per_90_factor
        metrics_df['attributed_turnovers_per_90'] = metrics_df['attributed_turnovers'] * per_90_factor
        metrics_df['value_added'] = metrics_df['attributed_turnovers'] - metrics_df['total_marginal_xTurnover_calibrated']
        metrics_df['value_added_per_90'] = metrics_df['value_added'] * per_90_factor

        # Solo vs Coordinated
        solo_mask = (marginal_df['chain_size'] == 1)
        coord_mask = (marginal_df['chain_size'] > 1)

        solo_stats = marginal_df[solo_mask].groupby('player_id').agg(
            solo_chains=('global_chain_id', 'count'),
            solo_xTurnover_total=('marginal_xTurnover_calibrated', 'sum'),
            solo_attributed=('attributed_turnover', 'sum')
        )
        coord_stats = marginal_df[coord_mask].groupby('player_id').agg(
            coordinated_chains=('global_chain_id', 'count'),
            coordinated_xTurnover_total=('marginal_xTurnover_calibrated', 'sum'),
            coordinated_attributed=('attributed_turnover', 'sum'),
            avg_contribution_share=('contribution_share', 'mean')
        )
        metrics_df = metrics_df.join(solo_stats).join(coord_stats).fillna(0)
        
        metrics_df['solo_xTurnover_per_90'] = metrics_df['solo_xTurnover_total'] * per_90_factor
        metrics_df['coordinated_xTurnover_per_90'] = metrics_df['coordinated_xTurnover_total'] * per_90_factor
        metrics_df['solo_xTurnover_ratio'] = metrics_df['solo_xTurnover_total'] / metrics_df['total_marginal_xTurnover_calibrated'].replace(0, 1)

        # Dominance
        dominance_stats = marginal_df.groupby('player_id').agg(
            median_contribution_share=('contribution_share', 'median')
        )
        metrics_df = metrics_df.join(dominance_stats)

        # Consistency (Match-Level)
        match_level = marginal_df.groupby(['player_id', 'match_id']).agg(
            match_xTurnover=('marginal_xTurnover_calibrated', 'sum')
        ).reset_index()
        consistency = match_level.groupby('player_id').agg(
            xTurnover_per_match_mean=('match_xTurnover', 'mean'),
            xTurnover_per_match_std=('match_xTurnover', 'std')
        ).fillna(0)
        consistency['xTurnover_cv'] = consistency['xTurnover_per_match_std'] / consistency['xTurnover_per_match_mean'].replace(0, 1)
        metrics_df = metrics_df.join(consistency)

        # Negative Impact
        neg_mask = (marginal_df['raw_shapley_value'] < 0)
        neg_stats = marginal_df[neg_mask].groupby('player_id').agg(
            negative_chains=('global_chain_id', 'count'),
            total_negative_impact=('raw_shapley_value', 'sum'),
            avg_negative_impact=('raw_shapley_value', 'mean')
        )
        metrics_df = metrics_df.join(neg_stats).fillna(0)
        metrics_df['negative_impact_per_90'] = metrics_df['total_negative_impact'] * per_90_factor

        # Chain Quality
        chain_quality = marginal_df.groupby('player_id').agg(
            avg_chain_xTurnover_full=('xTurnover_full', 'mean'),
            max_chain_xTurnover_full=('xTurnover_full', 'max'),
            std_chain_xTurnover_full=('xTurnover_full', 'std')
        ).fillna(0)
        metrics_df = metrics_df.join(chain_quality)

        metrics_df['xTurnover_conversion_rate'] = metrics_df['attributed_turnovers'] / metrics_df['total_marginal_xTurnover_calibrated'].replace(0, 1)
        metrics_df['pressing_risk'] = (metrics_df['defensive_penalty_per_90'] / metrics_df['xTurnover_per_90'].replace(0, 1))*1000

        # Rename defensive penalty for ID match
        metrics_df.rename(columns={'defensive_penalty': 'total_defensive_penalty'}, inplace=True)
        metrics_df['total_defensive_penalty'] = metrics_df['total_defensive_penalty'].fillna(0)

        # Filter minimum minutes played
        metrics_df = metrics_df[metrics_df['minutes_played'] >= 900].copy()
        
        # Reset index to make player_id a column
        metrics_df = metrics_df.reset_index()
        return metrics_df

    @staticmethod
    def export_to_excel(sub_ranked_df: pd.DataFrame, output_path: str):
        print(f"\n📂 Exporting 8-Sheet Tactical Data to {output_path}...")
        sub_ranked_df = sub_ranked_df.sort_values(by='xTurnover_per_90', ascending=False)
        id_cols = ['player_name', 'team_name', 'minutes_played', 'matches_played']

        sheets = {
            "Player Rankings": ['player_id', 'player_name', 'chains_participated', 'total_marginal_xTurnover_calibrated', 'total_defensive_penalty', 'attributed_turnovers', 'avg_chain_size', 'minutes_played', 'matches_played', 'team_name', 'xTurnover_per_90', 'defensive_penalty_per_90', 'net_defensive_value_per_90', 'attributed_turnovers_per_90', 'value_added', 'value_added_per_90'],
            "All Sub-Metrics": [c for c in sub_ranked_df.columns],
            "Activity": id_cols + ['xTurnover_per_90', 'chains_participated', 'chains_per_90', 'chains_per_match', 'avg_chain_size'],
            "Efficiency": id_cols + ['xTurnover_per_90', 'xTurnover_per_chain', 'chains_per_90', 'avg_chain_size'],
            "Solo vs Coordinated": id_cols + ['xTurnover_per_90', 'solo_xTurnover_per_90', 'coordinated_xTurnover_per_90', 'solo_xTurnover_ratio', 'solo_chains', 'coordinated_chains', 'solo_xTurnover_total', 'coordinated_xTurnover_total'],
            "Dominance & Chain Quality": id_cols + ['xTurnover_per_90', 'avg_contribution_share', 'median_contribution_share', 'avg_chain_xTurnover_full', 'max_chain_xTurnover_full', 'std_chain_xTurnover_full'],
            "Conversion & Consistency": id_cols + ['xTurnover_per_90', 'xTurnover_conversion_rate', 'attributed_turnovers', 'total_marginal_xTurnover_calibrated', 'xTurnover_per_match_mean', 'xTurnover_per_match_std', 'xTurnover_cv'],
            "Negative Impact": id_cols + ['xTurnover_per_90', 'negative_chains', 'total_negative_impact', 'net_defensive_value_per_90', 'defensive_penalty_per_90', 'avg_negative_impact', 'negative_impact_per_90', 'pressing_risk']
        }
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name, cols in sheets.items():
                existing_cols = [c for c in cols if c in sub_ranked_df.columns]
                df_sheet = sub_ranked_df[existing_cols].copy()
                df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
        print("✅ Excel Export Complete!")

# %%
# ==========================================
# 6. PIPELINE EXECUTION
# ==========================================

if __name__ == "__main__":
    print("🚀 Starting Chain-Level Model Processing...")

    import pickle
    import os
    
    cache_dir = CONFIG.OUTPUT_DIR / "pipeline_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "preprocessing_cache.pkl"

    if cache_file.exists():
        print(f"♻️  Loading preprocessed data from cache: {cache_file}")
        print("   (Delete this directory if you need to re-run the raw data extraction).")
        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)
        master_chain_df = cache_data['master_chain_df']
        master_engagement_df = cache_data['master_engagement_df']
        all_physical_stats = cache_data['all_physical_stats']
        player_chain_mapping = cache_data['player_chain_mapping']
        team_name_map = cache_data['team_name_map']
        player_team_map_list = cache_data['player_team_map_list']
    else:
        all_chain_features = []  
        all_engagement_store = []  
        all_physical_stats = []  
        player_chain_mapping = {}  
    
        team_name_map = {}
        player_team_map_list = []
    
        match_files = glob.glob(os.path.join(CONFIG.ROOT_DIR, "tracking_parquets", "*.parquet"))
    
        if not match_files:
            print("⚠️ Tracking Parquets missing! Attempting to auto-generate them...")
            conversion_script = CONFIG.BASE_DIR / "convert_tracking_JSON_to_parquets.py"
            if conversion_script.exists():
                import subprocess
                subprocess.run(["python", str(conversion_script)], check=True)
                match_files = glob.glob(os.path.join(CONFIG.ROOT_DIR, "tracking_parquets", "*.parquet"))
                if not match_files:
                    print("❌ ERROR: Still no parquet files found. Exiting.")
                    sys.exit(1)
            else:
                print(f"❌ ERROR: Script not found at {conversion_script}")
                sys.exit(1)
    
        for filepath in tqdm(match_files, desc="Compiling Data", unit="match"):
            filename = os.path.basename(filepath)
            match_id = filename.split('.')[0]
            try:
                track_path = filepath
                dyn_path = os.path.join(CONFIG.ROOT_DIR, "dynamic", f"{match_id}.parquet")
                meta_path = os.path.join(CONFIG.ROOT_DIR, "meta", f"{match_id}.json")
                phys_path = os.path.join(CONFIG.ROOT_DIR, "physical", f"{match_id}.parquet")
                
                result = MatchProcessor.process_single_match(match_id, track_path, dyn_path, meta_path, phys_path, fps=CONFIG.FPS)
                
                if result is None: continue
                    
                all_chain_features.append(result['chain_features'])
                all_engagement_store.append(result['engagement_store'])
                if result['physical_stats'] is not None:
                    all_physical_stats.append(result['physical_stats'])
                player_team_map_list.append(result['player_team_map'])
                team_name_map.update(result['team_names'])
    
                for chain_id, mapping in result['player_chain_map'].items():
                    player_chain_mapping[f"{match_id}_{chain_id}"] = mapping
    
            except Exception as e:
                tqdm.write(f"Error in {match_id}: {e}")
    
        print("\n🔄 Compiling Chain-Level Dataset...")
        master_chain_df = pd.concat(all_chain_features, ignore_index=True).copy()
        master_engagement_df = pd.concat(all_engagement_store, ignore_index=True).copy()
    
        master_chain_df['global_chain_id'] = master_chain_df['match_id'].astype(str) + "_" + master_chain_df['pressing_chain_index'].astype(str)
        master_engagement_df['global_chain_id'] = master_engagement_df['match_id'].astype(str) + "_" + master_engagement_df['pressing_chain_index'].astype(str)

        print("\n💾 Saving preprocessed data to cache...")
        cache_data = {
            'master_chain_df': master_chain_df,
            'master_engagement_df': master_engagement_df,
            'all_physical_stats': all_physical_stats,
            'player_chain_mapping': player_chain_mapping,
            'team_name_map': team_name_map,
            'player_team_map_list': player_team_map_list
        }
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)
        print("✅ Cache saved successfully!")

    print(f"\n📊 Dataset Statistics:")
    print(f"   Total Chains: {len(master_chain_df):,}")
    print(f"   Total Engagements: {len(master_engagement_df):,}")
    
    # ====================================================================
    # 🧪 FEATURE EXPERIMENTATION ZONE
    # Add any columns you want to strictly EXCLUDE from the model below.
    # ====================================================================
    exclude_cols = ['global_chain_id', 'chain_success', 'conceded_xT', 'match_id', 
                    'game_state', 'third_start', 'is_coordinated_press', 'engagement_density',
                    'chain_size'] # Excluded due to 0.88 correlation with chain_size
                    
    feature_cols = [c for c in master_chain_df.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols if master_chain_df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    print(f"\n🧩 Selected Features for XGBoost ({len(feature_cols)}):")
    for f in feature_cols:
        print(f"   - {f}")

    master_chain_df[feature_cols] = master_chain_df[feature_cols].fillna(0)
    X = master_chain_df[feature_cols]
    y = master_chain_df['chain_success']

    weight_ratio = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n🗺️ GENERATING FEATURE CORRELATION HEATMAP...")
    try:
        import seaborn as sns
        corr_matrix = master_chain_df[feature_cols].corr(method='spearman')
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        plt.figure(figsize=(18, 14))
        cmap = sns.diverging_palette(20, 220, as_cmap=True)
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .7}, 
                    annot=True, fmt=".2f", annot_kws={"size": 8})
        plt.title('Spearman Correlation Matrix of Pressing Chain Features', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("   ⚠️ Seaborn is not installed. To view the heatmap, run: pip install seaborn")
    print("="*80)

    print("\n🏋️ Training XGBoost Model for Chain-Level Turnover Prediction...")
    trainer = TurnoverModel(feature_cols=feature_cols, weight_ratio=weight_ratio)
    trainer.train_model(X_train, y_train)
    trainer.evaluate(X_test, y_test)
    trainer.explain_with_shap(X_test)
    
    shapley_engine = ShapleyAttributionEngine()
    marginal_df = shapley_engine.calculate_exact_shapley(
        model=trainer.model,
        master_chain_df=master_chain_df,
        master_engagement_df=master_engagement_df,
        player_chain_mapping=player_chain_mapping,
        feature_cols=feature_cols
    )
    
    print(f"\n✅ Calculated {len(marginal_df):,} player-chain marginal contributions")
    
    print("\n🔀 Applying Residual Distribution (Total Value Recovery)...")
    print(f"   Max per-chain residual after redistribution: < 1e-6")
    
    multi_player_mask = marginal_df['chain_size'] > 1
    if multi_player_mask.any():
        print("\n⏱️  Temporal Weight Stats (multi-player chains only):")
        print(f"   Mean weight:   {marginal_df.loc[multi_player_mask, 'temporal_weight'].mean():.3f}")
        print(f"   Median weight: {marginal_df.loc[multi_player_mask, 'temporal_weight'].median():.3f}")
        print(f"   Min weight:    {marginal_df.loc[multi_player_mask, 'temporal_weight'].min():.3f}")
        print(f"   Floor applied: {shapley_engine.temporal_weight_floor}")

    print("\n📊 Attribution Summary:")
    sum_full = marginal_df.drop_duplicates('global_chain_id')['xTurnover_full'].sum()
    sum_marg = marginal_df['marginal_xTurnover_calibrated'].sum()
    total_attributed_turnovers = marginal_df['attributed_turnover'].sum()
    total_actual_turnovers = master_chain_df['chain_success'].sum()
    print(f"   Sum of xTurnover_full (all chains):        {sum_full:.2f}")
    print(f"   Sum of residual-adjusted marginals:        {sum_marg:.2f}")
    print(f"   Total Value Recovery check (diff):         {abs(sum_full - sum_marg):.4f}")
    print(f"   Total Attributed Turnovers:                {total_attributed_turnovers:.2f}")
    print(f"   Total Actual Turnovers:                    {total_actual_turnovers}")
    print(f"   Attribution Efficiency:                    {(total_attributed_turnovers/total_actual_turnovers)*100 if total_actual_turnovers > 0 else 0:.1f}%")
    print(f"   Total Defensive Penalty (Risk Exposure):   {marginal_df['defensive_penalty'].sum():.2f}")
    print(f"\n✅ Season Total xTurnover (attributed): {sum_marg:.2f}")

    print("\n" + "-"*75)
    print("Top 10 Players with most negative pressing contributions:")
    print("-" * 75)
    neg_mask = (marginal_df['raw_shapley_value'] < 0)
    neg_stats = marginal_df[neg_mask].groupby('player_name').agg(
        negative_chains=('global_chain_id', 'count'),
        avg_negative_impact=('raw_shapley_value', 'mean'),
        worst_single_mistake=('raw_shapley_value', 'min')
    ).sort_values('negative_chains', ascending=False).head(10)
    print(neg_stats.to_string(formatters={'avg_negative_impact': '{:.4f}'.format, 'worst_single_mistake': '{:.4f}'.format}))

    print("\n✅ CHAIN-LEVEL SHAPLEY MODEL COMPLETE!")
    output_path_chains = CONFIG.OUTPUT_DIR / "xTurnover_chains.parquet"
    master_chain_df.to_parquet(output_path_chains, index=False)

    print("\n👑 Calculating Player Rankings (CALIBRATED VALUES)...")
    player_stats = marginal_df.groupby('player_id').agg(
        player_name=('player_name', 'first'),
        total_marginal_xTurnover_calibrated=('marginal_xTurnover_calibrated', 'sum'),
        attributed_turnovers=('attributed_turnover', 'sum'),
        defensive_penalty=('defensive_penalty', 'sum'),
        chains_participated=('global_chain_id', 'nunique'),
        avg_chain_size=('chain_size', 'mean')
    ).reset_index()
    
    player_stats = player_stats.sort_values('total_marginal_xTurnover_calibrated', ascending=False)
    
    # Generate and Export Sub-Metrics
    print("\n" + "="*120)
    print("GENERATING FULL TACTICAL O.O.P METRICS AND EXPORTING")
    print("="*120)
    master_physical_df = pd.concat(all_physical_stats, ignore_index=True) if all_physical_stats else None
    
    sub_metrics_df = MetricsAndExportEngine.calculate_sub_metrics(
        marginal_df=marginal_df,
        ranking_df=player_stats,
        physical_stats_df=master_physical_df,
        team_name_map=team_name_map,
        player_team_map_list=player_team_map_list
    )
    
    # Display top players from the extended metrics dataset
    print("\n============================================================================================================================================")
    print("TOP 20 PLAYERS - CALIBRATED MARGINAL xTO PER 90 (HIGH BLOCK PRESSING)")
    print("============================================================================================================================================")
    top_cols = ['player_name', 'team_name', 'minutes_played', 'chains_participated', 
                'total_marginal_xTurnover_calibrated', 'xTurnover_per_90', 
                'defensive_penalty_per_90', 'net_defensive_value_per_90', 
                'attributed_turnovers_per_90', 'value_added_per_90']
    
    display_df = sub_metrics_df.sort_values('xTurnover_per_90', ascending=False)
    
    # Check what columns actually exist
    existing_top_cols = [c for c in top_cols if c in display_df.columns]
    
    # Map the output to match user request visually if possible
    # Rename xTurnover_per_90 to marginal_xTurnover_per_90 just for display
    print_df = display_df[existing_top_cols].head(20).copy()
    if 'xTurnover_per_90' in print_df.columns:
        print_df = print_df.rename(columns={'xTurnover_per_90': 'marginal_xTurnover_per_90'})
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(print_df.to_string(index=False))
    
    print("\n================================================================================")
    print("CALIBRATION QUALITY CHECK - Top 10 Players")
    print("================================================================================")
    print("Expected vs Actual turnovers (should be close for well-calibrated model)")
    print("-" * 80)
    
    calib_df = display_df[['player_name', 'total_marginal_xTurnover_calibrated', 'attributed_turnovers']].head(10).copy()
    calib_df.columns = ['Player', 'Expected', 'Actual']
    calib_df['Difference'] = calib_df['Actual'] - calib_df['Expected']
    print(calib_df.to_string(index=False))
    
    mean_abs_diff = calib_df['Difference'].abs().mean()
    mean_diff = calib_df['Difference'].mean()
    
    print(f"\nMean Absolute Difference: {mean_abs_diff:.2f}")
    print(f"Mean Difference: {mean_diff:.2f} (positive = outperformed, negative = underperformed)")

    output_path_players = CONFIG.OUTPUT_DIR / "player_xTurnover_rankings.parquet"
    player_stats.to_parquet(output_path_players, index=False)

    total_attributed_marg_file = CONFIG.OUTPUT_DIR / "xTurnover_marginal_contributions_Shapley.parquet"
    marginal_df.to_parquet(total_attributed_marg_file, index=False)
    
    print("\n💾 Saving outputs...")
    print(f"   Marginal contributions saved to: {total_attributed_marg_file}")
    print(f"   Chain-level data saved to: {output_path_chains}")
    print(f"   Player aggregations saved to: {output_path_players}")

    print("\n✅ CHAIN-LEVEL SHAPLEY MODEL COMPLETE!")
    print("📝 Model Architecture: Chain-level xTurnover with Counterfactual Shapley Marginal Attribution")
    print("📊 Use 'total_marginal_xTurnover_calibrated' or 'xTurnover_per_90' column for rankings and visualizations")
    print("📊 Both raw and calibrated values saved to output files")

    excel_output_path = CONFIG.OUTPUT_DIR / "Player_xTurnover_Advanced_Metrics.xlsx"
    MetricsAndExportEngine.export_to_excel(sub_metrics_df, str(excel_output_path))
    
    submetrics_parquet_path = CONFIG.OUTPUT_DIR / "player_xTurnover_submetrics.parquet"
    sub_metrics_df.to_parquet(submetrics_parquet_path, index=False)
    print(f"💾 Full Sub-Metrics Parquet saved to: {submetrics_parquet_path}")

    print("\n✨ PIPELINE EXECUTION FINISHED SUCCESSFULLY ✨")
# %%
