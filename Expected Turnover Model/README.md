# Master Thesis Project: Ajax Defensive Metrics (xTO)
Evaluating defensive actions in football using tracking data. Building models based on SkillCorner data in collaboration with Twelve Football.

## Project Structure & Setup

This repository contains the full automated pipeline for calculating and visualizing the Out-of-Possession (OOP) xTurnover models.

```text
Thesis Project Workspace/
├── Expected Turnover Model/
│   ├── convert_tracking_JSON_to_parquets.py   # Step 1: Data Conversion
│   ├── xTO_pipeline_refactored.py             # Step 2: ML Model & SHAP pipeline
│   ├── xto_tactical_dashboard.py              # Step 3: Streamlit tactical visuals
│   ├── requirements.txt
│   ├── .env.example                           # Setup your SkillCorner data path
│   └── README.md
```

## Setup Instructions

### 1. Environment and Dependencies
Ensure you have Python 3.9+ installed. Create your virtual environment and install the required libraries:

```bash
python -m venv .venv
# Activate the virtual environment:
# On Windows: .venv\Scripts\activate
# On Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure SkillCorner Data Path
You must have the SkillCorner tracking files locally. 
Create a new file named `.env` in the `Expected Turnover Model` folder with the following content:

```env
SKILLCORNER_DATA_DIR="C:/path/to/SkillcornerData/1/2024"
```

### 3. Execution Flow

#### Step 1: Extract JSON to Parquets
You must first convert the raw JSON tracking data into Parquets for performance.
```bash
python "Expected Turnover Model/convert_tracking_JSON_to_parquets.py"
```

#### Step 2: Run the Model Pipeline
Executes the physical extraction, spatial/chain aggregations, trains the XGBoost chain-level model, and calculates exact Shapley attributions.
```bash
python Expected Turnover Model/xTO_pipeline_refactored.py
```

#### Step 3: Launch the Dashboard
A local browser window will open automatically.
```bash
streamlit run Expected Turnover Model/xto_tactical_dashboard.py
```

## Dashboard User Guide

The **xTO Tactical Dashboard** is a Streamlit application designed for professional tactical analysis. It offers two main environments for exploring Out-of-Possession (OOP) performance: the **Chain Visualizer** (Film Room) and the **Player Comparison** (Scouting Room).

### 1. App Overview & Navigation
The dashboard aims to bridge the gap between complex mathematical physics engines and practical football coaching. It contains two main tabs:
- **🎬 Chain Visualizer (Film Room):** Replays specific multi-player pressing chains as animated 2D pitch views.
- **🔍 Player Comparison (Scouting Room):** Provides macroscopic positional-percentile radar charts based on six fundamental OOP pillars, alongside sortable metric tables.

### 2. Page 1: Pressing Chain Visualizer

The Chain Visualizer allows analysts to replay specific defensive pressing sequences (chains) as animated 2D pitch visualizations (GIFs). It provides immediate visual context to the mathematical models:
- **Marginal xTurnover Target Overlay:** The application dynamically renders the pressing sequence, highlighting the pressing cohort and explicitly placing the exact **Marginal xTO** above each player's head during the animated replay.
- **Contextual Awareness:** The tactical film room lists the exact Shapley distribution, game state, sequence duration, and whether the possession ended in an actual turnover or defensive failure.

### 3. Page 2: Player Radar Plots & Tactical Metrics

This section is the core of the scouting suite, comparing players across critical defensive KPIs using customized radar plots. Below is the tactical and mathematical glossary of the core metrics driving these visualizations:

#### Core xTO Return Metrics

- **xTurnover_per_90 (Total Pressing Reward)**
  - **Definition & Formula:** The sum of a player's calibrated marginal Shapley attributions across all full-chains, normalized per 90 minutes. 
  - **Footballing Intuition:** The ultimate measure of a player's off-ball defensive impact. High values indicate elite ball-winners and defensive catalysts who actively force turnovers.

- **defensive_penalty_per_90 (Expected Threat Leaked)**
  - **Definition & Formula:** The Expected Threat (xT) conceded by the team in the subsequent pass vectors immediately following a failed pressing chain, distributed to the player via Shapley values, and normalized per 90.
  - **Footballing Intuition:** Measures the direct defensive cost of breaking shape. High values indicate a player whose unsuccessful presses frequently expose the rest of the team to dangerous opposition attacks.

#### Volume & Basic Efficiency

- **chains_per_90 (Volume)**
  - **Definition & Formula:** Total number of distinct pressing chains a player actively engaged in per 90 minutes.
  - **Footballing Intuition:** Pure OOP work rate and stamina. It answers the question: "How physically active is the player out-of-possession?"

- **xTurnover_per_chain (Efficiency)**
  - **Definition & Formula:** Total Marginal xTO divided by the total number of chains participated in.
  - **Footballing Intuition:** Quality over quantity. Does the player make their pressing events count, or do they waste energy on low-probability, ineffective chasing?

#### Tactical Intelligence & Dominance

- **avg_contribution_share (Dominance)**
  - **Definition & Formula:** A player's average % share ($[Marginal\ xTO / Full\ Chain\ xTO]$) specifically in coordinated (multi-player) presses.
  - **Footballing Intuition:** Who is the alpha in the pressing pack? High scores indicate the player does the heavy lifting in group presses, anchoring the defensive action rather than just arriving as a bystander.

- **avg_chain_xTurnover_full (Tactical IQ)**
  - **Definition & Formula:** The average total raw xTO probability of the entire chains a player chooses to join or trigger.
  - **Footballing Intuition:** Measures reading of the game. Players with high scores trigger presses in highly advantageous geometric situations (e.g., trapping opponents on the touchline with limited passing options).

- **solo_xTurnover_ratio (Self-Sufficiency vs. Team Reliance)**
  - **Definition & Formula:** The percentage of a player's total xTurnover generated exclusively from solo presses (chain size = 1) versus coordinated, multi-player team presses.
  - **Footballing Intuition:** Differentiates system players from individual ball-winners. A low ratio indicates reliance on tactical group structures, while a high ratio highlights duel monsters who can disrupt the opponent unassisted.

- **negative_impact_per_90 (Shape Discipline)**
  - **Definition & Formula:** The sum of strictly negative Shapley values (instances where the player's presence mathematically *reduced* the chain's turnover probability compared to if they hadn't pressed) per 90 minutes.
  - **Footballing Intuition:** Identifies detrimental pressing. High values suggest a player frequently breaks the tactical shape, gets easily bypassed, or obstructs teammates' defensive geometry.

- **pressing_risk (Risk vs. Reward Efficiency)**
  - **Definition & Formula:** The ratio of Expected Threat conceded versus turnovers generated. Typically calculated as defensive penalty / xTurnover (per 1000).
  - **Footballing Intuition:** Evaluates defensive decision-making. A low score means the player wins the ball back safely without exposing the system. A high score implies a reckless pressing style—creating turnovers but leaking immense threat when bypassed.
