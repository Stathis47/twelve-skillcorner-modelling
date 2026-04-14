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
