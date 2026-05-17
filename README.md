# Biomechanics to Basketball Intelligence: SHAP-Driven Shooting Corrections via 3D Visualization
## UCSB Data Science Capstone 2025-2026
### Anna Gornyitzki, Phillip Gurevich, Jay Leung, Sophie Lian, Shahil Patel

End-to-end pipeline that turns marker-based biomechanics into per-shot coaching cues and renders them in a Three.js web visualizer. The system combines model explainability (SHAP) with bounded counterfactual search (DiCE) to suggest realistic pose adjustments for predicted-miss shots.

---

## Project Overview

This project analyzes high-dimensional shooting biomechanics, learns what drives makes vs misses, and produces actionable, bounded pose recommendations. The output is consumed by an interactive Three.js visualizer that overlays original vs recommended mechanics for each shot.

Core ideas:
- **Predictive model**: A Made/Miss classifier trained on engineered biomechanical and context features.
- **Attribution**: SHAP identifies which biomechanical features most hurt a shot.
- **Recommendation**: DiCE searches for the smallest feasible changes that flip a miss into a make, constrained by anatomical and per-shot caps.
- **Visualization**: A web-based Three.js viewer plays back the original skeleton and the recommended corrections.

---

## SHAP + DiCE Recommendation Pipeline

### 1) Data preparation
- Load `capstone2026v2.csv` and standardize handedness (left-handed shots mirrored into a dominant-hand frame).
- Clean sentinel values (`|x| > 1e6`), apply shot-type filters, and add context features (shot distance/type/year/time).
- Engineer features:
	- **Raw biomechanics** (angles, velocities, ROM, alignment)
	- **Within-player z-scores** for personalization
	- **Dominant vs non-dominant symmetry diffs**

### 2) Stage 1 (diagnostic)
Biomechanics → ball-flight regressors (LightGBM) for four metrics. This stage is diagnostic and validates signal quality; it does not feed recommendations.

### 3) Made/Miss classifier (drives recommendations)
Train an `LGBMClassifier` on the full feature matrix `X_full`. This model is the target for SHAP and DiCE downstream. The model is always retrained to stay in sync with the current feature schema.

### 4) SHAP feature attribution
For each predicted-miss shot, SHAP surfaces the **most harmful biomechanical features**. Context-only features (shot type, distance, year, etc.) and derived features (`__zwithin`, `DomDiff`) are filtered out so the coaching cues are strictly biomechanical.

### 5) DiCE bounded counterfactual search
DiCE Random searches for the smallest set of feature changes that flips a miss → make. The search is constrained by:
- **Anatomical bounds** per joint (physiologic limits)
- **Per-player feasibility** ranges
- **Per-shot delta caps** (e.g., ±15 deg for angles, ±20% for velocities, ±20 deg for ROM)

Only features DiCE actually moves are exported as coaching cues.

---

## Visualization (Three.js)

The web visualizer renders:
- **Original pose** (measured biomechanics)
- **Recommended pose** (DiCE counterfactual)
- **Per-shot feature deltas** for quick coaching interpretation

The Three.js app loads `skeleton_exports/skeleton_export_<id>.csv` and plays back shot frames with overlays for original vs recommended mechanics.

---

## Output Files

### Primary (used by Three.js)
`skeleton_exports/skeleton_export_<id>.csv`

Each row is a shot with a variable number of features (only those DiCE moved):
- **Metadata**: `PlayerId`, `ShotId`, `Shot.Type`, `hand`, `is_make`
- **Ranked features**: `feat_1..feat_n`
- **Values**: `orig_<rank>`, `rec_<rank>`, `delta_<rank>`

Wide schema that includes every pose column. Kept for backward compatibility.

---

## How to Run

### 1) Generate exports (notebook)
Open and run the notebook:

- `shooting-lab-viz/shap_analysis_final_optimized.ipynb`

Key settings:
- `RUN_ALL_PLAYERS = True` to batch process all players
- Single-player diagnostic mode defaults to `Player 163`

Outputs are written to:
- `shooting-lab-viz/skeleton_exports/`

### 2) Run the Three.js visualizer

From the `shooting-lab-viz/` folder:

1. Install dependencies
	 - `npm install`
2. Start the dev server
	 - `npm run dev`
3. Open the local URL printed in the terminal

The app loads data from:
- `shooting-lab-viz/public/data/`

Ensure the latest `skeleton_exports/` CSVs are in `public/data/skeleton_exports/` before launching the visualizer.

---

## Screenshot

<img width="1385" height="692" alt="Screenshot 2026-05-17 at 9 23 05 AM" src="https://github.com/user-attachments/assets/d6270bb5-2661-4a73-9c00-8148f74ee5ac" />

---

## Project Structure (high level)

- `shooting-lab-viz/` — Three.js visualization app
	- `public/data/` — CSV data served to the app
	- `skeleton_exports/` — notebook export output (move to public/data for viewing)
- `shap_analysis.ipynb` — full SHAP + DiCE pipeline
- `capstone2026v2.csv` — master biomechanics dataset

---

## Notes

- The visualizer consumes **skeleton export** files only.
- If DiCE fails to find counterfactuals too often, loosen `delta_cap_for_feature` in the notebook.
- Context features are used in modeling but excluded from recommendations so only biomechanical changes are surfaced.
