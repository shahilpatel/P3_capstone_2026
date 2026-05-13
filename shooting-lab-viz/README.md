# Shooting Lab Visualizer

A real-time 3D basketball biomechanics analysis tool that visualizes shooting form across four phases (PreHitch, Hitch, PostHitch, Release) with SHAP-based correction recommendations.

## What This App Does

The Shooting Lab Visualizer takes biomechanical data from basketball players' shots and renders interactive 3D skeletons showing:

1. **Four-phase breakdown** — Each shooting phase displays a skeleton with precise joint angles (knees, hips, shoulders, elbows, ankles, torso)
2. **Ball trajectory** — The ball position at each phase, plus a projected arc to the rim
3. **SHAP corrections** — Machine learning-based biomechanical recommendations overlaid as a semi-transparent amber "ghost" skeleton
4. **Metadata** — Player name, shot date, shot type, location, make/miss, and correction count
5. **Sidebar annotations** — Text descriptions of each correction with original angle → recommended angle → delta

### Key Features

- **Interactive camera** — Orbit controls to rotate and zoom the 3D view
- **Correction overlay** — Click "Show SHAP Correction" to toggle between the actual pose and the recommended corrected pose
- **Player/shot selection** — Dropdown to browse all players and all their shots
- **Visual indicators** — [A] badge on shots that have SHAP analysis available
- **Make/Miss indicators** — ✓ (made) or ✗ (missed) badges on each shot

---

## How to Run

### Prerequisites
- Node.js 18+
- npm

### Setup & Start

```bash
# Install dependencies
npm install

# Development server (with hot reload)
npm run dev

# Open browser to: http://localhost:5173
```

The dev server runs on port 5173 and serves the app with Vite's fast refresh.

### File Locations

- **Main CSV data**: `/public/data/capstone2026v2.csv` (main shot metadata)
- **SHAP export data**: `/public/data/all_skeleton_exports.csv` (biomech corrections)
- **Skeleton exports** (individual): `/public/data/skeleton_exports/*.csv` (165 player files)

---

## Architecture & File Overview

### Frontend Files

#### **main.js** (Entry Point & UI Logic)
- **Role**: Application orchestrator
- **Responsibilities**:
  - Loads CSV data via PapaParse
  - Creates 4 Three.js canvases (one per phase)
  - Manages player/shot selection and dropdown UI
  - Handles SHAP correction toggle
  - Updates sidebar annotations
  - Orchestrates data flow to visualization engine
- **Key Variables**:
  - `csvRows`: All shots from capstone2026v2.csv
  - `exportRows`: All SHAP corrections from all_skeleton_exports.csv
  - `currentRow`: The currently selected shot
  - `currentExportRow`: The SHAP data for the current shot
  - `currentParsed`: Parsed corrections and annotations
  - `correctionActive`: Boolean toggle state for SHAP overlay
  - `filteredRows`: Shots for the selected player
- **Key Functions**:
  - `selectPlayer(name)`: Filter shots by player, rebuild dropdown
  - `selectShot(idx)`: Load a specific shot's data and pose
  - `hasShapData(playerName, shotId)`: Check if SHAP data exists
  - `refresh()`: Re-render all 4 phases with current pose/corrections
  - `updateSidebar()`: Display correction annotations
- **UI Elements**: Player select, shot select, toggle button, badges, sidebar panels

#### **poseEngine.js** (Skeleton Posing)
- **Role**: Apply CSV biomechanical data to a skeleton's bones
- **Responsibilities**:
  - Maps bone names to Three.js skeleton hierarchy
  - Caches rest pose so skeletons can reset
  - Applies phase-specific rotations to bones (knee flexion, hip flexion, etc.)
  - Applies SHAP correction deltas on top of base pose
  - Computes ball trajectory (3D arc)
- **Key Functions**:
  - `indexBones(rootObject)`: Build name→bone lookup
  - `cacheRestPose(boneIndex)`: Store rest pose so we can return to it
  - `resetToRestPose(boneIndex)`: Restore all bones to rest position
  - `applyPose(row, phase, boneIndex, corrections)`: **Core function**
    - Reads CSV columns like `maxElbowFlexionLeftPostHitch`
    - Converts degrees → radians, applies sign/scale rules
    - Rotates the bone on its axis (x/y/z)
    - If `corrections` provided, adds delta before rotating
  - `computeBallTrajectory(row)`: Calculate 3D points for ball path
- **Data Flow**: CSV row → Phase → BONE_MAPPING rules → Bone rotations

#### **stickFigure.js** (3D Skeleton Model)
- **Role**: Build the Three.js skeleton mesh
- **Responsibilities**:
  - Creates a humanoid skeleton from cylinders and spheres
  - Builds a hierarchy of bones (hips → spine → shoulders → arms/legs)
  - Colors bones (blue for main skeleton, amber for ghost)
  - Exports bone names for later lookup
- **Key Functions**:
  - `buildStickFigure()`: Create skeleton, return { root, boneIndex }
  - Creates ~30 bones with anatomically correct naming (leftShoulder, rightElbow, etc.)
- **Note**: This is a procedurally-built skeleton. Real data uses this same hierarchy.

#### **boneMapping.js** (Biomechanics → Skeleton)
- **Role**: Map CSV column names to skeleton rotations
- **Responsibilities**:
  - Defines 4 phase blocks: PreHitch, Hitch, PostHitch, Release
  - Each phase has 10-12 rules mapping CSV columns to bones
  - Each rule specifies: column name, bone name, axis (x/y/z), sign, optional scale
- **Examples**:
  ```javascript
  { column: 'maxElbowFlexionLeftPostHitch', bone: 'leftElbow', axis: 'x', sign: +1 }
  // Reads CSV column → rotates leftElbow on X axis → degrees to radians
  ```
- **Sign Corrections**: Verified from actual CSV data to ensure joints bend correctly
  - Knee flexion: negative CSV values + sign +1 = backward swing (correct)
  - Hip flexion: positive CSV values + sign +1 = forward bend (correct)
  - Torso: small negative lean + sign +1 = backward tilt (correct)
- **Handedness Note**: Data reads `hand` column at runtime (always 'Left' in this dataset, but code supports both)

#### **shapBridge.js** (SHAP Data Transform)
- **Role**: Translate ML recommendations into skeleton rotations
- **Responsibilities**:
  - Parse skeleton export rows (feat_1 → feat_5, with orig/rec/delta columns)
  - Map feature names to bones (e.g., "LeftDominantKneeFlexion" → leftKnee)
  - Separate visualizable corrections (joint angles) from text-only annotations
  - Match main CSV shots to SHAP export rows by PlayerId + ShotId
- **Key Functions**:
  - `findExportForShot(exportRows, mainRow)`: Match PlayerId + ShotId
  - `parseExportRow(row)`: Extract corrections, annotations, changed bones
  - `featToBone(featName, hand)`: Map feature → bone (handles Dominant/NonDominant)
- **Data Format**:
  - Main CSV: `Name` (e.g., "Player 1"), auto-calculated `ShotId` (1, 2, 3...)
  - Export CSV: `PlayerId` (e.g., "Player 1"), `ShotId` (1, 2, 3...), `hand`, `feat_1–5`, `orig_1–5`, `rec_1–5`, `delta_1–5`
- **SHAP Logic**:
  - For each shot, machine learning identified up to 5 key biomechanical adjustments
  - `delta` = recommended angle change (in degrees)
  - Ghost skeleton applies these deltas on top of actual pose

#### **index.html** (UI Layout & Styling)
- **Role**: DOM structure and styling
- **Layout**:
  - Left sidebar: player/shot controls + correction toggle + badges
  - Right sidebar: correction annotations (list of recommendations)
  - Center: 4 canvas panels (2×2 grid) for PreHitch, Hitch, PostHitch, Release
  - Progress bars show phase transitions
- **Styling**: Dark theme (charcoal bg), accent colors (blue for UI, amber for ghost, orange for ball)
- **Interactivity**: Form controls, button state management

---

## Data Pipeline

### Input Data

**1. capstone2026v2.csv** (Main Shot Metadata)
- 25,868 rows (all shots from 163 players)
- Columns: `Name` (player), `Date`, `Shot.Type`, `Shot.Location`, `Made`, `hand`, biomech columns (maxElbowFlexion, etc.), ball position (BallPosx, BallPosy, etc.) at each phase
- One row per shot
- **Role**: Base shot data, biomechanical measurements, make/miss labels

**2. all_skeleton_exports.csv** (SHAP Recommendations)
- 10,987 rows (subset of capstone rows with ML recommendations)
- Columns: `PlayerId`, `ShotId`, `hand`, `feat_1–5` (feature names), `orig_1–5` (original angles), `rec_1–5` (recommended angles), `delta_1–5` (angle changes)
- Only shots that the ML model flagged as improvable (outside optimal 40–60th percentile of made shots)
- **Role**: Machine learning recommendations, linked by PlayerId + ShotId

**3. skeleton_exports/*.csv** (Individual Player Exports)
- 165 files, one per player (e.g., `skeleton_export_Player 1.csv`)
- 10,987 rows total (merged into all_skeleton_exports.csv)
- Same format as all_skeleton_exports.csv
- **Role**: Source data before merge; kept for reference

### Data Processing (Backend)

**In main.js at startup**:

1. **Load capstone2026v2.csv**
   - Parse all 25,868 rows via PapaParse
   - Pre-calculate `ShotId` for each player:
     - Reset counter to 0 when player changes
     - Increment and assign ShotId 1, 2, 3... per player
   - Store in `csvRows`

2. **Load all_skeleton_exports.csv**
   - Parse all 10,987 SHAP recommendation rows
   - Store in `exportRows`
   - Each row includes `PlayerId` ("Player 1") and `ShotId` ("1", "2", etc.)

3. **Match & Display**
   - When user selects a player: filter csvRows by name
   - When user selects a shot: 
     - Get PlayerId from csvRows
     - Look up matching row in exportRows using `findExportForShot()`
     - If match found: parse SHAP data, show [A] badge
     - If no match: show shot without corrections

### Data Matching (Critical for SHAP)

```javascript
// The matching logic in shapBridge.js
function findExportForShot(exportRows, mainRow) {
  const playerName = mainRow.Name || '';  // e.g., "Player 1"
  const shotId = mainRow.ShotId ? mainRow.ShotId.toString() : '';  // e.g., "1"
  return exportRows.find(r => {
    const pId = r.PlayerId ? r.PlayerId.trim() : '';
    const sId = r.ShotId ? r.ShotId.trim() : '';
    return pId === playerName && sId === shotId;
  }) || null;
}
```

**Why it's tricky**:
- Main CSV has `Name` (player), export CSV has `PlayerId` (same value, different column)
- Both must match **exactly** (including whitespace, capitalization)
- ShotId must be pre-calculated consistently in both datasets
- Only 10,987 out of 25,868 shots have SHAP data (intentional; ML only targets improvable shots)

---

## UI Flow

### Initial Load
```
User opens app
  → main.js loads capstone2026v2.csv (25,868 rows, pre-calculate ShotId)
  → main.js loads all_skeleton_exports.csv (10,987 rows)
  → Player dropdown populated with all 163 unique player names
  → Select first player
```

### Player Selection
```
User selects player (e.g., "Player 1")
  → filteredRows = all shots for that player (e.g., 42 shots)
  → Shot dropdown rebuilt with all 42 shots
  → For each shot, hasShapData() checks if [A] badge should display
  → First shot auto-selected
```

### Shot Selection
```
User selects a shot (e.g., Shot 5)
  → currentRow = that shot's data
  → Look up PlayerId + ShotId in exportRows via findExportForShot()
  → If found: currentExportRow = SHAP data, currentParsed = corrections
  → If not found: currentExportRow = null, no corrections available
  → Display metadata: player, date, shot type, location
  → Display badges: Made/Missed, correction count (if SHAP exists)
  → Call refresh() to render 4 phases with actual pose
  → Ghost skeleton hidden by default
  → Sidebar shows correction annotations (if any)
```

### Correction Toggle
```
User clicks "Show SHAP Correction"
  → correctionActive = true
  → Call refresh()
  → For each phase panel:
    - Apply actual pose to main skeleton (blue)
    - Apply actual pose + deltas to ghost skeleton (amber)
    - Ghost skeleton becomes visible with glow effect
  → User can rotate camera to compare poses side-by-side
```

### Sidebar
```
Displays list of corrections for current shot (if SHAP data exists)
- Each correction shows: feature name, original angle, recommended angle, delta
- Visualizable corrections (joint angles) in full detail
- Non-visualizable corrections (text descriptions) in italics
```

---

## Backend Rendering Pipeline

### Per-Frame Render

**1. refresh() function**:
```javascript
for each of 4 phase panels:
  1. Reset skeleton to rest pose
  2. For that phase, apply base pose from currentRow
     - Read CSV columns for that phase
     - Apply rotations to skeleton bones
  3. Position ball at that phase
  4. Compute and draw trajectory arc
  5. If correctionActive && SHAP data exists:
     - Reset ghost skeleton to rest pose
     - Apply base pose from currentRow to ghost
     - Apply SHAP deltas on top
     - Ghost becomes visible
  6. Render scene (Three.js draws the frame)
```

**2. applyPose() core logic** (in poseEngine.js):
```javascript
for each bone mapping rule for this phase:
  1. Read CSV column value (e.g., maxElbowFlexionLeftPostHitch)
  2. Parse as float (angle in degrees)
  3. If corrections object has a key for this column, add the delta
  4. Convert to radians: radians = degrees * sign * scale * DEG2RAD
  5. Get the bone object from boneIndex
  6. Apply rotation on specified axis:
     bone.rotation.x (or y/z) = radians
```

**3. Three.js Rendering**:
- Each Panel has a WebGLRenderer
- Scene contains: ambient light, 3 directional lights, main skeleton, ghost skeleton (when visible), ball, trajectory line, grid
- OrbitControls allow user to rotate camera around target (hips at height 1.1)
- Renderer draws each phase independently

---

## Current Issues & Next Steps

### Issue 1: SHAP Data Only Shows for Player 1 ❌

**Symptom**: Only Player 1's shots show [A] badge; other players show no SHAP data despite having records in all_skeleton_exports.csv

**Root Cause** (under investigation): 
- `hasShapData()` is being called but returning false for other players
- Possible causes:
  - PlayerId format mismatch (e.g., "Player 1" vs "Player 1 " with spaces)
  - ShotId mismatch (integer vs string, off-by-one, etc.)
  - exportRows not fully loading (truncated CSV, parsing error)

**Debug Steps**:
1. Open browser console (F12)
2. Select different players and shots
3. Watch console for debug logs showing:
   - `Loaded X skeleton export rows` (should be 10,987)
   - `hasShapData(Player X, Y) = false` (investigate why for Player X)
   - Sample PlayerId and ShotId values from exportRows
4. Compare with expected values:
   - PlayerId should be "Player 1", "Player 163", etc.
   - ShotId should be "1", "2", "3", etc. (1-based, reset per player)

**Fix Strategy**:
- Add `.trim()` to both sides of comparison
- Verify notebook pre-calculation of ShotId matches frontend calculation
- Check for encoding issues (UTF-8 vs ASCII, BOMs, etc.)
- May need to re-export all_skeleton_exports.csv if corruption suspected

---

### Issue 2: Ghost Skeleton Doesn't Show Corrections ❌

**Symptom**: When clicking "Show SHAP Correction", button color changes but skeleton appears identical; no visible delta applied

**Root Cause** (under investigation):
- Ghost skeleton is toggled to visible, but corrections not being applied
- Possible causes:
  - `showGhost()` not being called correctly
  - `applyPose()` not accepting corrections parameter properly
  - Corrections object is empty or malformed
  - Ghost skeleton mesh not updating (physics cache issue)

**Debug Steps**:
1. In browser console, add breakpoint in `refresh()` when `correctionActive === true`
2. Log: `currentParsed.corrections` (should have keys like "maxElbowFlexionLeftPostHitch")
3. Log: `Object.keys(currentParsed.corrections).length` (should be > 0)
4. Check if `showGhost()` is being called (add console.log)
5. Verify ghost bone rotations are actually different from main skeleton

**Fix Strategy**:
- Verify parseExportRow() is correctly extracting deltas
- Ensure applyPose() applies corrections to the correct axis/sign
- May need to explicitly update ghost skeleton's matrix after applying corrections
- Consider adding temporary highlight color to ghost bones to verify they're moving

---

### Issue 3: Release Phase Pose Too Horizontal ⚠️

**Symptom**: At Release phase, skeleton's shooting arm is too horizontal (should be more upright); elbow looks too extended

**Root Cause** (suspected):
- Elbow flexion column reading is incorrect (could be extension instead)
- Sign inversion issue (applying negative when should be positive)
- CSV data for Release phase is noisy or wrong
- boneMapping.js has wrong rule for Release elbow

**Investigation Needed**:
1. Open a shot in the app
2. Look at Release phase (rightmost canvas, assuming right-handed)
3. Check what angle is being applied:
   - In browser console: log `currentRow['maxElbowFlexionRightRelease']`
   - This should be ~20–40° (arm extended but not fully straight)
4. Compare with other phases to see if Release is an outlier
5. Check actual CSV data directly:
   ```bash
   head -5 public/data/capstone2026v2.csv | tr ',' '\n' | grep -i "release"
   ```

**Potential Fixes**:
- Check if Release phase uses a different column (e.g., "ElbowExtension" vs "ElbowFlexion")
- Verify sign in boneMapping.js for Release → should be +1 for flexion (bend), -1 for extension (straighten)
- If CSV data is wrong, may need data cleaning in pandas notebook
- Consider adding scale factor to reduce magnitude if angles are too large

---

## Next Steps (Priority Order)

### 1. Debug SHAP Data Matching (CRITICAL)
**Goal**: Get [A] badge showing for all 158 players with SHAP data

**Tasks**:
- [ ] Add console logging to hasShapData() to see exact PlayerId/ShotId comparisons
- [ ] Dump sample rows from exportRows and csvRows in console
- [ ] Check for whitespace, encoding, or type mismatches
- [ ] Verify notebook's ShotId calculation matches frontend's pre-calculation
- [ ] Re-export all_skeleton_exports.csv if corruption suspected
- [ ] Test with Player 163 (highest SHAP count, ~1,436 rows)

**Success Criteria**: All players with SHAP data show [A] badge on applicable shots

---

### 2. Fix Ghost Skeleton Visualization (CRITICAL)
**Goal**: Ghost skeleton shows visible correction deltas when toggled

**Tasks**:
- [ ] Log corrections object to ensure it has keys
- [ ] Verify showGhost() is actually being called
- [ ] Check that applyPose() is applying corrections to ghost (not overwriting them)
- [ ] Ensure ghost skeleton's matrix is updated after pose application
- [ ] Test with a shot known to have SHAP data (Player 1)
- [ ] May need to call ghost skeleton's `.updateMatrixWorld()` after pose

**Success Criteria**: When toggle clicked, ghost skeleton visibly differs from main skeleton with amber glow

---

### 3. Fix Release Phase Pose (MEDIUM PRIORITY)
**Goal**: Shooting arm position at Release is anatomically correct (upright, elbow extended ~20–40°)

**Tasks**:
- [ ] Inspect Release phase CSV columns for right arm
- [ ] Compare Release elbow angle vs PostHitch (should go from ~90° to ~20–40°)
- [ ] Check sign/scale in boneMapping.js for Release
- [ ] Test with multiple players to see if issue is consistent or data-specific
- [ ] Consider adding visual debugging (angle overlay in UI)

**Success Criteria**: Release pose shows arm extending upward and forward, not too horizontal

---

### 4. Add More SHAP Coverage Info (NICE TO HAVE)
**Goal**: Users understand why only 42% of shots have SHAP data

**Tasks**:
- [ ] Add tooltip or help text explaining SHAP coverage
- [ ] Show player's SHAP coverage percentage (X shots with analysis out of Y total)
- [ ] Maybe add a stat somewhere like "158 of 163 players have analysis"

**Success Criteria**: Users understand SHAP is selective, not broken

---

## Debugging Checklist

**When something doesn't work**:

1. **Open browser console** (F12 → Console tab)
2. **Check for loading errors** — look for red messages
3. **Look at debug logs**:
   - `Loaded X skeleton export rows` (should see count)
   - `Player: X, ShotId: Y` (should see correct values)
   - `Found SHAP data: true/false` (indicates match result)
   - Sample export rows printed at startup
4. **Check Network tab** (F12 → Network) — verify CSVs are downloaded
5. **Use `debugger;`** statement to pause in poseEngine.js, check bone rotations
6. **Log intermediate values** in the functions you're debugging

---

## File Tree

```
/Users/annagornyitzki/shooting-lab-viz/
├── README.md                          # This file
├── package.json                       # Dependencies, scripts
├── main.js                            # Entry point, UI orchestration
├── poseEngine.js                      # Skeleton posing logic
├── stickFigure.js                     # 3D skeleton model builder
├── boneMapping.js                     # CSV column → bone rotation rules
├── shapBridge.js                      # SHAP data transformation
├── index.html                         # DOM structure & styling
├── public/
│   ├── data/
│   │   ├── capstone2026v2.csv         # Main shot data (25,868 rows)
│   │   ├── all_skeleton_exports.csv   # SHAP recommendations (10,987 rows)
│   │   └── skeleton_exports/          # Individual player exports (165 files)
│   └── models/                        # Reserved for future imported models
├── data_investigation.ipynb           # Jupyter notebook for data merging/debugging
└── shap_analysis_clean.ipynb          # Original SHAP analysis pipeline
```

---

## Questions?

- **How do I see what bones are being moved?** → Look at boneMapping.js; each rule lists the bone name
- **How do corrections work?** → poseEngine.js `applyPose()` adds `corrections[columnName]` to the angle
- **Why are some shots missing SHAP data?** → Only improvable shots outside the optimal 40–60th percentile get recommendations
- **Can I add new phases?** → Yes, add to PHASES array in boneMapping.js and extend canvas grid in index.html
- **Why is the ghost skeleton amber?** → Visual distinction from main skeleton; helps see the delta when both are visible

---

## Credits & Data

- **Biomechanical Data**: Basketball shooting motion captures, 163 players
- **SHAP Analysis**: Machine learning-based recommendations from `shap_analysis_clean.ipynb`
- **Visualization**: Three.js 3D rendering, Vite build tool, PapaParse CSV parsing
