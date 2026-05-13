// shapBridge.js
//
// Translates SHAP skeleton export rows into corrections the skeleton can apply.
// Handles Dominant/NonDominant → Left/Right translation.

// Map feature name fragments to skeleton bone names
const BONE_MAP = [
  { match: ['knee','flexion'],     side: true,  bone: s => `${s}Knee` },
  { match: ['knee','extension'],   side: true,  bone: s => `${s}Knee` },
  { match: ['hip','flexion'],      side: true,  bone: s => `${s}Hip` },
  { match: ['shoulder','flexion'], side: true,  bone: s => `${s}Shoulder` },
  { match: ['shoulder','extension'],side: true, bone: s => `${s}Shoulder` },
  { match: ['elbow','flexion'],    side: true,  bone: s => `${s}Elbow` },
  { match: ['elbow','extension'],  side: true,  bone: s => `${s}Elbow` },
  { match: ['ankle','dorsiflexion'],side: true, bone: s => `${s}Ankle` },
  { match: ['ankle','plantarflexion'],side: true,bone: s => `${s}Ankle` },
  { match: ['torso','flexion'],    side: false, bone: () => 'spineLower' },
  { match: ['feetalignment'],     side: false, bone: () => 'hips' },
  { match: ['hipalignment'],      side: false, bone: () => 'hips' },
  { match: ['shoulderalignment'], side: false, bone: () => 'spineUpper' },
];

function dominantToSide(name, hand) {
  const isLeft = (hand || 'Right').toLowerCase().startsWith('l');
  if (name.includes('NonDominant')) {
    return { col: name.replace('NonDominant', isLeft ? 'Right' : 'Left'), side: isLeft ? 'right' : 'left' };
  }
  if (name.includes('Dominant')) {
    return { col: name.replace('Dominant', isLeft ? 'Left' : 'Right'), side: isLeft ? 'left' : 'right' };
  }
  if (name.includes('Avg')) {
    // Average of both sides — apply to dominant side for visualization
    return { col: name, side: isLeft ? 'left' : 'right' };
  }
  return { col: name, side: null };
}

function featToBone(featName, hand) {
  const { col, side } = dominantToSide(featName, hand);
  const lower = col.toLowerCase();

  for (const rule of BONE_MAP) {
    const matches = rule.match.every(m => lower.includes(m));
    if (!matches) continue;
    if (rule.side) {
      const s = side || 'right';
      return { bone: rule.bone(s), col };
    }
    return { bone: rule.bone(), col };
  }
  return null; // not visualizable
}

/**
 * Parse a skeleton export row into visualizable corrections and text annotations.
 *
 * @param {Object} row - one row from the skeleton export CSV
 * @returns {{ corrections: Object, annotations: Array, changedBones: string[] }}
 *   corrections: { csvColumnName: deltaDegrees } for applyPose
 *   annotations: [{ feat, orig, rec, delta }] for non-visualizable features
 *   changedBones: ['leftElbow', 'rightKnee', ...] for highlighting
 */
export function parseExportRow(row) {
  const hand = row.hand || 'Right';
  const corrections = {};
  const annotations = [];
  const changedBones = [];

  for (let i = 1; i <= 5; i++) {
    const feat  = row[`feat_${i}`];
    const delta = parseFloat(row[`delta_${i}`]);
    const orig  = parseFloat(row[`orig_${i}`]);
    const rec   = parseFloat(row[`rec_${i}`]);

    if (!feat || isNaN(delta) || Math.abs(delta) < 0.001) continue;

    const mapping = featToBone(feat, hand);

    if (mapping) {
      // Visualizable joint angle — find the matching boneMapping column
      const { bone, col } = mapping;
      corrections[col] = delta;
      if (!changedBones.includes(bone)) changedBones.push(bone);
      annotations.push({ feat, orig, rec, delta, visualizable: true, bone });
    } else {
      // Not visualizable — text annotation only
      annotations.push({ feat, orig, rec, delta, visualizable: false, bone: null });
    }
  }

  return { corrections, annotations, changedBones };
}

/**
 * Find the skeleton export row matching a shot from the main CSV.
 *
 * The main CSV has `Name` ("Player 101") and a `ShotId` we now pre-calculate.
 * The export CSV has `PlayerId` ("Player 101") and `ShotId`.
 * This function bridges the two.
 *
 * @param {Object[]} exportRows - all rows from skeleton export CSV
 * @param {Object} mainRow - current row from the main CSV
 * @returns {Object|null}
 */
export function findExportForShot(exportRows, mainRow) {
  if (!mainRow || !exportRows) return null;

  const playerName = mainRow.Name || '';
  const shotId = mainRow.ShotId ? mainRow.ShotId.toString() : '';

  if (!playerName || !shotId) return null;

  return exportRows.find(r => {
    const pId = r.PlayerId ? r.PlayerId.trim() : '';
    const sId = r.ShotId ? r.ShotId.trim() : '';
    return pId === playerName && sId === shotId;
  }) || null;
}
