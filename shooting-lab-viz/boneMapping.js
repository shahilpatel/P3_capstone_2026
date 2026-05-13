// boneMapping.js (v9 — signs verified from CSV data)
//
// KEY FINDINGS FROM CSV INSPECTION:
//   - Player is LEFT-handed (hand = 'Left'). Left arm = shooting arm.
//   - Knee flexion values are NEGATIVE (e.g., -74°). Sign +1 is correct —
//     the negative CSV value produces a negative rotation which bends the
//     shin backward behind the thigh (anatomical flexion). Do NOT flip to -1.
//   - Hip flexion values are POSITIVE (~45°). Sign +1 is correct.
//   - Ankle dorsiflexion values are POSITIVE (~20-25°). Sign +1 correct.
//   - Torso flexion is NEGATIVE at PreHitch/Hitch (slight backward lean),
//     small magnitude. Sign +1 correct — produces small backward tilt.
//   - Shoulder flexion at PostHitch ~130° (both arms high — shooting form).
//   - Elbow flexion at PostHitch ~98° shooting arm (set position, correct).
//
// HANDEDNESS: The code reads `row.hand` at runtime to determine which is
// the shooting arm. For this dataset it's always Left, but the mapping
// applies the same columns to both sides regardless — the ball tracking
// in main.js handles which hand to follow.

export const PHASES = ['PreHitch', 'Hitch', 'PostHitch', 'Release'];

export const BONE_MAPPING = {
  PreHitch: [
    // Knee flexion: CSV negative values + sign +1 = negative rotation = shin swings backward ✓
    { column: 'maxKneeFlexionLeftPreHitch',  bone: 'leftKnee',  axis: 'x', sign: +1 },
    { column: 'maxKneeFlexionRightPreHitch', bone: 'rightKnee', axis: 'x', sign: +1 },
    // Hip flexion: positive values → sign +1
    { column: 'maxHipFlexionLeftPreHitch',  bone: 'leftHip',  axis: 'x', sign: +1 },
    { column: 'maxHipFlexionRightPreHitch', bone: 'rightHip', axis: 'x', sign: +1 },
    // Torso: negative = lean back, small values
    { column: 'TorsoFlexionAtT0', bone: 'spineLower', axis: 'x', sign: +1 },
    // Shoulder flexion: positive, ~64° at PreHitch (arm partway up)
    { column: 'maxShoulderFlexionLeftPreHitch',  bone: 'leftShoulder',  axis: 'x', sign: +1, scale: 0.7 },
    { column: 'maxShoulderFlexionRightPreHitch', bone: 'rightShoulder', axis: 'x', sign: +1, scale: 0.7 },
    // Elbow flexion: positive, ~100° at PreHitch
    { column: 'maxElbowFlexionLeftPreHitch',  bone: 'leftElbow',  axis: 'x', sign: +1 },
    { column: 'maxElbowFlexionRightPreHitch', bone: 'rightElbow', axis: 'x', sign: +1 },
    // Ankle dorsiflexion: positive values
    { column: 'maxAnkleDorsiflexionLeftPreHitch',  bone: 'leftAnkle',  axis: 'x', sign: +1 },
    { column: 'maxAnkleDorsiflexionRightPreHitch', bone: 'rightAnkle', axis: 'x', sign: +1 },
  ],
  Hitch: [
    { column: 'maxKneeFlexionLeftHitch',  bone: 'leftKnee',  axis: 'x', sign: +1 },
    { column: 'maxKneeFlexionRightHitch', bone: 'rightKnee', axis: 'x', sign: +1 },
    { column: 'maxHipFlexionLeftHitch',  bone: 'leftHip',  axis: 'x', sign: +1 },
    { column: 'maxHipFlexionRightHitch', bone: 'rightHip', axis: 'x', sign: +1 },
    { column: 'maxTorsoFlexionHitch', bone: 'spineLower', axis: 'x', sign: +1 },
    { column: 'maxShoulderFlexionLeftHitch',  bone: 'leftShoulder',  axis: 'x', sign: +1, scale: 0.7 },
    { column: 'maxShoulderFlexionRightHitch', bone: 'rightShoulder', axis: 'x', sign: +1, scale: 0.7 },
    { column: 'maxElbowFlexionLeftHitch',  bone: 'leftElbow',  axis: 'x', sign: +1 },
    { column: 'maxElbowFlexionRightHitch', bone: 'rightElbow', axis: 'x', sign: +1 },
    { column: 'maxAnkleDorsiflexionLeftHitch',  bone: 'leftAnkle',  axis: 'x', sign: +1 },
    { column: 'maxAnkleDorsiflexionRightHitch', bone: 'rightAnkle', axis: 'x', sign: +1 },
  ],
  PostHitch: [
    { column: 'maxKneeFlexionLeftPostHitch',  bone: 'leftKnee',  axis: 'x', sign: +1 },
    { column: 'maxKneeFlexionRightPostHitch', bone: 'rightKnee', axis: 'x', sign: +1 },
    { column: 'maxHipFlexionLeftPostHitch',  bone: 'leftHip',  axis: 'x', sign: +1 },
    { column: 'maxHipFlexionRightPostHitch', bone: 'rightHip', axis: 'x', sign: +1 },
    { column: 'maxTorsoFlexionPostHitch', bone: 'spineLower', axis: 'x', sign: +1 },
    { column: 'maxShoulderFlexionLeftPostHitch',  bone: 'leftShoulder',  axis: 'x', sign: +1, scale: 0.7 },
    { column: 'maxShoulderFlexionRightPostHitch', bone: 'rightShoulder', axis: 'x', sign: +1, scale: 0.7 },
    { column: 'maxElbowFlexionLeftPostHitch',  bone: 'leftElbow',  axis: 'x', sign: +1 },
    { column: 'maxElbowFlexionRightPostHitch', bone: 'rightElbow', axis: 'x', sign: +1 },
    { column: 'maxAnkleDorsiflexionLeftPostHitch',  bone: 'leftAnkle',  axis: 'x', sign: +1 },
    { column: 'maxAnkleDorsiflexionRightPostHitch', bone: 'rightAnkle', axis: 'x', sign: +1 },
  ],
  Release: [
    // At release: knees nearly extended (small negative flexion values)
    { column: 'maxKneeFlexionLeftPostHitch',  bone: 'leftKnee',  axis: 'x', sign: +1 },
    { column: 'maxKneeFlexionRightPostHitch', bone: 'rightKnee', axis: 'x', sign: +1 },
    { column: 'maxHipFlexionLeftPostHitch',  bone: 'leftHip',  axis: 'x', sign: +1 },
    { column: 'maxHipFlexionRightPostHitch', bone: 'rightHip', axis: 'x', sign: +1 },
    // Shooting arm: shoulder fully extended overhead (~132°), elbow nearly straight
    { column: 'maxShoulderFlexionLeftPostHitch',  bone: 'leftShoulder',  axis: 'x', sign: +1, scale: 0.7 },
    { column: 'maxShoulderFlexionRightPostHitch', bone: 'rightShoulder', axis: 'x', sign: +1, scale: 0.7 },
    // Elbow extension at release (~22°) — arm almost fully straight
    { column: 'maxElbowExtensionLeftPostHitch',  bone: 'leftElbow',  axis: 'x', sign: +1 },
    { column: 'maxElbowExtensionRightPostHitch', bone: 'rightElbow', axis: 'x', sign: +1 },
  ],
};

export function bonesUsed() {
  const set = new Set();
  Object.values(BONE_MAPPING).forEach(rules => {
    rules.forEach(r => set.add(r.bone));
  });
  return [...set];
}
