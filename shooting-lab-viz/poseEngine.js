// poseEngine.js
//
// Functions that take a CSV row and apply rotations to a skeleton.
// Kept separate from Three.js scene management so it stays testable.

import * as THREE from 'three';
import { BONE_MAPPING, bonesUsed } from './boneMapping.js';

const deg2rad = THREE.MathUtils.degToRad;

// Walk a model and build a name→bone lookup.
// Mixamo models nest the skeleton inside Armature > Hips > ... so we recurse.
export function indexBones(rootObject) {
  const index = {};
  rootObject.traverse(obj => {
    if (obj.isBone) {
      index[obj.name] = obj;
    }
  });
  return index;
}

// Reset every bone the mapping touches back to its rest pose.
// Rest pose rotations are stored on first call so we can return to them later.
const restPoseCache = new WeakMap();

export function cacheRestPose(boneIndex) {
  const snapshot = {};
  for (const name in boneIndex) {
    const b = boneIndex[name];
    snapshot[name] = { x: b.rotation.x, y: b.rotation.y, z: b.rotation.z };
  }
  restPoseCache.set(boneIndex, snapshot);
}

export function resetToRestPose(boneIndex) {
  const snap = restPoseCache.get(boneIndex);
  if (!snap) return;
  for (const name in snap) {
    const b = boneIndex[name];
    b.rotation.x = snap[name].x;
    b.rotation.y = snap[name].y;
    b.rotation.z = snap[name].z;
  }
}

// Apply one phase's pose to a skeleton.
// row: a CSV row object (keys = column names, values = strings or numbers)
// phase: 'PreHitch' | 'Hitch' | 'PostHitch' | 'Release'
// boneIndex: result of indexBones()
// corrections: optional { columnName: deltaDegrees } to apply on top
export function applyPose(row, phase, boneIndex, corrections = {}) {
  resetToRestPose(boneIndex);

  const rules = BONE_MAPPING[phase];
  if (!rules) {
    console.warn(`No mapping for phase "${phase}"`);
    return;
  }

  for (const rule of rules) {
    const raw = row[rule.column];
    if (raw === undefined || raw === '' || raw === null) continue;
    let deg = parseFloat(raw);
    if (Number.isNaN(deg)) continue;

    // Apply any correction delta for this column.
    if (corrections[rule.column] !== undefined) {
      deg += corrections[rule.column];
    }

    // Apply sign, optional scale, and convert to radians.
    const scale = rule.scale !== undefined ? rule.scale : 1.0;
    const rad = deg2rad(deg * rule.sign * scale);

    const bone = boneIndex[rule.bone];
    if (!bone) {
      // Print once per missing bone, not every frame.
      if (!applyPose._warned) applyPose._warned = new Set();
      if (!applyPose._warned.has(rule.bone)) {
        console.warn(`Bone not found: ${rule.bone}`);
        applyPose._warned.add(rule.bone);
      }
      continue;
    }

    // Additive: start from rest, add the rotation on the named axis.
    bone.rotation[rule.axis] += rad;
  }
}

// Interpolate between two adjacent phase keyframes at position t in [0, 3].
// t=0 → PreHitch, t=1 → Hitch, t=2 → PostHitch, t=3 → Release.
// Bone angles are lerped so motion is continuous as the slider moves.
export function applyPoseInterpolated(row, t, boneIndex, corrections = {}) {
  resetToRestPose(boneIndex);

  const phaseNames = ['PreHitch', 'Hitch', 'PostHitch', 'Release'];
  t = Math.max(0, Math.min(3, t));

  const idxA = Math.floor(t);
  const idxB = Math.min(idxA + 1, 3);
  const alpha = t - idxA;

  const rulesA = BONE_MAPPING[phaseNames[idxA]] || [];
  const rulesB = BONE_MAPPING[phaseNames[idxB]] || [];

  // Build angle map keyed by "bone:axis" so both phases contribute to the same slot.
  const angles = {};

  for (const rule of rulesA) {
    const raw = row[rule.column];
    if (raw === undefined || raw === '' || raw === null) continue;
    let deg = parseFloat(raw);
    if (isNaN(deg)) continue;
    if (corrections[rule.column] !== undefined) deg += corrections[rule.column];
    const scale = rule.scale ?? 1.0;
    const rad = deg2rad(deg * rule.sign * scale);
    const key = `${rule.bone}:${rule.axis}`;
    angles[key] = { bone: rule.bone, axis: rule.axis, a: rad, b: 0 };
  }

  for (const rule of rulesB) {
    const raw = row[rule.column];
    if (raw === undefined || raw === '' || raw === null) continue;
    let deg = parseFloat(raw);
    if (isNaN(deg)) continue;
    if (corrections[rule.column] !== undefined) deg += corrections[rule.column];
    const scale = rule.scale ?? 1.0;
    const rad = deg2rad(deg * rule.sign * scale);
    const key = `${rule.bone}:${rule.axis}`;
    if (angles[key]) {
      angles[key].b = rad;
    } else {
      angles[key] = { bone: rule.bone, axis: rule.axis, a: 0, b: rad };
    }
  }

  for (const { bone: boneName, axis, a, b } of Object.values(angles)) {
    const bone = boneIndex[boneName];
    if (!bone) continue;
    bone.rotation[axis] += a + (b - a) * alpha;
  }
}

// Compute a ball trajectory from a CSV row.
// Returns an array of THREE.Vector3 points sampling the parabolic flight.
// Always aimed FORWARD (-Z in Three.js) from the player regardless of
// court position, so it looks correct from the side-view camera.
export function computeBallTrajectory(row, numPoints = 60) {
  const v0 = parseFloat(row['Initial.Ball.Velocity']);
  const angDeg = parseFloat(row['Initial.Ball.Angle']);

  if (Number.isNaN(v0) || Number.isNaN(angDeg)) return [];

  const ang = deg2rad(angDeg);
  const vHoriz = v0 * Math.cos(ang);
  const vVert = v0 * Math.sin(ang);
  const g = 9.81;

  // Flight time: solve for when ball descends to hoop height (3.048m / 10ft).
  // Start height ~2.4m (typical release). Use 2.4 as fallback.
  const startH = parseFloat(row['BallZRelease']) || 2.4;
  const targetH = 3.048;
  const a = -0.5 * g, b = vVert, c = startH - targetH;
  const disc = b * b - 4 * a * c;
  if (disc < 0) return [];
  const tFlight = (-b - Math.sqrt(disc)) / (2 * a);
  if (!isFinite(tFlight) || tFlight <= 0) return [];

  // Build arc: starts at origin (0,0,0), goes forward (-Z) and up (+Y).
  // main.js anchors the first point to the ball mesh position.
  const points = [];
  for (let i = 0; i <= numPoints; i++) {
    const t = (i / numPoints) * tFlight;
    const forward = vHoriz * t;
    const height = vVert * t - 0.5 * g * t * t;
    points.push(new THREE.Vector3(0, height, -forward));
  }
  return points;
}
