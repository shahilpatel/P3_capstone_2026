// main.js — single-canvas stop-motion scrubber with SHAP correction overlay
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import Papa from 'papaparse';

import { buildStickFigure } from './stickFigure.js';
import { cacheRestPose, applyPoseInterpolated, computeBallTrajectory } from './poseEngine.js';
import { parseExportRow, findExportForShot } from './shapBridge.js';

const BG_COLOR    = 0x1e1e22;
const GRID_MAIN   = 0x2e2e33;
const GRID_SUB    = 0x242428;
const BALL_COLOR  = 0xe87a2e;
const TRAJ_COLOR  = 0x3b82f6;
const GHOST_COLOR = 0xf59e0b;
const VISUAL_THRESHOLD = 3; // degrees below which joint delta won't be visible on skeleton

class Panel {
  constructor(canvas) {
    this.canvas = canvas;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setClearColor(BG_COLOR, 1);

    this.scene = new THREE.Scene();
    this.scene.add(new THREE.AmbientLight(0xdde0e8, 0.5));
    const key = new THREE.DirectionalLight(0xfff5ee, 1.1);
    key.position.set(3, 6, 3); this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xc8d4f0, 0.4);
    fill.position.set(-3, 2, -2); this.scene.add(fill);
    const rim = new THREE.DirectionalLight(0xffffff, 0.25);
    rim.position.set(0, -2, -4); this.scene.add(rim);

    const grid = new THREE.GridHelper(5, 10, GRID_MAIN, GRID_SUB);
    grid.material.opacity = 0.5; grid.material.transparent = true;
    this.scene.add(grid);

    this.camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
    this.camera.position.set(4.8, 1.8, 1.2);
    this.camera.lookAt(0, 1.1, 0);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 1.1, 0);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.update();

    // Main skeleton
    const { root, boneIndex } = buildStickFigure();
    this.figure = root;
    this.boneIndex = boneIndex;
    this.scene.add(root);
    cacheRestPose(boneIndex);

    // Ghost skeleton for SHAP corrections (amber, initially hidden)
    const { root: ghostRoot, boneIndex: ghostBoneIndex } = buildStickFigure();
    this.ghostFigure = ghostRoot;
    this.ghostBoneIndex = ghostBoneIndex;
    this.ghostFigure.visible = false;
    this.ghostFigure.traverse(obj => {
      if (obj.isMesh && obj.material) {
        obj.material = obj.material.clone();
        obj.material.color.setHex(GHOST_COLOR);
        obj.material.emissive.setHex(0x996600);
        obj.material.emissiveIntensity = 0.4;
        obj.material.transparent = true;
        obj.material.opacity = 0.8;
      }
    });
    this.scene.add(ghostRoot);
    cacheRestPose(ghostBoneIndex);

    // Ball mesh
    const ballGeom = new THREE.SphereGeometry(0.10, 32, 20);
    const ballMat = new THREE.MeshStandardMaterial({ color: BALL_COLOR, roughness: 0.55, metalness: 0.05 });
    this.ballMesh = new THREE.Mesh(ballGeom, ballMat);
    this.ballMesh.visible = false;
    this.scene.add(this.ballMesh);

    this.trajectoryLine = null;
  }

  setBallPosition(row) {
    const hand = (row.hand || 'Right').toString().toLowerCase().startsWith('l')
      ? 'leftHandTip' : 'rightHandTip';
    const anchor = this.boneIndex[hand];
    if (!anchor) { this.ballMesh.visible = false; return; }
    this.figure.updateMatrixWorld(true);
    const tmp = new THREE.Vector3();
    anchor.getWorldPosition(tmp);
    this.ballMesh.position.set(tmp.x, tmp.y, tmp.z - 0.05);
    this.ballMesh.visible = true;
  }

  setBallWorldPosition(pos) {
    if (!pos) { this.ballMesh.visible = false; return; }
    this.ballMesh.position.copy(pos);
    this.ballMesh.visible = true;
  }

  showGhost(row, corrections, t) {
    this.ghostFigure.visible = true;
    applyPoseInterpolated(row, t, this.ghostBoneIndex, corrections);
  }

  hideGhost() {
    this.ghostFigure.visible = false;
  }

  setTrajectory(points, anchor = null) {
    if (this.trajectoryLine) {
      this.scene.remove(this.trajectoryLine);
      this.trajectoryLine.geometry.dispose();
      this.trajectoryLine.material.dispose();
      this.trajectoryLine = null;
    }
    if (!points || points.length < 2) return;
    const base = anchor ? anchor.clone() : this.ballMesh.position.clone();
    const offset = points[0].clone();
    const pts = points.map(p => p.clone().sub(offset).add(base));
    const geom = new THREE.BufferGeometry().setFromPoints(pts);
    const mat = new THREE.LineBasicMaterial({ color: TRAJ_COLOR });
    this.trajectoryLine = new THREE.Line(geom, mat);
    this.scene.add(this.trajectoryLine);
  }

  resize() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  render() {
    this.controls.update();
    if (this.figure.userData.updateLines) this.figure.userData.updateLines();
    if (this.ghostFigure.visible && this.ghostFigure.userData.updateLines) {
      this.ghostFigure.userData.updateLines();
    }
    this.renderer.render(this.scene, this.camera);
  }
}

// --- Bootstrap ---

const panel = new Panel(document.getElementById('canvas-main'));

const statusEl       = document.getElementById('status');
const metaEl         = document.getElementById('meta');
const badgeEl        = document.getElementById('madeBadge');
const corrBadgeEl    = document.getElementById('correctionBadge');
const shapBadgeEl    = document.getElementById('shapBadge');
const playerSelect   = document.getElementById('playerSelect');
const shotSelect     = document.getElementById('shotSelect');
const toggleBtn      = document.getElementById('toggleCorrection');
const sidebarSub     = document.getElementById('sidebarSub');
const sidebarContent = document.getElementById('sidebarContent');
const slider         = document.getElementById('timelineSlider');
const phaseLabelEl   = document.getElementById('phaseLabel');
const playBtn        = document.getElementById('playBtn');
const playHint       = document.getElementById('playHint');
const speed075Btn    = document.getElementById('speed075');
const speed05Btn     = document.getElementById('speed05');
const speed025Btn    = document.getElementById('speed025');

let csvRows = [], filteredRows = [], currentRow = null;
let exportRows = [];
let correctionActive = false;
let currentExportRow = null;
let currentParsed = null;
let currentT = 0;
let currentDetach = null;
let currentTrajectory = [];
let isPlaying = false;
let playStartMs = 0;
let playDuration = 2.0;
let playRaf = null;
let speedMultiplier = 1.0;

const SPEED_BUTTONS = [
  { btn: speed075Btn, value: 0.75 },
  { btn: speed05Btn, value: 0.5 },
  { btn: speed025Btn, value: 0.25 },
];

function getNumber(row, keys) {
  for (const key of keys) {
    const raw = row[key];
    if (raw === undefined || raw === null || raw === '') continue;
    const n = parseFloat(raw);
    if (Number.isFinite(n)) return n;
  }
  return null;
}

function getTotalShotTime(row) {
  const total = getNumber(row, [
    'TotalShotTime',
    'Total Shot Time',
  ]);
  if (total && total > 0) return total;

  const pre = getNumber(row, ['Pre-Hitch Time', 'PreHitchTime']);
  const hitch = getNumber(row, ['Hitch Time', 'HitchTime']);
  const post = getNumber(row, ['Post Hitch Time', 'PostHitchTime']);
  const parts = [pre, hitch, post].filter(v => typeof v === 'number' && v > 0);
  if (parts.length > 0) return parts.reduce((a, b) => a + b, 0);

  return 2.0;
}

function isTimeFeature(name) {
  if (!name) return false;
  const n = name.toLowerCase();
  return n.includes('time');
}

function isVeloFeature(name) {
  if (!name) return false;
  const n = name.toLowerCase();
  return n.includes('velo') || n.includes('velocity');
}

function clamp(val, min, max) {
  return Math.min(max, Math.max(min, val));
}

function getPlaybackDuration(row, parsed, useCorrections) {
  const base = getTotalShotTime(row);
  if (!useCorrections || !parsed || !parsed.annotations || parsed.annotations.length === 0) {
    return base;
  }

  let duration = base;
  let veloFactor = 1.0;

  for (const ann of parsed.annotations) {
    if (isTimeFeature(ann.feat) && Number.isFinite(ann.delta)) {
      duration += ann.delta;
    } else if (isVeloFeature(ann.feat) && Number.isFinite(ann.delta)) {
      const denom = Math.max(Math.abs(ann.orig || 0), 1e-3);
      const pct = ann.delta / denom;
      veloFactor *= 1 / (1 + pct);
    }
  }

  duration *= clamp(veloFactor, 0.5, 1.5);
  return clamp(duration, 0.5, 6.0);
}

function setPlayState(next) {
  isPlaying = next;
  playBtn.textContent = isPlaying ? 'Pause' : 'Play';
  playBtn.classList.toggle('active', isPlaying);
}

function setSpeedMultiplier(mult) {
  speedMultiplier = mult;
  SPEED_BUTTONS.forEach(({ btn, value }) => {
    if (!btn) return;
    btn.classList.toggle('active', value === speedMultiplier);
  });
  if (currentRow) {
    playDuration = getPlaybackDuration(currentRow, currentParsed, correctionActive) / speedMultiplier;
    playHint.textContent = `Auto · ${playDuration.toFixed(2)}s`;
  }
}

function stopPlayback() {
  if (playRaf) cancelAnimationFrame(playRaf);
  playRaf = null;
  setPlayState(false);
}

function tickPlayback(nowMs) {
  const elapsed = (nowMs - playStartMs) / 1000;
  const t = Math.min(3, (elapsed / playDuration) * 3);
  currentT = t;
  slider.value = currentT;
  updatePhaseLabel(currentT);
  refresh();

  if (t >= 3) {
    stopPlayback();
    return;
  }
  playRaf = requestAnimationFrame(tickPlayback);
}

function startPlayback() {
  if (!currentRow) return;
  currentT = 0;
  slider.value = 0;
  updatePhaseLabel(0);
  refresh();
  playDuration = getPlaybackDuration(currentRow, currentParsed, correctionActive) / speedMultiplier;
  playHint.textContent = `Auto · ${playDuration.toFixed(2)}s`;
  playStartMs = performance.now();
  setPlayState(true);
  playRaf = requestAnimationFrame(tickPlayback);
}

function updatePhaseLabel(t) {
  const phases = ['Pre-Hitch', 'Hitch', 'Post-Hitch', 'Release'];
  const idx = Math.min(3, Math.floor(t + 0.01));
  phaseLabelEl.textContent = phases[idx];
}

// Load main CSV
Papa.parse('/data/capstone2026v2.csv', {
  download: true, header: true, skipEmptyLines: true,
  complete: ({ data }) => {
    const playerShotCounts = {};
    csvRows = data.map(row => {
      const playerName = row.Name;
      if (!playerName) return row;
      playerShotCounts[playerName] = (playerShotCounts[playerName] || 0) + 1;
      return { ...row, ShotId: playerShotCounts[playerName] };
    });

    const players = [...new Set(csvRows.map(r => r.Name).filter(Boolean))].sort((a, b) => {
      const numA = parseInt(a.replace(/\D/g, '')) || 0;
      const numB = parseInt(b.replace(/\D/g, '')) || 0;
      return numA - numB || a.localeCompare(b);
    });
    playerSelect.innerHTML = '';
    players.forEach(name => {
      const opt = document.createElement('option');
      opt.value = name; opt.textContent = name;
      playerSelect.appendChild(opt);
    });
    statusEl.textContent = `${csvRows.length.toLocaleString()} shots · ${players.length} players`;
    selectPlayer(players[0]);
  },
  error: err => { console.error(err); statusEl.textContent = 'CSV load failed.'; },
});

// Load skeleton exports
// Papa.parse('/data/all_skeleton_exports.csv', {
Papa.parse('/data/skeleton_exports/skeleton_export_Player 1.csv', {
  download: true, header: true, skipEmptyLines: true,
  complete: ({ data }) => {
    exportRows = data;
    console.log(`Loaded ${exportRows.length} skeleton export rows`);
    if (exportRows.length > 0) {
      console.log('Sample export row:', exportRows[0]);
      console.log('Sample PlayerId values:', exportRows.slice(0, 5).map(r => r.PlayerId));
      console.log('Sample ShotId values:', exportRows.slice(0, 5).map(r => r.ShotId));
    }
  },
  error: () => { console.log('No skeleton exports found — corrections unavailable.'); },
});

function hasShapData(playerName, shotId) {
  return exportRows.some(r =>
    r.PlayerId === playerName && r.ShotId === String(shotId)
  );
}

function selectPlayer(name) {
  filteredRows = csvRows.filter(r => r.Name === name);
  shotSelect.innerHTML = '';
  filteredRows.forEach((row, i) => {
    const made = (row.Made || '').toString().toUpperCase() === 'TRUE';
    const hasSHAP = hasShapData(name, row.ShotId);
    const shapInd = hasSHAP ? ' [A]' : '';
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = `Shot ${i + 1} ${made ? '✓' : '✗'}${shapInd}  —  ${row['Shot.Location'] || ''}  —  ${row['Shot.Type'] || ''}`;
    shotSelect.appendChild(opt);
  });
  if (filteredRows.length > 0) selectShot(0);
}

playerSelect.addEventListener('change', e => selectPlayer(e.target.value));

function selectShot(idx) {
  currentRow = filteredRows[idx];
  if (!currentRow) return;

  console.log(`\n=== Selected Shot ===`);
  console.log(`Player: ${currentRow.Name}, ShotId: ${currentRow.ShotId}`);

  const made = (currentRow.Made || '').toString().toUpperCase() === 'TRUE';
  metaEl.innerHTML =
    `<span class="val">${currentRow.Name || ''}</span> · ` +
    `${currentRow.Date || ''} · ` +
    `<span class="val">${currentRow['Shot.Type'] || ''}</span> · ` +
    `<span class="val">${currentRow['Shot.Location'] || ''}</span>`;
  badgeEl.innerHTML = made
    ? `<span class="badge made">Made</span>`
    : `<span class="badge missed">Missed</span>`;

  currentExportRow = findExportForShot(exportRows, currentRow);
  console.log(`Found SHAP data: ${!!currentExportRow}`);
  currentParsed = currentExportRow ? parseExportRow(currentExportRow) : null;

  const hasShap = !!currentExportRow;
  shapBadgeEl.innerHTML = hasShap
    ? `<span class="badge shap-available">Correction Available</span>`
    : `<span class="badge shap-none">No Correction</span>`;

  if (currentParsed && currentParsed.annotations.length > 0) {
    corrBadgeEl.innerHTML = `<span class="badge correction">${currentParsed.annotations.length} corrections</span>`;
  } else {
    corrBadgeEl.innerHTML = '';
  }

  updateSidebar();

  if (correctionActive) {
    correctionActive = false;
    toggleBtn.textContent = 'Show Corrective Skeleton';
    toggleBtn.classList.remove('active');
  }

  // Reset slider to start of shot
  currentT = 0;
  slider.value = 0;
  updatePhaseLabel(0);

  currentDetach = computeDetachInfo(currentRow);
  currentTrajectory = computeBallTrajectory(currentRow);

  stopPlayback();
  playDuration = getPlaybackDuration(currentRow, currentParsed, correctionActive) / speedMultiplier;
  playHint.textContent = `Auto · ${playDuration.toFixed(2)}s`;

  refresh();
}

function updateSidebar() {
  if (!currentParsed || currentParsed.annotations.length === 0) {
    sidebarSub.textContent = 'No corrections for this shot';
    sidebarContent.innerHTML = '<div class="no-corrections">This shot has no SHAP corrections.<br>Try selecting a missed shot.</div>';
    return;
  }

  const vizCount = currentParsed.annotations.filter(a => a.visualizable).length;
  const txtCount = currentParsed.annotations.filter(a => !a.visualizable).length;
  sidebarSub.textContent = `${vizCount} on skeleton · ${txtCount} data-only`;

  let html = '';
  for (const ann of currentParsed.annotations) {
    const sign = ann.delta > 0 ? 'positive' : 'negative';
    const deltaStr = (ann.delta > 0 ? '+' : '') + ann.delta.toFixed(2);
    const cardClass = ann.visualizable ? 'visualizable' : 'text-only';
    const isMinimal = ann.visualizable && Math.abs(ann.delta) < VISUAL_THRESHOLD;

    const displayName = ann.feat
      .replace(/__zwithin$/, ' (z)')
      .replace(/DomDiff/, 'Δ')
      .replace(/Dominant/g, 'Dom')
      .replace(/NonDominant/g, 'Non');

    html += `<div class="correction-card ${cardClass}">
      <div class="cc-feat">${displayName}</div>
      <div class="cc-values">
        <div class="cc-val">from <span>${ann.orig.toFixed(2)}</span></div>
        <div class="cc-val">to <span>${ann.rec.toFixed(2)}</span></div>
        <div class="cc-delta ${sign}">${deltaStr}${ann.visualizable ? '°' : ''}</div>
      </div>
      ${ann.bone ? `<div class="cc-bone">${ann.bone}</div>` : ''}
      ${isMinimal ? `<div class="cc-minimal">Difference is minimal — may not appear on skeleton</div>` : ''}
    </div>`;
  }
  sidebarContent.innerHTML = html;
}

shotSelect.addEventListener('change', e => selectShot(parseInt(e.target.value, 10)));

function refresh() {
  if (!currentRow) return;

  applyPoseInterpolated(currentRow, currentT, panel.boneIndex, {});

  const detachT = currentDetach?.tDetach ?? 2.0;
  const anchor = currentDetach?.anchor ?? null;
  const hasTraj = currentTrajectory && currentTrajectory.length > 1;

  if (!hasTraj || currentT < detachT || !anchor) {
    panel.setBallPosition(currentRow);
    panel.setTrajectory([]);
  } else {
    const progress = Math.max(0, Math.min(1, (currentT - detachT) / (3 - detachT)));
    const idx = progress * (currentTrajectory.length - 1);
    const i0 = Math.floor(idx);
    const i1 = Math.min(i0 + 1, currentTrajectory.length - 1);
    const alpha = idx - i0;
    const p0 = currentTrajectory[i0];
    const p1 = currentTrajectory[i1];
    const offset = currentTrajectory[0];
    const p = new THREE.Vector3(
      p0.x + (p1.x - p0.x) * alpha,
      p0.y + (p1.y - p0.y) * alpha,
      p0.z + (p1.z - p0.z) * alpha,
    ).sub(offset).add(anchor);
    panel.setBallWorldPosition(p);
    panel.setTrajectory(currentTrajectory, anchor);
  }

  if (correctionActive && currentParsed && Object.keys(currentParsed.corrections).length > 0) {
    panel.showGhost(currentRow, currentParsed.corrections, currentT);
  } else {
    panel.hideGhost();
  }
}

function computeDetachInfo(row) {
  const isLeft = (row.hand || 'Right').toString().toLowerCase().startsWith('l');
  const handName = isLeft ? 'leftHandTip' : 'rightHandTip';
  const anchor = panel.boneIndex[handName];
  if (!anchor) return null;

  const samples = 31;
  let bestT = 2.0;
  let bestY = -Infinity;
  const tmp = new THREE.Vector3();

  for (let i = 0; i < samples; i++) {
    const t = (i / (samples - 1)) * 3.0;
    applyPoseInterpolated(row, t, panel.boneIndex, {});
    panel.figure.updateMatrixWorld(true);
    anchor.getWorldPosition(tmp);
    if (tmp.y > bestY) {
      bestY = tmp.y;
      bestT = t;
    }
  }

  applyPoseInterpolated(row, bestT, panel.boneIndex, {});
  panel.figure.updateMatrixWorld(true);
  anchor.getWorldPosition(tmp);

  return { tDetach: bestT, anchor: tmp.clone() };
}

// Timeline slider
slider.addEventListener('input', () => {
  currentT = parseFloat(slider.value);
  updatePhaseLabel(currentT);
  refresh();
  if (isPlaying) stopPlayback();
});

playBtn.addEventListener('click', () => {
  if (isPlaying) {
    stopPlayback();
  } else {
    startPlayback();
  }
});

toggleBtn.addEventListener('click', () => {
  if (!currentParsed || currentParsed.annotations.length === 0) return;
  correctionActive = !correctionActive;
  toggleBtn.textContent = correctionActive ? 'Hide Corrective Skeleton' : 'Show Corrective Skeleton';
  toggleBtn.classList.toggle('active', correctionActive);
  playDuration = getPlaybackDuration(currentRow, currentParsed, correctionActive) / speedMultiplier;
  playHint.textContent = `Auto · ${playDuration.toFixed(2)}s`;
  refresh();
});

SPEED_BUTTONS.forEach(({ btn, value }) => {
  if (!btn) return;
  btn.addEventListener('click', () => {
    setSpeedMultiplier(value);
    if (isPlaying) {
      stopPlayback();
      startPlayback();
    }
  });
});

setSpeedMultiplier(1.0);

// ← → scrub timeline; C toggle correction
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowRight') {
    currentT = Math.min(3, currentT + 0.15);
    slider.value = currentT;
    updatePhaseLabel(currentT);
    refresh();
  } else if (e.key === 'ArrowLeft') {
    currentT = Math.max(0, currentT - 0.15);
    slider.value = currentT;
    updatePhaseLabel(currentT);
    refresh();
  } else if (e.key === 'c' || e.key === 'C') {
    toggleBtn.click();
  }
});

window.addEventListener('resize', () => panel.resize());
panel.resize();

(function loop() {
  panel.render();
  requestAnimationFrame(loop);
})();
