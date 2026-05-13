// stickFigure.js — warm-white skeleton on cool-dark background
import * as THREE from 'three';

const L = {
  spineSeg: 0.13,
  neckSeg:  0.14,
  head:     0.09,
  shoulderHalfWidth: 0.20,
  upperArm: 0.30,
  forearm:  0.27,
  hand:     0.10,
  hipHalfWidth: 0.10,
  thigh:    0.45,
  shin:     0.42,
  foot:     0.20,
};

const LIMB_RADIUS  = 0.020;
const JOINT_RADIUS = 0.030;

// Warm white skeleton — reads as organic against cool UI chrome
export const LIMB_COLOR      = 0x3b82f6;
export const JOINT_COLOR     = 0x60a5fa;
export const HIGHLIGHT_COLOR = 0x3b82f6;  // accent blue for corrections

function pivot(name) {
  const o = new THREE.Object3D();
  o.name = name;
  return o;
}

function jointSphere() {
  return new THREE.Mesh(
    new THREE.SphereGeometry(JOINT_RADIUS, 16, 12),
    new THREE.MeshStandardMaterial({ color: JOINT_COLOR, roughness: 0.35, metalness: 0.05 })
  );
}

class Limb {
  constructor(parentName, childName, color = LIMB_COLOR, radius = LIMB_RADIUS) {
    this.parentName = parentName;
    this.childName  = childName;
    const geom = new THREE.CylinderGeometry(radius, radius, 1, 12);
    geom.translate(0, 0.5, 0);
    this.mat = new THREE.MeshStandardMaterial({
      color, roughness: 0.45, metalness: 0.05,
    });
    this.mesh   = new THREE.Mesh(geom, this.mat);
    this.tmpA   = new THREE.Vector3();
    this.tmpB   = new THREE.Vector3();
    this.tmpDir = new THREE.Vector3();
    this.upY    = new THREE.Vector3(0, 1, 0);
  }
  setColor(hex) { this.mat.color.setHex(hex); }
  update(boneIndex) {
    const pa = boneIndex[this.parentName];
    const cb = boneIndex[this.childName];
    if (!pa || !cb) return;
    pa.getWorldPosition(this.tmpA);
    cb.getWorldPosition(this.tmpB);
    this.tmpDir.subVectors(this.tmpB, this.tmpA);
    const len = this.tmpDir.length();
    if (len < 1e-6) { this.mesh.visible = false; return; }
    this.mesh.visible = true;
    this.mesh.position.copy(this.tmpA);
    this.mesh.quaternion.setFromUnitVectors(this.upY, this.tmpDir.clone().normalize());
    this.mesh.scale.set(1, len, 1);
  }
}

export function buildStickFigure() {
  const root = pivot('root');
  const hips = pivot('hips');
  hips.position.set(0, L.thigh + L.shin, 0);
  root.add(hips);

  const spineLower = pivot('spineLower'); hips.add(spineLower);
  const spineMid   = pivot('spineMid');   spineMid.position.set(0, L.spineSeg, 0); spineLower.add(spineMid);
  const spineUpper = pivot('spineUpper'); spineUpper.position.set(0, L.spineSeg, 0); spineMid.add(spineUpper);
  const neckBase   = pivot('neckBase');   neckBase.position.set(0, L.spineSeg, 0); spineUpper.add(neckBase);
  const head       = pivot('head');       head.position.set(0, L.neckSeg, 0); neckBase.add(head);

  function makeArm(side) {
    const sign = side === 'left' ? -1 : 1;
    const shoulder = pivot(side + 'Shoulder');
    shoulder.position.set(sign * L.shoulderHalfWidth, L.spineSeg, 0);
    spineUpper.add(shoulder);
    const elbow = pivot(side + 'Elbow');
    elbow.position.set(0, -L.upperArm, 0); shoulder.add(elbow);
    const wrist = pivot(side + 'Wrist');
    wrist.position.set(0, -L.forearm, 0); elbow.add(wrist);
    const handTip = pivot(side + 'HandTip');
    handTip.position.set(0, -L.hand, 0); wrist.add(handTip);
  }
  makeArm('left'); makeArm('right');

  function makeLeg(side) {
    const sign = side === 'left' ? -1 : 1;
    const hipJoint = pivot(side + 'Hip');
    hipJoint.position.set(sign * L.hipHalfWidth, 0, 0); hips.add(hipJoint);
    const knee = pivot(side + 'Knee');
    knee.position.set(0, -L.thigh, 0); hipJoint.add(knee);
    const ankle = pivot(side + 'Ankle');
    ankle.position.set(0, -L.shin, 0); knee.add(ankle);
    const toe = pivot(side + 'Toe');
    toe.position.set(0, 0, -L.foot); ankle.add(toe);
  }
  makeLeg('left'); makeLeg('right');

  const boneIndex = {};
  root.traverse(obj => { if (obj.name && obj !== root) boneIndex[obj.name] = obj; });

  const limbs = [
    new Limb('hips','spineLower'), new Limb('spineLower','spineMid'),
    new Limb('spineMid','spineUpper'), new Limb('spineUpper','neckBase'),
    new Limb('neckBase','head'),
    new Limb('leftShoulder','rightShoulder'),
    new Limb('leftShoulder','leftElbow'), new Limb('leftElbow','leftWrist'),
    new Limb('leftWrist','leftHandTip'),
    new Limb('rightShoulder','rightElbow'), new Limb('rightElbow','rightWrist'),
    new Limb('rightWrist','rightHandTip'),
    new Limb('leftHip','rightHip'),
    new Limb('leftHip','leftKnee'), new Limb('leftKnee','leftAnkle'),
    new Limb('leftAnkle','leftToe'),
    new Limb('rightHip','rightKnee'), new Limb('rightKnee','rightAnkle'),
    new Limb('rightAnkle','rightToe'),
  ];
  limbs.forEach(l => root.add(l.mesh));

  const jointNames = [
    'hips','spineMid','spineUpper','neckBase',
    'leftShoulder','rightShoulder','leftElbow','rightElbow',
    'leftWrist','rightWrist','leftHip','rightHip',
    'leftKnee','rightKnee','leftAnkle','rightAnkle',
  ];
  jointNames.forEach(n => { const j = boneIndex[n]; if (j) j.add(jointSphere()); });

  head.add(new THREE.Mesh(
    new THREE.SphereGeometry(L.head, 22, 16),
    new THREE.MeshStandardMaterial({ color: LIMB_COLOR, roughness: 0.45, metalness: 0.05 })
  ));

  function update() {
    root.updateMatrixWorld(true);
    for (const l of limbs) l.update(boneIndex);
  }
  update();

  root.userData.updateLines = update;
  return { root, boneIndex };
}
