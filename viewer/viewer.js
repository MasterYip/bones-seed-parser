/**
 * G1 robot Three.js motion viewer.
 * Served by _ViewerServer at /viewer.js — all fetches go to the same origin.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';

// The viewer page is served by the same server that hosts /g1.fbx, /anim-state, /csv
const SERVER = window.location.origin;

// ── G1Animation: parse CSV and interpolate frames ──────────────────────────
class G1Animation {
  constructor(csvText, name) {
    this.name    = name;
    this.columns = [];
    this.frames  = [];
    this.fps     = 120;
    this._parse(csvText);
    this.maxFrame = Math.max(0, this.frames.length - 1);
  }

  _parse(csv) {
    const lines = csv.trim().split('\n');
    if (lines.length < 2) return;
    this.columns = lines[0].split(',').map(s => s.trim());
    for (let i = 1; i < lines.length; i++) {
      const vals = lines[i].split(',');
      const fd = {};
      for (let j = 0; j < this.columns.length; j++)
        fd[this.columns[j]] = parseFloat(vals[j]) || 0;
      this.frames.push(fd);
    }
  }

  getFrameData(frame) {
    if (!this.frames.length) return null;
    const f  = Math.max(0, Math.min(frame, this.maxFrame));
    const f0 = Math.floor(f);
    const f1 = Math.min(Math.ceil(f), this.maxFrame);
    const t  = f - f0;
    if (t === 0 || f0 === f1) return Object.assign({}, this.frames[f0]);
    const a = this.frames[f0], b = this.frames[f1], r = {};
    for (const c of this.columns) r[c] = a[c] * (1 - t) + b[c] * t;
    return r;
  }
}

// ── Model3DG1: load FBX, convert SkinnedMeshes, apply PBR materials ────────
const DARK_PARTS = [
  'ankle_roll_link', 'rubber_hand', 'hip_pitch_link',
  '_pelvis_visual',  'head_link',   '_logo_link_visual',
];
const MAT_DARK  = new THREE.MeshStandardMaterial({
  color: new THREE.Color(0.02, 0.02, 0.03), metalness: 0.7, roughness: 0.15, envMapIntensity: 1.2,
});
const MAT_WHITE = new THREE.MeshStandardMaterial({
  color: new THREE.Color(0.82, 0.82, 0.84), metalness: 0.35, roughness: 0.3, envMapIntensity: 1.2,
});

class Model3DG1 {
  constructor() {
    this.root   = null;
    this.joints = new Map();
    this.restQ  = new Map();
    this.anim   = null;
  }

  async load(url) {
    const loader = new FBXLoader();
    this.root = await new Promise((ok, err) => loader.load(url, ok, null, err));
    this.root.rotation.set(-Math.PI / 2, 0, 0, 'YXZ');
    // FBX vertices are in metres (UnitScaleFactor=1.0, Z-up) — no extra scale needed
    this.root.scale.setScalar(1);
    // Collect joint objects (first occurrence wins — skeleton may have duplicates)
    this.root.traverse(c => {
      if (c.name && (c.name.includes('joint') || c.name === 'root'))
        if (!this.joints.has(c.name)) this.joints.set(c.name, c);
    });
    // Save rest quaternions before any pose changes
    for (const [n, j] of this.joints) this.restQ.set(n, j.quaternion.clone());

    // Convert SkinnedMesh → plain Mesh parented under its primary bone
    const sms = [];
    this.root.traverse(c => { if (c.isSkinnedMesh) sms.push(c); });
    for (const sm of sms) {
      const bone = sm.skeleton?.bones?.[0];
      if (!bone) continue;
      const mesh = new THREE.Mesh(sm.geometry, sm.material);
      mesh.name = sm.name;
      mesh.frustumCulled = false;
      mesh.castShadow    = true;
      mesh.geometry.computeVertexNormals();
      bone.add(mesh);
      sm.parent?.remove(sm);
      sm.skeleton?.dispose();
    }

    // Remove cosmetic pelvis helper group
    const pg = this.root.children.find(c => c.name === '_pelvis_grp');
    if (pg) this.root.remove(pg);

    // Apply PBR materials
    this.root.traverse(c => {
      if (!c.isMesh) return;
      const n      = (c.name || '').toLowerCase();
      const isDark = DARK_PARTS.some(p => n.includes(p));
      const old    = c.material;
      c.material   = (isDark ? MAT_DARK : MAT_WHITE).clone();
      if (old?.dispose) old.dispose();
    });

    return this;
  }

  setFrame(frame) {
    if (!this.anim) return;
    const fd = this.anim.getFrameData(frame);
    if (!fd) return;

    // Root: translate (cm → m) + ZYX Euler rotation
    const root = this.joints.get('floating_base_joint');
    if (root) {
      // CSV values are in cm; FBX is in metres — multiply by 0.01
      root.position.set(
        (fd.root_translateX || 0) * 0.01,
        (fd.root_translateY || 0) * 0.01,
        (fd.root_translateZ || 0) * 0.01,
      );
      root.rotation.set(
        THREE.MathUtils.degToRad(fd.root_rotateX || 0),
        THREE.MathUtils.degToRad(fd.root_rotateY || 0),
        THREE.MathUtils.degToRad(fd.root_rotateZ || 0),
        'ZYX',
      );
    }

    // Per-joint DOF: compose restQ × dofQuat
    const ax = new THREE.Vector3();
    const q  = new THREE.Quaternion();
    for (const [jn, joint] of this.joints) {
      if (jn === 'floating_base_joint') continue;
      const angle = fd[jn + '_dof'];
      if (angle === undefined || isNaN(angle)) continue;
      if      (jn.includes('pitch') || jn.includes('knee') || jn.includes('elbow')) ax.set(0, 1, 0);
      else if (jn.includes('roll'))  ax.set(1, 0, 0);
      else if (jn.includes('yaw'))   ax.set(0, 0, 1);
      else                           ax.set(0, 1, 0);
      q.setFromAxisAngle(ax, THREE.MathUtils.degToRad(angle));
      const rq = this.restQ.get(jn);
      if (rq) joint.quaternion.copy(rq).multiply(q);
      else    joint.quaternion.copy(q);
    }
  }
}

// ── Theme ──────────────────────────────────────────────────────────────────
const BG_LIGHT = 0xf5f5f5;
const BG_DARK  = 0x0e0e1a;
let isDark = false;

// ── Scene setup ────────────────────────────────────────────────────────────
const canvas   = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ antialias: true, canvas });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled     = true;
renderer.shadowMap.type        = THREE.PCFSoftShadowMap;
renderer.toneMapping           = THREE.CineonToneMapping;
renderer.toneMappingExposure   = 1.05;
renderer.outputColorSpace      = THREE.SRGBColorSpace;
renderer.setClearColor(BG_LIGHT, 1);

const scene = new THREE.Scene();
scene.fog   = new THREE.Fog(BG_LIGHT, 18, 40);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(2.2, 1.7, 2.2);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0.8, 0);
controls.maxDistance   = 20;
controls.minDistance   = 0.3;

// Lights
scene.add(new THREE.HemisphereLight(0xffffff, 0xbbbbcc, 2.5));

const key = new THREE.DirectionalLight(0xfff5e8, 2.2);
key.position.set(3, 6, 4);
key.castShadow = true;
key.shadow.mapSize.set(1024, 1024);
key.shadow.camera.near   = 0.1;
key.shadow.camera.far    = 20;
key.shadow.camera.left   = -8; key.shadow.camera.right = 8;
key.shadow.camera.top    =  8; key.shadow.camera.bottom = -8;
scene.add(key);

const fill = new THREE.DirectionalLight(0xd0e8ff, 1.0);
fill.position.set(-4, 3, -2);
scene.add(fill);

const rim = new THREE.DirectionalLight(0xc8d8ff, 0.7);
rim.position.set(1, 4, -5);
scene.add(rim);

let currentGrid = new THREE.GridHelper(20, 40, 0xaaaaaa, 0xcccccc);
scene.add(currentGrid);

// Resize handler
function resize() {
  const wrap = canvas.parentElement;
  const w = wrap ? wrap.clientWidth  : 800;
  const h = wrap ? wrap.clientHeight : 500;
  if (w === 0 || h === 0) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
resize();
new ResizeObserver(resize).observe(canvas.parentElement);

// ── Animation state ────────────────────────────────────────────────────────
let g1 = null, frame = 0, playing = false, lastTs = null;
const SPEEDS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0];
let speedIdx = 2;

const btnPlay  = document.getElementById('btn-play');
const slider   = document.getElementById('frame-slider');
const frameLbl = document.getElementById('frame-lbl');
const speedLbl = document.getElementById('speed-lbl');
const statusEl = document.getElementById('status');
const hintEl   = document.getElementById('hint');

btnPlay.addEventListener('click', () => {
  playing = !playing;
  btnPlay.textContent = playing ? '⏸' : '▶';
});
slider.addEventListener('input', () => {
  frame = parseFloat(slider.value);
  if (g1) g1.setFrame(frame);
});
document.getElementById('btn-slower').addEventListener('click', () => {
  speedIdx = Math.max(0, speedIdx - 1);
  speedLbl.textContent = '×' + SPEEDS[speedIdx];
});
document.getElementById('btn-faster').addEventListener('click', () => {
  speedIdx = Math.min(SPEEDS.length - 1, speedIdx + 1);
  speedLbl.textContent = '×' + SPEEDS[speedIdx];
});

// ── Theme toggle ───────────────────────────────────────────────────────────
function setTheme(dark) {
  isDark = dark;
  const bg = dark ? BG_DARK : BG_LIGHT;
  renderer.setClearColor(bg, 1);
  scene.fog.color.setHex(bg);
  document.body.style.background = '#' + new THREE.Color(bg).getHexString();
  scene.remove(currentGrid);
  currentGrid = dark
    ? new THREE.GridHelper(20, 40, 0x223344, 0x1a2533)
    : new THREE.GridHelper(20, 40, 0xaaaaaa, 0xcccccc);
  scene.add(currentGrid);
  statusEl.style.color = dark ? '#6060a8' : '#6666aa';
  hintEl.style.color   = dark ? '#3a3a6a' : '#8888bb';
  document.getElementById('btn-theme').textContent = dark ? '☀️' : '🌙';
}
// Apply initial light-theme colours
statusEl.style.color = '#6666aa';
hintEl.style.color   = '#8888bb';
document.getElementById('btn-theme').addEventListener('click', () => setTheme(!isDark));

// ── Poll for motion updates ────────────────────────────────────────────────
let _seenVersion = -1;
async function poll() {
  try {
    const r = await fetch(SERVER + '/anim-state', { cache: 'no-store' });
    if (!r.ok) return;
    const { version, name } = await r.json();
    if (version === _seenVersion || version === 0) return;
    _seenVersion = version;
    hintEl.style.display = 'none';
    statusEl.textContent  = 'Loading ' + name + '…';
    const csvR = await fetch(SERVER + '/csv', { cache: 'no-store' });
    if (!csvR.ok) return;
    const csvText = await csvR.text();
    if (g1) {
      g1.anim = new G1Animation(csvText, name);
      frame = 0; playing = true; btnPlay.textContent = '⏸';
      slider.max   = g1.anim.maxFrame;
      slider.value = 0;
      frameLbl.textContent = '0 / ' + g1.anim.maxFrame;
      statusEl.textContent = name + '  (' + g1.anim.frames.length + ' frames)';
    }
  } catch (_) { /* network noise — ignore */ }
}
setInterval(poll, 300);

// ── Render loop ────────────────────────────────────────────────────────────
function animate(ts) {
  requestAnimationFrame(animate);
  if (playing && g1?.anim) {
    if (lastTs !== null) {
      const dt = (ts - lastTs) / 1000;
      frame += dt * (g1.anim.fps || 120) * SPEEDS[speedIdx];
      if (frame > g1.anim.maxFrame) frame = 0;
    }
    slider.value  = frame;
    frameLbl.textContent = Math.floor(frame) + ' / ' + g1.anim.maxFrame;
  }
  lastTs = ts;
  if (g1) g1.setFrame(frame);
  controls.update();
  renderer.render(scene, camera);
}

// ── Load G1 FBX model ──────────────────────────────────────────────────────
(async () => {
  try {
    g1 = await new Model3DG1().load(SERVER + '/g1.fbx');
    scene.add(g1.root);
    statusEl.textContent = 'Ready — click a row to preview';
  } catch (e) {
    statusEl.textContent = 'Model unavailable (' + e.message + ')';
    console.warn('[viewer] FBX load failed:', e);
  }
  requestAnimationFrame(animate);
})();
