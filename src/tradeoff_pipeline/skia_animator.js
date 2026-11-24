#!/usr/bin/env node
/**
 * Tradeoff Layer (@TradeoffLayer) — True 60fps 2D Vector, Physics & Rigged Sprite Animation Engine.
 * Powered by Google Skia (`@napi-rs/canvas`) streaming raw RGBA frames directly into FFmpeg.
 *
 * Forensic 1:1 recreation of @TradeoffLayer's real animation language (ZERO PowerPoint slide cards!):
 *   1. `character_walk_handoff`  — Full-color cartoon character cutouts on 2-bone IK walking stick legs
 *                                  walking across sunny vector campuses, carrying & handing over chip props (`yB9SC4HbH5o`).
 *   2. `rain_buckets_analogy`    — Everyday physical metaphor simulation: narrow clogged pipe vs wide parallel
 *                                  streams filling glass beakers with sine-wave rising water & pixel-to-crisp reveal (`2d883UK7p9E`).
 *   3. `chip_3d_flip_stack`      — Workbench tweezers 3D-flipping silicon dies (`scaleY(cos)`), 12 DRAM layers
 *                                  spring-stacking vertically onto a 2.5D interposer, copper TSV elevators drilling
 *                                  through with glowing data pulses (`2d883UK7p9E`).
 *   4. `whiteboard_systolic_sim` — Presenter pointing articulated stick arm at a clean rounded board running a live
 *                                  GPU memory-starved pause (`||`) -> HBM diagonal systolic wavefront simulation (`yB9SC4HbH5o`).
 *   5. `dark_cross_section_meter`— Matte #0B0B0E token/bandwidth Bezier orb meter ticking continuously + progressive
 *                                  stroke-drawn cross-section cable/interposer packet journey (`0vZCvW7IzmA`).
 *   6. `dark_gpu_partition_split`— Matrix weight tiles filling & overflowing 1 GPU perimeter in coral red, then
 *                                  spring-splitting across 4 GPUs with pulsing yellow NCCL sync lasers (`0vZCvW7IzmA`).
 *   7. `map_supply_chain_arcs`   — Warm cream vector supply-chain map with waving flags & flying HBM wafers along
 *                                  dashed parabolic trajectories (`KTY7TnGIVNY`).
 *   8. `wafer_conveyor_zoom_out` — 3-to-1 silicon wafer compression & smooth 2.4x->0.7x virtual camera zoom-out
 *                                  revealing fab squeeze / custom lock-in scale (`cER6uZO6gvA` / `2d883UK7p9E`).
 */

const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");

// Resolve @napi-rs/canvas from either local node_modules or worker animator_env
let canvasPkg;
try {
  canvasPkg = require("@napi-rs/canvas");
} catch (_) {
  canvasPkg = require("/content/tradeoff_layer_carbon/animator_env/node_modules/@napi-rs/canvas");
}
const { createCanvas, loadImage, GlobalFonts } = canvasPkg;

// Register clean system sans & mono fonts
for (const fp of [
  "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
  "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
  "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
  "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
]) {
  if (fs.existsSync(fp)) {
    try { GlobalFonts.registerFromPath(fp); } catch (_) {}
  }
}

const SANS = "Liberation Sans, DejaVu Sans, Arial, sans-serif";
const MONO = "DejaVu Sans Mono, Liberation Mono, monospace";

// --- Math & Physics Helpers ---
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const lerp = (a, b, t) => a + (b - a) * clamp(t, 0, 1);
const easeOutCubic = (t) => { const x = clamp(t, 0, 1); return 1 - Math.pow(1 - x, 3); };
const easeInOutCubic = (t) => {
  const x = clamp(t, 0, 1);
  return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
};
const springOut = (t) => {
  const x = clamp(t, 0, 1);
  if (x === 0 || x === 1) return x;
  return 1 - Math.exp(-6.5 * x) * Math.cos(9.5 * x);
};

function roundRectPath(ctx, x, y, w, h, r) {
  const rad = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rad, y);
  ctx.arcTo(x + w, y, x + w, y + h, rad);
  ctx.arcTo(x + w, y + h, x, y + h, rad);
  ctx.arcTo(x, y + h, x, y, rad);
  ctx.arcTo(x, y, x + w, y, rad);
  ctx.closePath();
}

function drawCoverImage(ctx, img, w, h, zoom = 1.0, panX = 0.0, panY = 0.0) {
  if (!img) {
    const g = ctx.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, "#bae6fd");
    g.addColorStop(0.72, "#e0f2fe");
    g.addColorStop(0.721, "#e7c9a9");
    g.addColorStop(1, "#d4ad84");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
    return;
  }
  const scale = Math.max(w / img.width, h / img.height) * zoom;
  const dw = img.width * scale;
  const dh = img.height * scale;
  const dx = (w - dw) * (0.5 + panX * 0.35);
  const dy = (h - dh) * (0.5 + panY * 0.35);
  ctx.drawImage(img, dx, dy, dw, dh);
}

// --- 2-Bone IK Stick Limb + Full-Color Cartoon Torso Actor Rig (`yB9SC4HbH5o` style!) ---
function drawRiggedActor(ctx, opts) {
  const {
    x, footY, scale = 1.0, bustImg = null,
    walkSpeed = 0.0, t = 0.0, facing = 1,
    speakingAmp = 0.0, reachTarget = null,
    shirtColor = "#2563eb", hairColor = "#4b3621",
    badgeText = "",
  } = opts;

  const legLen = 132 * scale;
  const walkPhase = t * 7.5;
  const isWalking = Math.abs(walkSpeed) > 0.05;
  const stride = isWalking ? Math.sin(walkPhase) * 28 * scale : 0;
  const bounce = isWalking ? Math.abs(Math.cos(walkPhase)) * 7 * scale : Math.sin(t * 2.4) * 2.5 * scale;
  const hipY = footY - legLen + bounce;

  ctx.save();
  ctx.strokeStyle = "#141821";
  ctx.fillStyle = "#141821";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const lw = Math.max(3, 4.5 * scale);
  ctx.lineWidth = lw;

  // 2-bone IK walking legs + half-moon shoes
  for (const side of [-1, 1]) {
    const hipX = x + side * 20 * scale;
    const phase = walkPhase + (side === 1 ? 0 : Math.PI);
    const footX = hipX + (isWalking ? Math.sin(phase) * 28 * scale : 0);
    const lift = isWalking ? Math.max(0, -Math.cos(phase)) * 18 * scale : 0;
    const curFootY = footY - lift;
    const kneeX = (hipX + footX) * 0.5 + facing * (isWalking ? 11 * scale : 3 * scale);
    const kneeY = (hipY + curFootY) * 0.52;

    ctx.beginPath();
    ctx.moveTo(hipX, hipY);
    ctx.lineTo(kneeX, kneeY);
    ctx.lineTo(footX, curFootY);
    ctx.stroke();

    // Half-circle black shoe
    ctx.beginPath();
    ctx.arc(footX + facing * 5 * scale, curFootY, 13 * scale, Math.PI, 0, false);
    ctx.closePath();
    ctx.fill();
  }

  const torsoH = 128 * scale;
  const shoulderY = hipY - torsoH + 34 * scale;

  // Back arm
  const backShX = x - facing * 42 * scale;
  const backSwing = isWalking ? -stride * 0.7 : Math.sin(t * 2.1) * 5 * scale;
  ctx.beginPath();
  ctx.moveTo(backShX, shoulderY);
  ctx.lineTo(backShX - facing * 20 * scale + backSwing * 0.4, shoulderY + 42 * scale);
  ctx.lineTo(backShX - facing * 10 * scale + backSwing, shoulderY + 78 * scale);
  ctx.stroke();

  // Upper body + head (AI full-color cutout sprite OR vibrant procedural cel-shaded vector character)
  if (bustImg) {
    const targetH = 215 * scale * (1 + 0.016 * speakingAmp);
    const aspect = bustImg.width / Math.max(1, bustImg.height);
    const targetW = clamp(targetH * aspect, 90 * scale, 220 * scale);
    ctx.drawImage(bustImg, x - targetW / 2, hipY - targetH + 12 * scale, targetW, targetH);
  } else {
    // Vibrant full-color procedural cel-shaded character
    const tw = 48 * scale;
    const topY = hipY - torsoH;
    roundRectPath(ctx, x - tw, topY, tw * 2, torsoH, 18 * scale);
    ctx.fillStyle = shirtColor;
    ctx.fill();
    ctx.stroke();

    // White collar V
    ctx.beginPath();
    ctx.moveTo(x - 16 * scale, topY + 2 * scale);
    ctx.lineTo(x, topY + 28 * scale);
    ctx.lineTo(x + 16 * scale, topY + 2 * scale);
    ctx.fillStyle = "#f8fafc";
    ctx.fill();
    ctx.stroke();

    // Head
    const headY = topY - 48 * scale;
    ctx.beginPath();
    ctx.ellipse(x, headY, 40 * scale, 45 * scale, 0, 0, Math.PI * 2);
    ctx.fillStyle = "#f7c59f";
    ctx.fill();
    ctx.stroke();

    // Hair
    ctx.beginPath();
    ctx.arc(x, headY - 10 * scale, 42 * scale, Math.PI * 0.95, Math.PI * 2.05);
    ctx.fillStyle = hairColor;
    ctx.fill();
    ctx.stroke();

    // Glasses & eyes
    const eyeY = headY - 3 * scale;
    const blink = (t % 3.6) > 3.46;
    ctx.fillStyle = "#141821";
    for (const exOff of [-14, 14]) {
      const ex = x + exOff * scale + facing * 4 * scale;
      roundRectPath(ctx, ex - 10 * scale, eyeY - 7 * scale, 20 * scale, 14 * scale, 4 * scale);
      ctx.stroke();
      if (!blink) {
        ctx.beginPath();
        ctx.arc(ex, eyeY, 3.2 * scale, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    // Mouth synced to voice
    const my = headY + 19 * scale;
    if (speakingAmp > 0.16) {
      ctx.beginPath();
      ctx.ellipse(x + facing * 3 * scale, my, 9 * scale, (3 + 7 * speakingAmp) * scale, 0, 0, Math.PI * 2);
      ctx.fillStyle = "#be3b3b";
      ctx.fill();
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.arc(x + facing * 3 * scale, my - 3 * scale, 10 * scale, 0.15 * Math.PI, 0.85 * Math.PI);
      ctx.stroke();
    }
  }

  // Front gesturing / reaching stick arm (drawn over torso so prop handoffs & pointing read clearly!)
  const frontShX = x + facing * 42 * scale;
  let handX, handY;
  if (reachTarget) {
    handX = reachTarget.x;
    handY = reachTarget.y;
  } else {
    handX = frontShX + facing * 56 * scale + (isWalking ? stride * 0.8 : Math.cos(t * 3.2) * 8 * scale);
    handY = shoulderY + 8 * scale + Math.sin(t * 3.6) * 10 * scale;
  }
  const elbX = (frontShX + handX) * 0.5 - facing * 8 * scale;
  const elbY = Math.max(shoulderY, handY) + 22 * scale;
  ctx.beginPath();
  ctx.moveTo(frontShX, shoulderY);
  ctx.lineTo(elbX, elbY);
  ctx.lineTo(handX, handY);
  ctx.stroke();

  // 3-finger cartoon stick hand
  for (const ang of [-0.4, 0, 0.4]) {
    ctx.beginPath();
    ctx.moveTo(handX, handY);
    ctx.lineTo(handX + Math.cos(ang) * facing * 10 * scale, handY + Math.sin(ang) * 10 * scale);
    ctx.stroke();
  }

  // Minimalist floating name/role tag only if specified
  if (badgeText) {
    ctx.font = `bold ${Math.round(20 * scale)}px ${SANS}`;
    const tw = ctx.measureText(badgeText).width + 24;
    const bx = x - tw / 2;
    const by = hipY - 238 * scale;
    roundRectPath(ctx, bx, by, tw, 32 * scale, 10);
    ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
    ctx.fill();
    ctx.fillStyle = "#f8fafc";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(badgeText, x, by + 16 * scale);
  }
  ctx.restore();
  return { handX, handY };
}

// --- Draw glowing isometric 3D HBM Chip Stack Prop (crisp vector so layers & TSVs animate!) ---
function drawHbmChipModule(ctx, cx, cy, scale = 1.0, layers = 8, glowPhase = 0.0, label = "HBM") {
  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(scale, scale);

  // Green/gold substrate base
  roundRectPath(ctx, -64, 18, 128, 22, 6);
  ctx.fillStyle = "#0f766e";
  ctx.fill();
  ctx.strokeStyle = "#13202c";
  ctx.lineWidth = 3;
  ctx.stroke();

  // Gold solder micro-bumps
  ctx.fillStyle = "#fbbf24";
  for (let i = -50; i <= 50; i += 14) {
    ctx.beginPath();
    ctx.arc(i, 42, 3.5, 0, Math.PI * 2);
    ctx.fill();
  }

  // Stacked sapphire silicon dies
  for (let l = 0; l < layers; l++) {
    const ly = 10 - l * 10;
    const pulse = 0.5 + 0.5 * Math.sin(glowPhase * 5 - l * 0.6);
    roundRectPath(ctx, -48, ly, 96, 9, 3);
    ctx.fillStyle = l % 2 === 0 ? "#2563eb" : "#3b82f6";
    ctx.fill();
    ctx.strokeStyle = "#0f172a";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Glowing TSV vertical dots
    ctx.fillStyle = pulse > 0.6 ? "#fde047" : "#38bdf8";
    for (const vx of [-28, -10, 10, 28]) {
      ctx.fillRect(vx - 1.5, ly + 2, 3, 5);
    }
  }

  // Top cap label
  const topY = 10 - layers * 10 - 20;
  roundRectPath(ctx, -50, topY, 100, 22, 5);
  ctx.fillStyle = "#1e293b";
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#38bdf8";
  ctx.font = `bold 13px ${MONO}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, 0, topY + 11);
  ctx.restore();
}

// ============================================================================
// 8 PARAMETERIZED @TRADEOFFLAYER 2D ANIMATION CHOREOGRAPHIES (ZERO SLIDE BOXES)
// ============================================================================

// 1. `character_walk_handoff` — Rigged characters walking in, camera zoom, handing over HBM/GPU module (`tpu_intro`)
function renderCharacterWalkHandoff(ctx, assets, scene, w, h, t, dur, amp, isShort) {
  const u = t / Math.max(0.5, dur);
  const camZoom = lerp(1.02, 1.14, easeInOutCubic(u));
  const bg = assets.sceneBg || assets.stage_courtyard;

  ctx.save();
  ctx.translate(w / 2, h / 2);
  ctx.scale(camZoom, camZoom);
  ctx.translate(-w / 2, -h / 2);
  drawCoverImage(ctx, bg, w, h, 1.0, (u - 0.5) * 0.15, 0);

  // Subtle floor ground line
  const floorY = isShort ? h * 0.69 : h * 0.86;
  ctx.fillStyle = "rgba(15, 23, 42, 0.08)";
  ctx.fillRect(0, floorY, w, h - floorY);

  const walkT = clamp(t / Math.min(3.2, dur * 0.42), 0, 1);
  const handoffT = easeInOutCubic((t - dur * 0.42) / Math.max(0.8, dur * 0.25));

  const leftStart = isShort ? -120 : 120;
  const leftEnd = isShort ? w * 0.31 : w * 0.36;
  const rightStart = isShort ? w + 120 : w - 120;
  const rightEnd = isShort ? w * 0.70 : w * 0.64;

  const ax = lerp(leftStart, leftEnd, easeOutCubic(walkT));
  const bx = lerp(rightStart, rightEnd, easeOutCubic(walkT));
  const sc = isShort ? 0.88 : 1.08;

  const metrics = scene.metrics || [];
  const tagA = (scene.diagram_nodes && scene.diagram_nodes[0]) || (metrics[0] && metrics[0].value) || "SK hynix";
  const tagB = (scene.diagram_nodes && scene.diagram_nodes[2]) || (metrics[1] && metrics[1].value) || "AI GPU";
  const propLabel = (metrics[0] && metrics[0].value) || "HBM3E";

  const propX = lerp(ax + 78 * sc, bx - 78 * sc, handoffT);
  const propY = floorY - 195 * sc + Math.sin(t * 3.5) * 4;

  drawRiggedActor(ctx, {
    x: ax,
    footY: floorY,
    scale: sc,
    bustImg: assets.char_engineer,
    walkSpeed: walkT < 0.98 ? 1.0 : 0.0,
    t,
    facing: 1,
    speakingAmp: u < 0.55 ? amp : 0.05,
    reachTarget: walkT > 0.85 ? { x: propX - 24 * sc, y: propY + 10 } : null,
    shirtColor: "#2563eb",
    badgeText: String(tagA).slice(0, 18),
  });

  drawRiggedActor(ctx, {
    x: bx,
    footY: floorY,
    scale: sc,
    bustImg: assets.char_ceo,
    walkSpeed: walkT < 0.98 ? 1.0 : 0.0,
    t: t + 0.4,
    facing: -1,
    speakingAmp: u >= 0.55 ? amp : 0.05,
    reachTarget: handoffT > 0.15 ? { x: propX + 24 * sc, y: propY + 10 } : null,
    shirtColor: "#059669",
    hairColor: "#cbd5e1",
    badgeText: String(tagB).slice(0, 18),
  });

  // Draw carried/handed-off prop (AI cutout prop OR animated glowing HBM tower)
  const popS = springOut(t / 0.6);
  if (assets.prop_chip_stack && u < 0.5) {
    const pw = 135 * sc * popS;
    ctx.drawImage(assets.prop_chip_stack, propX - pw / 2, propY - pw / 2, pw, pw);
  } else {
    drawHbmChipModule(ctx, propX, propY + 16, sc * 0.92 * popS, 8, t, String(propLabel).slice(0, 10));
  }

  // Crisp floating minimal stat bubble on handoff completion (like "$1 / 2025" coin in `2d883UK7p9E`)
  if (handoffT > 0.4 && metrics.length > 0) {
    const bs = springOut((handoffT - 0.4) / 0.5);
    const statTxt = `${metrics[0].label}: ${metrics[0].value}`;
    ctx.save();
    ctx.translate((ax + bx) / 2, floorY - 355 * sc);
    ctx.scale(bs, bs);
    ctx.font = `bold 24px ${MONO}`;
    const bw = ctx.measureText(statTxt).width + 36;
    roundRectPath(ctx, -bw / 2, -24, bw, 48, 24);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#0f172a";
    ctx.stroke();
    ctx.fillStyle = "#0f172a";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(statTxt, 0, 0);
    ctx.restore();
  }
  ctx.restore();
}

// 2. `rain_buckets_analogy` — Physical droplets & rising sine-wave water in glass beakers (`sony_buckets`!)
function renderRainBucketsAnalogy(ctx, assets, scene, w, h, t, dur, amp, isShort) {
  const u = t / Math.max(0.5, dur);
  const sky = ctx.createLinearGradient(0, 0, 0, h);
  sky.addColorStop(0, "#bae6fd");
  sky.addColorStop(0.74, "#e0f2fe");
  sky.addColorStop(0.741, "#e6c39c");
  sky.addColorStop(1, "#cfa06e");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, w, h);

  // Soft cartoon clouds & greenery silhouettes (`sony_buckets.jpg` exact backdrop!)
  ctx.fillStyle = "rgba(255,255,255,0.82)";
  for (const [cx, cy, r] of [[w * 0.22, h * 0.16, 48], [w * 0.72, h * 0.14, 56]]) {
    const drift = Math.sin(t * 0.3 + cx) * 18;
    ctx.beginPath();
    ctx.arc(cx + drift, cy, r, 0, Math.PI * 2);
    ctx.arc(cx + drift + r * 0.85, cy + 8, r * 0.78, 0, Math.PI * 2);
    ctx.arc(cx + drift - r * 0.8, cy + 12, r * 0.7, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = "#4ade80";
  for (let bx = 0; bx <= w + 120; bx += 130) {
    ctx.beginPath();
    ctx.arc(bx, h * 0.74, 88 + (bx % 3) * 18, Math.PI, 0);
    ctx.fill();
  }

  // Phase 1 (first 45%): Narrow pipe -> slow single drips -> barely filling beakers ("Narrow Doorway")
  // Phase 2 (last 55%): Wide parallel spouts -> heavy droplet rain -> rapidly rising cyan water ("HBM Parallel Lanes")
  const wideMode = u >= 0.46;
  const pipeY = isShort ? h * 0.22 : h * 0.15;
  const deskY = isShort ? h * 0.67 : h * 0.76;

  const beakerCount = 3;
  const bW = isShort ? 175 : 210;
  const bH = isShort ? 250 : 285;
  const gap = isShort ? 58 : 110;
  const startX = (w - (beakerCount * bW + (beakerCount - 1) * gap)) / 2;

  // Overhead delivery pipe
  ctx.fillStyle = "#334155";
  ctx.strokeStyle = "#0f172a";
  ctx.lineWidth = 3.5;
  const pipeH = wideMode ? 30 : 12;
  roundRectPath(ctx, startX - 40, pipeY, beakerCount * bW + (beakerCount - 1) * gap + 80, pipeH, 8);
  ctx.fill();
  ctx.stroke();

  for (let i = 0; i < beakerCount; i++) {
    const bx = startX + i * (bW + gap);
    const by = deskY - bH;
    const spoutX = bx + bW / 2;

    // Nozzle
    ctx.fillStyle = wideMode ? "#0284c7" : "#ef4444";
    const nw = wideMode ? 44 : 10;
    ctx.fillRect(spoutX - nw / 2, pipeY + pipeH, nw, 18);
    ctx.strokeRect(spoutX - nw / 2, pipeY + pipeH, nw, 18);

    // Animated falling droplets
    const streams = wideMode ? 4 : 1;
    const dropSpeed = wideMode ? 3.4 : 1.1;
    ctx.fillStyle = wideMode ? "#0ea5e9" : "#60a5fa";
    for (let s = 0; s < streams; s++) {
      const sx = spoutX + (s - (streams - 1) / 2) * 11;
      for (let d = 0; d < (wideMode ? 4 : 1); d++) {
        const dp = ((t * dropSpeed + i * 0.31 + s * 0.19 + d * 0.25) % 1.0);
        const dy = lerp(pipeY + pipeH + 22, deskY - 24, dp);
        ctx.beginPath();
        ctx.ellipse(sx, dy, 4.5, 8.5, 0, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Fill level inside beaker
    const fillRatio = wideMode
      ? lerp(0.14, 0.84 - i * 0.06, easeOutCubic((u - 0.46) / 0.48))
      : lerp(0.03, 0.14, u / 0.46);

    // Clip to glass beaker interior & draw animated sine-wave water!
    ctx.save();
    roundRectPath(ctx, bx + 6, by + 8, bW - 12, bH - 14, 16);
    ctx.clip();
    const waterTop = deskY - 8 - (bH - 24) * fillRatio;
    ctx.beginPath();
    ctx.moveTo(bx, deskY);
    for (let wx = bx; wx <= bx + bW; wx += 8) {
      const wy = waterTop + Math.sin(wx * 0.045 + t * 6.5 + i) * (wideMode ? 5.5 : 2.0);
      ctx.lineTo(wx, wy);
    }
    ctx.lineTo(bx + bW, deskY);
    ctx.closePath();
    const wg = ctx.createLinearGradient(0, waterTop, 0, deskY);
    wg.addColorStop(0, wideMode ? "rgba(56, 189, 248, 0.88)" : "rgba(251, 146, 60, 0.78)");
    wg.addColorStop(1, wideMode ? "rgba(2, 132, 199, 0.92)" : "rgba(234, 88, 12, 0.88)");
    ctx.fillStyle = wg;
    ctx.fill();
    ctx.restore();

    // Glass beaker outline + measurement ticks (`sony_buckets.jpg`!)
    roundRectPath(ctx, bx, by, bW, bH, 18);
    ctx.strokeStyle = "#1e3a5f";
    ctx.lineWidth = 4;
    ctx.stroke();
    for (let tk = 1; tk <= 4; tk++) {
      const ty = deskY - (bH * tk) / 5;
      ctx.beginPath();
      ctx.moveTo(bx + bW - 24, ty);
      ctx.lineTo(bx + bW - 6, ty);
      ctx.stroke();
    }

    // Percentage tag on beaker
    ctx.fillStyle = "#0f172a";
    ctx.font = `bold 22px ${MONO}`;
    ctx.textAlign = "center";
    ctx.fillText(`${Math.round(fillRatio * 100)}%`, spoutX, deskY + 36);
  }

  // Minimalist floating status pill switching from Narrow Bus -> HBM Parallel Stack
  const modeLabel = wideMode ? "HBM 1,024-Bit Parallel Flow" : "Standard Flat Bus: Cores Starved";
  ctx.font = `bold ${isShort ? 28 : 30}px ${SANS}`;
  const mw = ctx.measureText(modeLabel).width + 44;
  roundRectPath(ctx, (w - mw) / 2, pipeY - 68, mw, 50, 25);
  ctx.fillStyle = wideMode ? "#0284c7" : "#dc2626";
  ctx.fill();
  ctx.fillStyle = "#ffffff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(modeLabel, w / 2, pipeY - 43);
}

// 3. `chip_3d_flip_stack` — Workbench tweezers, 3D cosine flip, 12-layer stack & TSV elevators (`sony_flip`!)
function renderChip3dFlipStack(ctx, assets, scene, w, h, t, dur, amp, isShort) {
  const u = t / Math.max(0.5, dur);

  // Warm wooden workbench surface (`sony_flip.jpg`!)
  const wood = ctx.createLinearGradient(0, 0, w, h);
  wood.addColorStop(0, "#f5d7b0");
  wood.addColorStop(1, "#e3b982");
  ctx.fillStyle = wood;
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(161, 98, 7, 0.12)";
  ctx.lineWidth = 2;
  for (let wy = 40; wy < h; wy += 54) {
    ctx.beginPath();
    ctx.moveTo(0, wy);
    ctx.bezierCurveTo(w * 0.3, wy + 8, w * 0.7, wy - 8, w, wy + 4);
    ctx.stroke();
  }

  const cx = w / 2;
  const cy = isShort ? h * 0.46 : h * 0.56;
  const sc = isShort ? 0.92 : 1.15;

  // Cream ceramic interposer carrier tray
  roundRectPath(ctx, cx - 355 * sc, cy - 190 * sc, 710 * sc, 360 * sc, 28 * sc);
  ctx.fillStyle = "#f8f5eb";
  ctx.fill();
  ctx.lineWidth = 4;
  ctx.strokeStyle = "#1e293b";
  ctx.stroke();

  // Silver 2.5D Silicon Interposer bridge inside tray
  const bridgeP = easeOutCubic(u / 0.22);
  ctx.save();
  ctx.globalAlpha = bridgeP;
  roundRectPath(ctx, cx - 295 * sc, cy + 45 * sc, 590 * sc, 52 * sc, 10 * sc);
  ctx.fillStyle = "#cbd5e1";
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#334155";
  ctx.font = `bold ${Math.round(18 * sc)}px ${MONO}`;
  ctx.textAlign = "center";
  ctx.fillText("2.5D SILICON INTERPOSER", cx, cy + 77 * sc);
  ctx.restore();

  // Right side: Green GPU Die sitting on interposer
  const gpuX = cx + 145 * sc;
  const gpuY = cy - 18 * sc;
  roundRectPath(ctx, gpuX - 115 * sc, gpuY, 230 * sc, 62 * sc, 10 * sc);
  ctx.fillStyle = "#10b981";
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#064e3b";
  ctx.font = `bold ${Math.round(21 * sc)}px ${MONO}`;
  ctx.fillText("AI GPU DIE", gpuX, gpuY + 37 * sc);

  // Left side: 12 DRAM dies flying in & stacking vertically + 3D cosine flip on top die!
  const stackX = cx - 135 * sc;
  const baseStackY = cy + 32 * sc;
  const totalLayers = 12;
  const stackPhase = clamp((u - 0.08) / 0.52, 0, 1);

  for (let l = 0; l < totalLayers; l++) {
    const lp = clamp((stackPhase * totalLayers - l), 0, 1);
    if (lp <= 0.01) continue;
    const dropY = lerp(baseStackY - 260 * sc, baseStackY - l * 15 * sc, springOut(lp));
    const flipScaleY = l === totalLayers - 1 && lp < 0.85 ? Math.abs(Math.cos((1 - lp) * Math.PI)) : 1.0;

    ctx.save();
    ctx.translate(stackX, dropY);
    ctx.scale(1, Math.max(0.18, flipScaleY));
    roundRectPath(ctx, -95 * sc, -7 * sc, 190 * sc, 13 * sc, 4 * sc);
    ctx.fillStyle = l % 2 === 0 ? "#1d4ed8" : "#3b82f6";
    ctx.fill();
    ctx.lineWidth = 2.2;
    ctx.strokeStyle = "#0f172a";
    ctx.stroke();
    ctx.restore();
  }

  // Precision Tweezers placing the top die (`sony_flip.jpg`!)
  if (u < 0.62) {
    const twOut = u > 0.50 ? easeInOutCubic((u - 0.50) / 0.12) : 0;
    const tx = stackX + 90 * sc + twOut * 220 * sc;
    const ty = baseStackY - Math.min(totalLayers, stackPhase * totalLayers) * 15 * sc - twOut * 140 * sc;
    ctx.save();
    ctx.strokeStyle = "#64748b";
    ctx.lineWidth = 7 * sc;
    ctx.beginPath();
    ctx.moveTo(tx + 170 * sc, ty - 130 * sc);
    ctx.lineTo(tx, ty - 4 * sc);
    ctx.moveTo(tx + 182 * sc, ty - 115 * sc);
    ctx.lineTo(tx + 4 * sc, ty + 8 * sc);
    ctx.stroke();
    ctx.restore();
  }

  // Phase 3 (u >= 0.55): Drill golden vertical TSV elevators straight through all 12 layers + fire glowing data pulses!
  if (u >= 0.52) {
    const drillP = easeOutCubic((u - 0.52) / 0.22);
    const topOfTower = baseStackY - (totalLayers - 1) * 15 * sc;
    const bottomOfTower = cy + 68 * sc;
    const curBottom = lerp(topOfTower, bottomOfTower, drillP);

    for (const vxOff of [-62, -31, 0, 31, 62]) {
      const vx = stackX + vxOff * sc;
      ctx.strokeStyle = "#fbbf24";
      ctx.lineWidth = 4 * sc;
      ctx.beginPath();
      ctx.moveTo(vx, topOfTower);
      ctx.lineTo(vx, curBottom);
      ctx.stroke();

      // Glowing packets shooting down TSVs & across 2.5D interposer into GPU!
      if (drillP > 0.8) {
        const pktP = ((t * 2.4 + (vxOff + 70) * 0.013) % 1.0);
        let px, py;
        if (pktP < 0.55) {
          px = vx;
          py = lerp(topOfTower, bottomOfTower, pktP / 0.55);
        } else {
          px = lerp(vx, gpuX, (pktP - 0.55) / 0.45);
          py = bottomOfTower;
        }
        ctx.fillStyle = "#38bdf8";
        ctx.beginPath();
        ctx.arc(px, py, 5.5 * sc, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Crisp in-world callout tag
    const tagAlpha = easeOutCubic((u - 0.58) / 0.2);
    ctx.save();
    ctx.globalAlpha = tagAlpha;
    ctx.fillStyle = "#0f172a";
    ctx.font = `bold ${Math.round(24 * sc)}px ${MONO}`;
    ctx.textAlign = "center";
    ctx.fillText("12-Hi DRAM + TSV Vertical Elevators", cx, cy - 218 * sc);
    ctx.restore();
  }
}

// 4. `whiteboard_systolic_sim` — Presenter + GPU Starved (`||`) -> HBM Diagonal Wavefront (`tpu_grid`!)
function renderWhiteboardSystolicSim(ctx, assets, scene, w, h, t, dur, amp, isShort) {
  const u = t / Math.max(0.5, dur);
  drawCoverImage(ctx, assets.sceneBg || assets.stage_studio, w, h, lerp(1.0, 1.05, u), 0, 0);

  const wbX = isShort ? 64 : 108;
  const wbY = isShort ? 185 : 88;
  const wbW = isShort ? w - 128 : 1290;
  const wbH = isShort ? 810 : 810;

  // Stand legs
  ctx.strokeStyle = "#64748b";
  ctx.lineWidth = 8;
  ctx.beginPath();
  ctx.moveTo(wbX + wbW * 0.26, wbY + wbH);
  ctx.lineTo(wbX + wbW * 0.23, wbY + wbH + 80);
  ctx.moveTo(wbX + wbW * 0.74, wbY + wbH);
  ctx.lineTo(wbX + wbW * 0.77, wbY + wbH + 80);
  ctx.stroke();

  // Clean white rounded board
  roundRectPath(ctx, wbX, wbY, wbW, wbH, 24);
  ctx.fillStyle = "#f8fafc";
  ctx.fill();
  ctx.lineWidth = 4;
  ctx.strokeStyle = "#94a3b8";
  ctx.stroke();

  // 4-color top accent strip (`tpu_grid.jpg`!)
  const stripCols = ["#3b82f6", "#ef4444", "#f59e0b", "#10b981"];
  const segW = (wbW - 80) / 4;
  stripCols.forEach((col, i) => {
    ctx.fillStyle = col;
    ctx.fillRect(wbX + 40 + i * segW, wbY + 16, segW - 6, 9);
  });

  const fastMode = u >= 0.48;
  // Top Memory Queue Bar (`tpu_grid.jpg` top slot bar!)
  const qX = wbX + 68;
  const qY = wbY + 58;
  const qW = wbW - 136;
  const qH = 64;
  roundRectPath(ctx, qX, qY, qW, qH, 12);
  ctx.fillStyle = "#e2e8f0";
  ctx.fill();
  ctx.strokeStyle = "#94a3b8";
  ctx.lineWidth = 2;
  ctx.stroke();

  const slots = 24;
  const slotW = (qW - 28) / slots;
  for (let s = 0; s < slots; s++) {
    const active = fastMode ? true : s < Math.floor(lerp(6, 22, (u / 0.48)));
    ctx.fillStyle = !active ? "#cbd5e1" : fastMode ? "#10b981" : "#f59e0b";
    roundRectPath(ctx, qX + 14 + s * slotW + 2, qY + 12, slotW - 5, qH - 24, 4);
    ctx.fill();
  }

  // Compute Core Grid (`8x5` inside dark-green / clean silicon die frame)
  const cols = isShort ? 6 : 8;
  const rows = 5;
  const dieX = wbX + 76;
  const dieY = wbY + 195;
  const dieW = wbW - 152;
  const dieH = wbH - 305;

  // Animated wire traces from queue bar into die
  const laneCount = fastMode ? cols : 2;
  for (let l = 0; l < laneCount; l++) {
    const lx = dieX + ((l + 0.5) / laneCount) * dieW;
    ctx.strokeStyle = fastMode ? "#3b82f6" : "#d97706";
    ctx.lineWidth = fastMode ? 3.5 : 2;
    ctx.beginPath();
    ctx.moveTo(lx, qY + qH);
    ctx.lineTo(lx, dieY);
    ctx.stroke();

    const dotP = ((t * (fastMode ? 4.2 : 0.9) + l * 0.21) % 1.0);
    ctx.fillStyle = fastMode ? "#10b981" : "#f59e0b";
    ctx.beginPath();
    ctx.arc(lx, lerp(qY + qH + 4, dieY - 4, dotP), fastMode ? 5.5 : 4, 0, Math.PI * 2);
    ctx.fill();
  }

  roundRectPath(ctx, dieX, dieY, dieW, dieH, 18);
  ctx.fillStyle = fastMode ? "#eff6ff" : "#1e3a2f";
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = fastMode ? "#60a5fa" : "#0f172a";
  ctx.stroke();

  const pad = 22;
  const cellW = (dieW - pad * 2 - (cols - 1) * 14) / cols;
  const cellH = (dieH - pad * 2 - (rows - 1) * 14) / rows;
  const wavePos = (t * 5.2) % (cols + rows + 2);
  const wavePalette = ["#3b82f6", "#f59e0b", "#ef4444", "#10b981"];

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cx = dieX + pad + c * (cellW + 14);
      const cy = dieY + pad + r * (cellH + 14);
      let cellCol = "#f1f5f9";
      if (fastMode) {
        const dist = Math.abs((r + c) - wavePos);
        if (dist < 1.6) {
          cellCol = wavePalette[(r + c) % wavePalette.length];
        }
      } else {
        cellCol = (r === 1 && c === 2) || (r === 3 && c === cols - 2) ? "#4ade80" : "#e2e8f0";
      }
      roundRectPath(ctx, cx, cy, cellW, cellH, 9);
      ctx.fillStyle = cellCol;
      ctx.fill();
      ctx.strokeStyle = "rgba(15,23,42,0.18)";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Tiny register lines inside each core tile (`tpu_grid.jpg`!)
      ctx.fillStyle = cellCol === "#e2e8f0" || cellCol === "#f1f5f9" ? "#94a3b8" : "rgba(255,255,255,0.85)";
      ctx.fillRect(cx + 10, cy + cellH * 0.34, cellW * 0.55, 4);
      ctx.fillRect(cx + 10, cy + cellH * 0.58, cellW * 0.38, 4);
    }
  }

  // Pulsing Pause `||` icon when memory-starved (`tpu_grid.jpg` exact detail!)
  if (!fastMode) {
    const pulse = 1 + 0.08 * Math.sin(t * 7);
    ctx.save();
    ctx.translate(dieX + dieW / 2, dieY + dieH / 2);
    ctx.scale(pulse, pulse);
    ctx.beginPath();
    ctx.arc(0, 0, 38, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255,255,255,0.95)";
    ctx.fill();
    ctx.fillStyle = "#dc2626";
    ctx.fillRect(-12, -16, 8, 32);
    ctx.fillRect(4, -16, 8, 32);
    ctx.restore();
  }

  // Bottom in-board live bandwidth counter
  const bwVal = fastMode ? Math.round(lerp(64, 425, easeOutCubic((u - 0.48) / 0.4))) : 38;
  ctx.fillStyle = fastMode ? "#0f172a" : "#b91c1c";
  ctx.font = `bold 28px ${MONO}`;
  ctx.textAlign = "left";
  ctx.fillText(
    fastMode ? `HBM Systolic Feed: ${bwVal} GB/s (All Cores Active)` : `Narrow Bus Queue Jammed: ${bwVal} GB/s (Cores Waiting)`,
    dieX,
    wbY + wbH - 36
  );

  // Rigged Presenter pointing at the active area of the board!
  const pointTarget = {
    x: wbX + wbW - 35,
    y: fastMode ? dieY + dieH * 0.5 : qY + qH * 0.5,
  };
  drawRiggedActor(ctx, {
    x: isShort ? w * 0.80 : 1650,
    footY: isShort ? 1365 : 965,
    scale: isShort ? 0.92 : 1.16,
    bustImg: assets.char_analyst || assets.char_engineer,
    walkSpeed: 0,
    t,
    facing: -1,
    speakingAmp: amp,
    reachTarget: isShort ? null : pointTarget,
    shirtColor: "#0284c7",
  });
}

// 5. `dark_cross_section_meter` — Matte #0B0B0E Bezier orb meter + progressive cross-section cable/fab packet run (`aicost_packet`!)
function renderDarkCrossSectionMeter(ctx, assets, scene, w, h, t, dur, amp, isShort) {
  const u = t / Math.max(0.5, dur);
  ctx.fillStyle = "#0b0b0e";
  ctx.fillRect(0, 0, w, h);

  // Subtle starfield micro-dots (`aicost_packet.jpg`!)
  ctx.fillStyle = "rgba(148, 163, 184, 0.22)";
  for (let i = 0; i < 48; i++) {
    const sx = ((i * 197) % w);
    const sy = ((i * 311) % h);
    ctx.fillRect(sx, sy, 2, 2);
  }

  const nodes = (scene.diagram_nodes && scene.diagram_nodes.length ? scene.diagram_nodes : ["Prompt", "HBM Stack", "TSV", "2.5D", "Cores"]).slice(0, 5);
  const metrics = scene.metrics || [];
  const mainMetric = metrics[0] || { label: "Bandwidth", value: "425 GB/s", tone: "green" };
  const numMatch = String(mainMetric.value).match(/(\d+(?:\.\d+)?)/);
  const targetNum = numMatch ? parseFloat(numMatch[1]) : 425;
  const suffix = numMatch ? String(mainMetric.value).replace(numMatch[1], "") : "";
  const curNum = (targetNum * easeOutCubic(clamp(t / (dur * 0.72), 0, 1)));
  const formattedVal = (targetNum % 1 !== 0 ? curNum.toFixed(1) : Math.round(curNum)) + suffix;

  // Top token/node pills + Bezier curves + sliding orbs into "The meter"
  const pillY = isShort ? 235 : 95;
  const gap = isShort ? 12 : 22;
  const pillW = Math.min(isShort ? 175 : 220, (w - 140 - gap * (nodes.length - 1)) / nodes.length);
  const rowW = nodes.length * pillW + (nodes.length - 1) * gap;
  const startX = (w - rowW) / 2;

  const meterW = isShort ? 420 : 440;
  const meterH = 155;
  const meterX = (w - meterW) / 2;
  const meterY = pillY + (isShort ? 185 : 165);

  nodes.forEach((nd, i) => {
    const px = startX + i * (pillW + gap);
    const pEnter = springOut((t - i * 0.1) / 0.45);
    roundRectPath(ctx, px, pillY, pillW, 56, 10);
    ctx.fillStyle = "#1c1d24";
    ctx.fill();
    ctx.strokeStyle = "#333644";
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = "#38bdf8";
    ctx.font = `bold ${isShort ? 20 : 22}px ${MONO}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(nd).slice(0, 14), px + pillW / 2, pillY + 28);

    // Bezier curve down into meter
    const x0 = px + pillW / 2, y0 = pillY + 56;
    const x3 = meterX + meterW * (0.2 + 0.6 * (i / Math.max(1, nodes.length - 1))), y3 = meterY;
    ctx.strokeStyle = "rgba(148, 163, 184, 0.45)";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.bezierCurveTo(x0, y0 + 55, x3, y3 - 55, x3, y3);
    ctx.stroke();

    // Animated glowing orb sliding down Bezier wire!
    if (pEnter > 0.3) {
      const bu = ((t * 1.35 + i * 0.22) % 1.0);
      const om = 1 - bu;
      const ox = om * om * om * x0 + 3 * om * om * bu * x0 + 3 * om * bu * bu * x3 + bu * bu * bu * x3;
      const oy = om * om * om * y0 + 3 * om * om * bu * (y0 + 55) + 3 * om * bu * bu * (y3 - 55) + bu * bu * bu * y3;
      ctx.fillStyle = "#38bdf8";
      ctx.beginPath();
      ctx.arc(ox, oy, 5, 0, Math.PI * 2);
      ctx.fill();
    }
  });

  // "The meter" central rounded box (`aicost_packet.jpg` exact look!)
  roundRectPath(ctx, meterX, meterY, meterW, meterH, 20);
  ctx.fillStyle = "#181920";
  ctx.fill();
  ctx.strokeStyle = "#64748b";
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.fillStyle = "#e2e8f0";
  ctx.font = `bold 26px ${SANS}`;
  ctx.textAlign = "center";
  ctx.fillText(mainMetric.label || "The meter", w / 2, meterY + 42);

  ctx.fillStyle = mainMetric.tone === "green" ? "#4ade80" : mainMetric.tone === "blue" ? "#60a5fa" : "#f87171";
  ctx.font = `bold 52px ${MONO}`;
  ctx.fillText(formattedVal || String(mainMetric.value), w / 2, meterY + 106);

  // Bottom half: Progressive white-line architectural cross-section + subsea/interposer trench (`aicost_packet.jpg`!)
  const groundY = isShort ? h * 0.66 : h * 0.77;
  const drawProg = easeOutCubic((u - 0.12) / 0.55);
  const maxX = lerp(80, w - 80, drawProg);

  ctx.save();
  ctx.beginPath();
  ctx.rect(0, meterY + meterH + 20, maxX, h);
  ctx.clip();

  // Ocean / Silicon Trench dip in the middle
  const trenchL = w * 0.44, trenchR = w * 0.66, trenchDepth = 92;
  ctx.fillStyle = "rgba(30, 64, 175, 0.28)";
  ctx.beginPath();
  ctx.moveTo(trenchL, groundY);
  ctx.quadraticCurveTo((trenchL + trenchR) / 2, groundY + trenchDepth, trenchR, groundY);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = "#cbd5e1";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(70, groundY);
  ctx.lineTo(trenchL, groundY);
  ctx.quadraticCurveTo((trenchL + trenchR) / 2, groundY + trenchDepth, trenchR, groundY);
  ctx.lineTo(w - 70, groundY);
  ctx.stroke();

  // Left minimalist vector buildings (Fab & Packaging)
  const drawBuilding = (bx, bw, bh, label) => {
    ctx.strokeStyle = "#cbd5e1";
    ctx.fillStyle = "#12141c";
    ctx.fillRect(bx, groundY - bh, bw, bh);
    ctx.strokeRect(bx, groundY - bh, bw, bh);
    ctx.fillStyle = "#475569";
    for (let wy = groundY - bh + 14; wy < groundY - 14; wy += 18) {
      for (let wx = bx + 10; wx < bx + bw - 12; wx += 16) {
        ctx.fillRect(wx, wy, 8, 9);
      }
    }
    ctx.fillStyle = "#94a3b8";
    ctx.font = `bold 16px ${MONO}`;
    ctx.textAlign = "center";
    ctx.fillText(label, bx + bw / 2, groundY + 26);
  };

  drawBuilding(w * 0.11, 76, 95, "SK hynix");
  drawBuilding(w * 0.26, 86, 125, "2.5D Pack");
  drawBuilding(w * 0.74, 125, 118, "AI Cluster");

  // Glowing blue cable + racing packet across the trench!
  ctx.strokeStyle = "#0ea5e9";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(w * 0.15, groundY - 16);
  ctx.lineTo(trenchL, groundY - 4);
  ctx.quadraticCurveTo((trenchL + trenchR) / 2, groundY + trenchDepth - 10, trenchR, groundY - 4);
  ctx.lineTo(w * 0.80, groundY - 22);
  ctx.stroke();

  const pktU = (t * 0.55) % 1.0;
  const pktX = lerp(w * 0.15, w * 0.80, pktU);
  let pktY = groundY - 12;
  if (pktX > trenchL && pktX < trenchR) {
    const tu = (pktX - trenchL) / (trenchR - trenchL);
    pktY = groundY + 4 * tu * (1 - tu) * (trenchDepth - 12);
  }
  ctx.fillStyle = "#38bdf8";
  ctx.beginPath();
  ctx.arc(pktX, pktY, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

// 6. `dark_gpu_partition_split` — Matrix cells overflowing 1 GPU -> spring-splitting across 4 GPUs (`aicost_gpu`!)
function renderDarkGpuPartitionSplit(ctx, assets, scene, w, h, t, dur, amp, isShort) {
  const u = t / Math.max(0.5, dur);
  ctx.fillStyle = "#0b0b0e";
  ctx.fillRect(0, 0, w, h);

  const splitT = easeInOutCubic((u - 0.46) / 0.28);
  const cx = w / 2;
  const cy = isShort ? h * 0.44 : h * 0.50;
  const boxSize = isShort ? 210 : 250;

  const offsets = [
    [-1, -1], [1, -1],
    [-1, 1],  [1, 1],
  ];
  const spreadX = isShort ? 145 : 230;
  const spreadY = isShort ? 145 : 155;

  // Draw pulsing yellow NCCL sync lasers between the 4 split GPUs once split!
  if (splitT > 0.25) {
    ctx.save();
    ctx.globalAlpha = clamp((splitT - 0.25) / 0.5, 0, 1);
    const pts = offsets.map(([ox, oy]) => [cx + ox * spreadX * splitT, cy + oy * spreadY * splitT]);
    const pairs = [[0, 1], [2, 3], [0, 2], [1, 3], [0, 3], [1, 2]];
    pairs.forEach(([a, b], idx) => {
      ctx.strokeStyle = "rgba(251, 191, 36, 0.45)";
      ctx.lineWidth = 2.5;
      ctx.setLineDash([8, 8]);
      ctx.lineDashOffset = -t * 48;
      ctx.beginPath();
      ctx.moveTo(pts[a][0], pts[a][1]);
      ctx.lineTo(pts[b][0], pts[b][1]);
      ctx.stroke();

      const pu = (t * 1.8 + idx * 0.17) % 1.0;
      ctx.setLineDash([]);
      ctx.fillStyle = "#fbbf24";
      ctx.beginPath();
      ctx.arc(lerp(pts[a][0], pts[b][0], pu), lerp(pts[a][1], pts[b][1], pu), 5, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.restore();
  }

  // Draw the 1 -> 4 GPU packages with glowing blue/gold HBM matrix tiles (`aicost_gpu.jpg`!)
  const count = splitT > 0.02 ? 4 : 1;
  for (let g = 0; g < count; g++) {
    const [ox, oy] = offsets[g];
    const gx = cx + ox * spreadX * splitT - boxSize / 2;
    const gy = cy + oy * spreadY * splitT - boxSize / 2;
    const overloaded = splitT <= 0.05 && u > 0.28;

    roundRectPath(ctx, gx, gy, boxSize, boxSize, 18);
    ctx.fillStyle = "#161821";
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = overloaded ? "#ef4444" : splitT > 0.4 ? "#38bdf8" : "#475569";
    ctx.stroke();

    const gridN = 6;
    const pad = 20;
    const cellS = (boxSize - pad * 2 - (gridN - 1) * 6) / gridN;
    const fillCells = splitT > 0.05 ? 20 : Math.floor(lerp(4, 36, clamp(u / 0.36, 0, 1)));

    for (let i = 0; i < gridN * gridN; i++) {
      if (i >= fillCells) break;
      const r = Math.floor(i / gridN), c = i % gridN;
      roundRectPath(ctx, gx + pad + c * (cellS + 6), gy + pad + r * (cellS + 6), cellS, cellS, 4);
      ctx.fillStyle = overloaded ? "#f87171" : g % 2 === 0 ? "#60a5fa" : "#eab308";
      ctx.fill();
    }

    ctx.fillStyle = "#e2e8f0";
    ctx.font = `bold 18px ${MONO}`;
    ctx.textAlign = "center";
    ctx.fillText(splitT > 0.1 ? `GPU #${g + 1} HBM` : overloaded ? "1 GPU HBM: OVERFLOW!" : "Single GPU HBM", gx + boxSize / 2, gy - 14);
  }

  // Minimalist bottom caption label (just like "The prompt / The weights" in `aicost_gpu.jpg`)
  ctx.fillStyle = splitT > 0.3 ? "#fbbf24" : "#94a3b8";
  ctx.font = `bold ${isShort ? 24 : 26}px ${MONO}`;
  ctx.textAlign = "center";
  ctx.fillText(
    splitT > 0.3 ? "Partitioned Across GPUs -> NCCL Sync Overhead" : "Model Weights Exceed Single-Package HBM Perimeter",
    cx,
    isShort ? h * 0.72 : h * 0.87
  );
}

// 7. `map_supply_chain_arcs` — Warm cream vector map + waving flags + flying chip arcs (`asml_intro`!)
function renderMapSupplyChainArcs(ctx, assets, scene, w, h, t, dur, amp, isShort) {
  const u = t / Math.max(0.5, dur);
  ctx.fillStyle = "#f8f5ed";
  ctx.fillRect(0, 0, w, h);

  // Subtle world map grid lines
  ctx.strokeStyle = "rgba(148, 163, 184, 0.18)";
  ctx.lineWidth = 1.5;
  for (let x = 120; x < w; x += 160) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
  for (let y = 120; y < h; y += 160) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  const stops = isShort
    ? [
        { x: w * 0.24, y: h * 0.28, name: "SK hynix", sub: "12-Hi HBM", col: "#93c5fd", flag: ["#ffffff", "#dc2626", "#1d4ed8"] },
        { x: w * 0.76, y: h * 0.44, name: "TSMC", sub: "2.5D CoWoS", col: "#86efac", flag: ["#dc2626", "#1d4ed8", "#ffffff"] },
        { x: w * 0.38, y: h * 0.62, name: "US Cloud", sub: "AI Clusters", col: "#fda4af", flag: ["#1d4ed8", "#ffffff", "#dc2626"] },
      ]
    : [
        { x: w * 0.20, y: h * 0.52, name: "SK hynix (Korea)", sub: "HBM3E Stacks", col: "#93c5fd", flag: ["#ffffff", "#dc2626", "#1d4ed8"] },
        { x: w * 0.50, y: h * 0.62, name: "TSMC (Taiwan)", sub: "2.5D Packaging", col: "#86efac", flag: ["#dc2626", "#1d4ed8", "#ffffff"] },
        { x: w * 0.80, y: h * 0.46, name: "US Hyperscalers", sub: "Nvidia & Apple", col: "#c4b5fd", flag: ["#1d4ed8", "#ffffff", "#dc2626"] },
      ];

  // Draw stylized vector island/region polygons + waving flags (`asml_intro.jpg`!)
  stops.forEach((st, idx) => {
    const sPop = springOut((t - idx * 0.25) / 0.55);
    ctx.save();
    ctx.translate(st.x, st.y);
    ctx.scale(sPop, sPop);

    // Organic landmass silhouette
    ctx.beginPath();
    for (let a = 0; a < Math.PI * 2; a += Math.PI / 7) {
      const rad = 78 + Math.sin(a * 3 + idx * 2) * 18;
      const px = Math.cos(a) * rad * 1.15;
      const py = Math.sin(a) * rad * 0.82;
      if (a === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fillStyle = st.col;
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#1e293b";
    ctx.stroke();

    // Flagpole + waving 3-band flag
    ctx.beginPath();
    ctx.moveTo(-24, -18);
    ctx.lineTo(-24, -108);
    ctx.stroke();
    const fw = 56, fh = 36;
    for (let b = 0; b < 3; b++) {
      ctx.fillStyle = st.flag[b];
      const waveY = Math.sin(t * 5 + b) * 2.5;
      ctx.fillRect(-24, -108 + b * (fh / 3) + waveY, fw, fh / 3);
    }
    ctx.strokeRect(-24, -108, fw, fh);

    ctx.fillStyle = "#0f172a";
    ctx.font = `bold 24px ${SANS}`;
    ctx.textAlign = "center";
    ctx.fillText(st.name, 0, 108);
    ctx.fillStyle = "#475569";
    ctx.font = `bold 18px ${MONO}`;
    ctx.fillText(st.sub, 0, 136);
    ctx.restore();
  });

  // Animated dashed Bezier arcs + flying HBM chip module along each leg!
  for (let leg = 0; leg < stops.length - 1; leg++) {
    const a = stops[leg], b = stops[leg + 1];
    const legStart = 0.12 + leg * 0.36;
    const legU = clamp((u - legStart) / 0.36, 0, 1);
    if (legU <= 0.01) continue;

    const midX = (a.x + b.x) / 2;
    const midY = Math.min(a.y, b.y) - 145;

    ctx.save();
    ctx.strokeStyle = "#475569";
    ctx.lineWidth = 3;
    ctx.setLineDash([10, 10]);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y - 20);
    ctx.quadraticCurveTo(midX, midY, b.x, b.y - 20);
    ctx.stroke();
    ctx.restore();

    const fu = easeInOutCubic(legU);
    const om = 1 - fu;
    const fx = om * om * a.x + 2 * om * fu * midX + fu * fu * b.x;
    const fy = om * om * (a.y - 20) + 2 * om * fu * midY + fu * fu * (b.y - 20);
    drawHbmChipModule(ctx, fx, fy, 0.58, 6, t, leg === 0 ? "HBM" : "CoWoS");
  }
}

// 8. `wafer_conveyor_zoom_out` — 3x Wafer Compression & Smooth Virtual Camera Zoom-Out (`hf_agents` / `2d883UK7p9E`!)
function renderWaferConveyorZoomOut(ctx, assets, scene, w, h, t, dur, amp, isShort) {
  const u = t / Math.max(0.5, dur);
  ctx.fillStyle = "#f4f6fa";
  ctx.fillRect(0, 0, w, h);

  // Smooth 2.1x -> 0.72x camera zoom-out revealing dozens of fab slots (`hf_agents.jpg` camera pull-back!)
  const camScale = lerp(1.95, isShort ? 0.72 : 0.78, easeInOutCubic(clamp(u * 1.15, 0, 1)));
  ctx.save();
  ctx.translate(w / 2, h * (isShort ? 0.44 : 0.50));
  ctx.scale(camScale, camScale);

  const cols = isShort ? 3 : 5;
  const rows = 3;
  const cellW = 285, cellH = 215;
  const ox0 = -((cols - 1) * cellW) / 2;
  const oy0 = -((rows - 1) * cellH) / 2;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c;
      const cx = ox0 + c * cellW;
      const cy = oy0 + r * cellH;
      const appear = springOut((t - idx * 0.08) / 0.45);
      if (appear <= 0.01) continue;

      ctx.save();
      ctx.translate(cx, cy);
      ctx.scale(appear, appear);

      // Clean fab bay card
      roundRectPath(ctx, -118, -82, 236, 164, 16);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.lineWidth = 3;
      ctx.strokeStyle = "#1e293b";
      ctx.stroke();

      // 3 iridescent silicon wafers compressing into 1 HBM tower!
      const mergeP = easeInOutCubic((t * 0.8 + idx * 0.15) % 1.0);
      for (const wOff of [-46, 0, 46]) {
        const wx = lerp(wOff, 0, mergeP);
        ctx.beginPath();
        ctx.arc(wx - 28, 0, 26, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(56, 189, 248, 0.55)";
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#0284c7";
        ctx.stroke();
      }
      drawHbmChipModule(ctx, 52, 8, 0.52, 6, t + idx, "1x HBM");
      ctx.restore();
    }
  }
  ctx.restore();

  // Minimalist floating badge at top
  const metrics = scene.metrics || [];
  const pillTxt = metrics[0] ? `${metrics[0].label}: ${metrics[0].value}` : "1 GB HBM = 3x Wafer Area";
  ctx.font = `bold ${isShort ? 26 : 28}px ${MONO}`;
  const pw = ctx.measureText(pillTxt).width + 44;
  roundRectPath(ctx, (w - pw) / 2, isShort ? 185 : 44, pw, 50, 25);
  ctx.fillStyle = "#0f172a";
  ctx.fill();
  ctx.fillStyle = "#fbbf24";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(pillTxt, w / 2, (isShort ? 185 : 44) + 25);
}

// --- Subtle Unobtrusive Corner Source Citation Watermark (`融合公開資訊 + 附上來源`) ---
function drawSubtleSourcePill(ctx, sources, w, h, t, isShort) {
  if (!sources || !sources.length) return;
  const enter = easeOutCubic((t - 0.2) / 0.4);
  if (enter <= 0.02) return;

  const src = sources[0];
  const label = `Source ${src.badge_label} • ${src.short_url || ""}`.slice(0, 68);
  ctx.save();
  ctx.globalAlpha = 0.86 * enter;
  ctx.font = `bold ${isShort ? 20 : 18}px ${MONO}`;
  const tw = Math.min(w - 64, ctx.measureText(label).width + 34);
  const bx = isShort ? (w - tw) / 2 : 32;
  const by = isShort ? 78 : h - 48;

  roundRectPath(ctx, bx, by, tw, 34, 17);
  ctx.fillStyle = "rgba(15, 23, 42, 0.78)";
  ctx.fill();
  ctx.strokeStyle = "rgba(74, 222, 128, 0.75)";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.fillStyle = "#4ade80";
  ctx.beginPath();
  ctx.arc(bx + 15, by + 17, 4.5, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#f1f5f9";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(label, bx + 26, by + 18);
  ctx.restore();
}

// --- Exact @TradeoffLayer Shorts Progressive Word-Highlighted Captions (`bmFUEo6A_1U`) ---
function drawShortsKaraokeCaptions(ctx, words, t, w, h) {
  if (!words || !words.length) return;
  const groupSize = 5;
  let active = null;
  for (let i = 0; i < words.length; i += groupSize) {
    const g = words.slice(i, i + groupSize);
    if (t >= g[0].start - 0.08 && t <= g[g.length - 1].end + 0.25) {
      active = g;
      break;
    }
  }
  if (!active) return;

  ctx.save();
  ctx.font = `bold 50px ${SANS}`;
  const lines = [[]];
  let curW = 0;
  for (const wd of active) {
    const ww = ctx.measureText(wd.word + " ").width;
    if (lines[lines.length - 1].length > 0 && curW + ww > w - 150 && lines.length < 2) {
      lines.push([wd]);
      curW = ww;
    } else {
      lines[lines.length - 1].push(wd);
      curW += ww;
    }
  }

  const baseY = h - 385;
  const boxH = lines.length * 66 + 32;
  roundRectPath(ctx, 52, baseY - 18, w - 104, boxH, 22);
  ctx.fillStyle = "rgba(10, 11, 15, 0.84)";
  ctx.fill();

  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  lines.forEach((ln, lIdx) => {
    const full = ln.map((x) => x.word).join(" ");
    let cursorX = (w - ctx.measureText(full).width) / 2;
    const ly = baseY + lIdx * 66;
    ln.forEach((wd, wIdx) => {
      const tok = wd.word + (wIdx < ln.length - 1 ? " " : "");
      ctx.fillStyle = t >= wd.start - 0.03 ? "#ffffff" : "#71717a";
      ctx.fillText(tok, cursorX, ly);
      cursorX += ctx.measureText(tok).width;
    });
  });
  ctx.restore();
}

function chooseAnimationType(scene, index) {
  if (scene.animation_type) return scene.animation_type;
  const cycle = [
    "character_walk_handoff",
    "dark_cross_section_meter",
    "rain_buckets_analogy",
    "chip_3d_flip_stack",
    "map_supply_chain_arcs",
    "whiteboard_systolic_sim",
    "dark_gpu_partition_split",
    "character_walk_handoff",
    "wafer_conveyor_zoom_out",
    "whiteboard_systolic_sim",
  ];
  return cycle[index % cycle.length];
}

async function renderSceneVideo(job) {
  const { scene, index = 0, ttsMeta, wavPath, visualDir, outMp4, w, h, fps, isShort } = job;
  const canvas = createCanvas(w, h);
  const ctx = canvas.getContext("2d");

  const tryLoad = async (name) => {
    if (!name) return null;
    const p = path.isAbsolute(name) ? name : path.join(visualDir, name);
    if (fs.existsSync(p)) {
      try { return await loadImage(p); } catch (_) { return null; }
    }
    return null;
  };

  const assets = {
    sceneBg: await tryLoad(job.bgFile),
    stage_courtyard: await tryLoad("stage_courtyard.png"),
    stage_cleanroom: await tryLoad("stage_cleanroom.png"),
    stage_datacenter: await tryLoad("stage_datacenter.png"),
    stage_studio: await tryLoad("stage_studio.png"),
    char_engineer: await tryLoad("char_engineer.png"),
    char_ceo: await tryLoad("char_ceo.png"),
    char_analyst: await tryLoad("char_analyst.png"),
    prop_chip_stack: await tryLoad("prop_chip_stack.png"),
    prop_gpu_board: await tryLoad("prop_gpu_board.png"),
    prop_wafer: await tryLoad("prop_wafer.png"),
  };

  const envelope = job.envelope || [];
  const nFrames = Math.max(1, envelope.length);
  const dur = nFrames / fps;
  const animType = chooseAnimationType(scene, index);
  const words = ttsMeta.words || [];
  const sources = scene.resolved_sources || [];

  const ff = spawn("ffmpeg", [
    "-y", "-loglevel", "error",
    "-f", "rawvideo", "-vcodec", "rawvideo",
    "-pix_fmt", "rgba", "-s", `${w}x${h}`, "-r", String(fps),
    "-i", "-",
    "-i", wavPath,
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
    "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
    "-shortest",
    outMp4,
  ], { stdio: ["pipe", "inherit", "inherit"] });

  const renderFnMap = {
    character_walk_handoff: renderCharacterWalkHandoff,
    rain_buckets_analogy: renderRainBucketsAnalogy,
    chip_3d_flip_stack: renderChip3dFlipStack,
    whiteboard_systolic_sim: renderWhiteboardSystolicSim,
    dark_cross_section_meter: renderDarkCrossSectionMeter,
    dark_gpu_partition_split: renderDarkGpuPartitionSplit,
    map_supply_chain_arcs: renderMapSupplyChainArcs,
    wafer_conveyor_zoom_out: renderWaferConveyorZoomOut,
  };
  const fn = renderFnMap[animType] || renderWhiteboardSystolicSim;
  const fadeFrames = 6;

  for (let f = 0; f < nFrames; f++) {
    const t = f / fps;
    const amp = envelope[f] || 0;
    ctx.clearRect(0, 0, w, h);

    fn(ctx, assets, scene, w, h, t, dur, amp, isShort);
    drawSubtleSourcePill(ctx, sources, w, h, t, isShort);

    if (isShort) {
      drawShortsKaraokeCaptions(ctx, words, t, w, h);
    }

    // Crisp 0.1s cross-fade at scene edges
    if (f < fadeFrames || f >= nFrames - fadeFrames) {
      const fadeAlpha = f < fadeFrames
        ? 1 - (f + 1) / fadeFrames
        : 1 - Math.max(0, (nFrames - f) / fadeFrames);
      ctx.fillStyle = `rgba(11, 11, 14, ${fadeAlpha.toFixed(3)})`;
      ctx.fillRect(0, 0, w, h);
    }

    const rawBuf = canvas.data();
    if (!ff.stdin.write(rawBuf)) {
      await new Promise((resolve) => ff.stdin.once("drain", resolve));
    }
  }

  ff.stdin.end();
  const rc = await new Promise((resolve) => ff.on("close", resolve));
  if (rc !== 0) {
    throw new Error(`ffmpeg exited with code ${rc} on scene ${scene.id}`);
  }
}

async function main() {
  const jobPath = process.argv[2];
  if (!jobPath) {
    console.error("Usage: node skia_animator.js <job.json>");
    process.exit(1);
  }
  const job = JSON.parse(fs.readFileSync(jobPath, "utf-8"));
  await renderSceneVideo(job);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
