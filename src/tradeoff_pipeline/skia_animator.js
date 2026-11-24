#!/usr/bin/env node
/**
 * Tradeoff Layer v3 — Multi-Layer Cartoon Rigging & Minimalist Micro-Infographic 60fps Engine.
 * Powered by Google Skia (`@napi-rs/canvas`), streaming raw RGBA frames directly into FFmpeg.
 *
 * Implements all 18 bespoke @TradeoffLayer visual templates across:
 *   - Illustrated flat-cartoon sets + neural BiRefNet multi-pose character cutouts + keyframed props
 *   - Signature warm cream (#F6EFE0) and midnight navy (#141D35) minimalist data & physics animations
 *   - Native 16:9 (1920x1080) Long-Form AND native 9:16 (1080x1920) Short layouts (zero KTV clutter)
 *   - Subtle fade-in/out frameless typographic source footnotes on quantitative metric shots
 */

const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const { createCanvas, loadImage } = require("@napi-rs/canvas");

const SANS = '"DejaVu Sans", "Liberation Sans", -apple-system, BlinkMacSystemFont, sans-serif';
const SERIF = '"DejaVu Serif", "Liberation Serif", Georgia, serif';
const MONO = '"DejaVu Sans Mono", "Liberation Mono", monospace';

const INK = "#18202C";
const CREAM = "#F6EFE0";
const ICE = "#EDF2F7";
const MIDNIGHT = "#141D35";

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const lerp = (a, b, t) => a + (b - a) * clamp(t, 0, 1);
const easeOutCubic = (t) => 1 - Math.pow(1 - clamp(t, 0, 1), 3);
const easeInOutCubic = (t) => {
  const x = clamp(t, 0, 1);
  return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
};
const springOut = (t) => {
  const x = clamp(t, 0, 1);
  if (x === 0 || x === 1) return x;
  return 1 - Math.pow(2, -9 * x) * Math.cos(x * Math.PI * 2.2);
};

function roundRect(ctx, x, y, w, h, r) {
  const rad = Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2);
  ctx.beginPath();
  ctx.moveTo(x + rad, y);
  ctx.arcTo(x + w, y, x + w, y + h, rad);
  ctx.arcTo(x + w, y + h, x, y + h, rad);
  ctx.arcTo(x, y + h, x, y, rad);
  ctx.arcTo(x, y, x + w, y, rad);
  ctx.closePath();
}

function fillStrokeBox(ctx, x, y, w, h, r, fill, stroke = INK, lw = 3.5) {
  roundRect(ctx, x, y, w, h, r);
  if (fill) {
    ctx.fillStyle = fill;
    ctx.fill();
  }
  if (stroke && lw > 0) {
    ctx.lineWidth = lw;
    ctx.strokeStyle = stroke;
    ctx.stroke();
  }
}

function drawCover(ctx, img, w, h, zoom = 1.0, panX = 0, panY = 0, fallback = ICE) {
  if (!img) {
    ctx.fillStyle = fallback;
    ctx.fillRect(0, 0, w, h);
    return;
  }
  const scale = Math.max(w / img.width, h / img.height) * zoom;
  const dw = img.width * scale;
  const dh = img.height * scale;
  const dx = (w - dw) * 0.5 + panX;
  const dy = (h - dh) * 0.5 + panY;
  ctx.drawImage(img, dx, dy, dw, dh);
}

// Crisp @TradeoffLayer Pinned Semiconductor Chip Vector Icon (Intel/AMD/SK/Samsung style!)
function drawPinnedChip(ctx, cx, cy, size, bodyFill, labelText, textFill = "#FFFFFF", subText = "") {
  ctx.save();
  ctx.translate(cx, cy);
  const hs = size * 0.5;
  const pinLen = size * 0.11;
  const pinW = Math.max(4, size * 0.065);
  const pins = 6;
  ctx.fillStyle = "#F59E0B";
  ctx.strokeStyle = INK;
  ctx.lineWidth = Math.max(2.2, size * 0.025);

  for (let i = 0; i < pins; i++) {
    const off = lerp(-hs * 0.72, hs * 0.72, i / (pins - 1));
    // Top & bottom pins
    fillStrokeBox(ctx, off - pinW / 2, -hs - pinLen + 2, pinW, pinLen, 2, "#FBBF24", INK, 2.2);
    fillStrokeBox(ctx, off - pinW / 2, hs - 2, pinW, pinLen, 2, "#FBBF24", INK, 2.2);
    // Left & right pins
    fillStrokeBox(ctx, -hs - pinLen + 2, off - pinW / 2, pinLen, pinW, 2, "#FBBF24", INK, 2.2);
    fillStrokeBox(ctx, hs - 2, off - pinW / 2, pinLen, pinW, 2, "#FBBF24", INK, 2.2);
  }

  fillStrokeBox(ctx, -hs, -hs, size, size, size * 0.13, bodyFill, INK, Math.max(3, size * 0.032));
  roundRect(ctx, -hs * 0.78, -hs * 0.78, size * 0.78, size * 0.78, size * 0.08);
  ctx.strokeStyle = "rgba(255,255,255,0.28)";
  ctx.lineWidth = 2;
  ctx.stroke();

  if (labelText) {
    ctx.fillStyle = textFill;
    let fSize = Math.round(size * 0.22);
    ctx.font = `900 ${fSize}px ${SANS}`;
    const maxTextW = size * 0.72;
    const measured = ctx.measureText(labelText).width;
    if (measured > maxTextW && measured > 0) {
      fSize = Math.max(11, Math.floor(fSize * (maxTextW / measured)));
      ctx.font = `900 ${fSize}px ${SANS}`;
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(labelText, 0, subText ? -size * 0.08 : 0);
  }
  if (subText) {
    ctx.fillStyle = "rgba(255,255,255,0.88)";
    let sfSize = Math.round(size * 0.125);
    ctx.font = `700 ${sfSize}px ${MONO}`;
    const sm = ctx.measureText(subText).width;
    if (sm > size * 0.72 && sm > 0) {
      sfSize = Math.max(10, Math.floor(sfSize * ((size * 0.72) / sm)));
      ctx.font = `700 ${sfSize}px ${MONO}`;
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(subText, 0, size * 0.20);
  }
  ctx.restore();
}

// Waist-up cartoon character behind a desk/counter (natural hands/torso, gentle talk motion)
function drawDeskActor(ctx, sprite, cx, bottomY, targetH, amp, t, flipX = false) {
  if (!sprite) return;
  const bob = Math.sin(t * 4.5) * 2.5 + amp * 5.0;
  const stretch = 1.0 + amp * 0.022;
  const aspect = sprite.width / Math.max(1, sprite.height);
  const dh = targetH * stretch;
  const dw = targetH * aspect;

  ctx.save();
  ctx.translate(cx, bottomY - bob * 0.4);
  if (flipX) ctx.scale(-1, 1);
  ctx.rotate(Math.sin(t * 2.8) * 0.012);
  ctx.drawImage(sprite, -dw * 0.5, -dh, dw, dh);
  ctx.restore();
}

// Full-body @TradeoffLayer character: crisp waist-up cartoon sprite + thin parallel twin black legs (`||`)
function drawStandingActor(ctx, sprite, cx, groundY, torsoH, legH, walkSpeed, amp, t, flipX = false) {
  const walking = Math.abs(walkSpeed) > 2.0;
  const stride = walking ? Math.sin(t * 7.5) * 14 : 0;
  const bounce = walking ? Math.abs(Math.cos(t * 7.5)) * 5 : amp * 3.5;
  const hipY = groundY - legH - bounce;
  const legGap = torsoH * 0.105;

  // Soft ground shadow
  ctx.save();
  ctx.fillStyle = "rgba(24,32,44,0.13)";
  ctx.beginPath();
  ctx.ellipse(cx, groundY + 3, torsoH * 0.24, 7, 0, 0, Math.PI * 2);
  ctx.fill();

  // Iconic thin twin black ink legs (`hi_amd_14s` & `short_ref_tpu` exact proportion!)
  ctx.strokeStyle = INK;
  ctx.lineWidth = 3.6;
  ctx.lineCap = "round";
  const lFootX = cx - legGap + stride;
  const rFootX = cx + legGap - stride;
  const lFootY = groundY - (walking ? Math.max(0, Math.sin(t * 7.5)) * 6 : 0);
  const rFootY = groundY - (walking ? Math.max(0, -Math.sin(t * 7.5)) * 6 : 0);

  ctx.beginPath();
  ctx.moveTo(cx - legGap, hipY - 12);
  ctx.lineTo(lFootX, lFootY);
  ctx.moveTo(cx + legGap, hipY - 12);
  ctx.lineTo(rFootX, rFootY);
  ctx.stroke();

  // Minimalist flat black oval shoes
  ctx.fillStyle = INK;
  const shoeDir = flipX ? -1 : 1;
  ctx.beginPath();
  ctx.ellipse(lFootX + shoeDir * 5, lFootY, 11, 4.5, 0, Math.PI, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(rFootX + shoeDir * 5, rFootY, 11, 4.5, 0, Math.PI, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  drawDeskActor(ctx, sprite, cx, hipY + 14, torsoH, amp, t, flipX);
}

function drawPropSpriteOrChip(ctx, propImg, cx, cy, size, fallbackLabel = "HBM") {
  if (propImg) {
    const aspect = propImg.width / Math.max(1, propImg.height);
    const dw = aspect >= 1 ? size : size * aspect;
    const dh = aspect >= 1 ? size / aspect : size;
    ctx.drawImage(propImg, cx - dw * 0.5, cy - dh * 0.5, dw, dh);
  } else {
    drawPinnedChip(ctx, cx, cy, size * 0.82, "#2563EB", fallbackLabel);
  }
}

// Subtle frameless typographic footnote ONLY on quantitative source-cited shots (zero black terminal bars!)
function drawSubtleSourceFootnote(ctx, scene, w, h, u, onDark = false) {
  const note = (scene.citation_note || "").trim();
  if (!note) return;
  const fadeIn = easeOutCubic((u - 0.10) / 0.14);
  const fadeOut = 1.0 - easeInOutCubic((u - 0.84) / 0.14);
  const alpha = clamp(fadeIn * fadeOut, 0, 1) * 0.62;
  if (alpha <= 0.02) return;

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = onDark ? "#CBD5E1" : "#475569";
  ctx.font = `600 15px ${SANS}`;
  ctx.textAlign = "right";
  ctx.textBaseline = "bottom";
  ctx.fillText(`Source: ${note}`, w - 34, h - 22);
  ctx.restore();
}

// Clean architectural signplate integrated into scene (like `AMD`, `intel`, `SONY` building signs)
function drawCleanSignplate(ctx, cx, cy, text, accent = "#1D4ED8", sc = 1.0) {
  if (!text) return;
  ctx.save();
  ctx.font = `900 ${Math.round(26 * sc)}px ${SANS}`;
  const tw = ctx.measureText(text).width;
  const pw = tw + 44 * sc;
  const ph = 48 * sc;
  fillStrokeBox(ctx, cx - pw / 2, cy - ph / 2, pw, ph, 10 * sc, "#FFFFFF", INK, 3.2);
  ctx.fillStyle = accent;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, cx, cy + 1);
  ctx.restore();
}

// ============================================================================
// 18 BESPOKE @TRADEOFFLAYER SHOT TEMPLATES
// ============================================================================

// 1. `exterior_hq_walk`: Sunny HQ / cleanroom corridor walk-in with prop
function tplExteriorHqWalk(ctx, A, sc, w, h, t, dur, amp, isShort) {
  const u = t / dur;
  drawCover(ctx, A.bg, w, h, lerp(1.0, 1.065, u), lerp(-12, 12, u), 0, "#DCEEFF");

  const groundY = isShort ? h * 0.80 : h * 0.88;
  const torsoH = isShort ? 330 : 340;
  const legH = isShort ? 175 : 155;
  const startX = isShort ? w * 0.24 : w * 0.30;
  const endX = isShort ? w * 0.46 : w * 0.45;
  const walkU = clamp(u / 0.58, 0, 1);
  const actorX = lerp(startX, endX, easeOutCubic(walkU));
  const speed = walkU < 0.98 ? 40 : 0;

  if (sc.big_stat) {
    const signY = isShort ? h * 0.15 : h * 0.12;
    const pop = springOut(u / 0.25);
    ctx.save();
    ctx.translate(w * 0.5, signY);
    ctx.scale(pop, pop);
    drawCleanSignplate(ctx, 0, 0, `${sc.big_stat} · ${sc.visual_label || ""}`.replace(/ · $/, ""), "#0F172A", isShort ? 1.12 : 1.04);
    ctx.restore();
  }

  drawStandingActor(ctx, A.leftActor || A.fallbackActor, actorX, groundY, torsoH, legH, speed, amp, t, false);

  if (A.prop || sc.big_stat) {
    const px = actorX + (isShort ? 155 : 175);
    const py = groundY - legH - torsoH * 0.38 + Math.sin(t * 4.0) * 5;
    const pScale = springOut((u - 0.12) / 0.28);
    ctx.save();
    ctx.translate(px, py);
    ctx.scale(pScale, pScale);
    drawPropSpriteOrChip(ctx, A.prop, 0, 0, isShort ? 155 : 165, sc.big_stat || "HBM");
    ctx.restore();
  }
  drawSubtleSourceFootnote(ctx, sc, w, h, u, false);
}

// 2. `desk_deal_transfer`: Two characters across foreground wooden desk sliding check/contract/chip
function tplDeskDealTransfer(ctx, A, sc, w, h, t, dur, amp, isShort) {
  const u = t / dur;
  drawCover(ctx, A.bg, w, h, lerp(1.0, 1.05, u), 0, 0, "#E8F1F8");

  const deskY = isShort ? h * 0.67 : h * 0.72;
  const charH = isShort ? 355 : 410;
  const lx = isShort ? w * 0.26 : w * 0.27;
  const rx = isShort ? w * 0.74 : w * 0.73;

  drawDeskActor(ctx, A.leftActor || A.fallbackActor, lx, deskY + 26, charH, amp, t, false);
  drawDeskActor(ctx, A.rightActor || A.altActor, rx, deskY + 26, charH, amp * 0.7, t + 1.1, true);

  // Bold-outlined foreground warm oak desk (`hi_amd_14s` / `hi_sony_38s` style!)
  fillStrokeBox(ctx, -20, deskY, w + 40, h - deskY + 20, 0, "#D7A86E", INK, 4.5);
  ctx.fillStyle = "#C39155";
  ctx.fillRect(0, deskY + 22, w, 14);

  // Deal document / check / HBM allocation sliding across desk!
  const slideP = easeOutCubic((u - 0.16) / 0.46);
  const docX = lerp(lx + 40, (lx + rx) * 0.5, slideP);
  const docY = deskY - 26;
  const cardW = isShort ? 330 : 370;
  const cardH = isShort ? 115 : 122;

  ctx.save();
  ctx.translate(docX, docY);
  fillStrokeBox(ctx, -cardW / 2, -cardH / 2, cardW, cardH, 12, "#FFFBEB", INK, 3.6);
  ctx.fillStyle = "#64748B";
  ctx.font = `800 17px ${MONO}`;
  ctx.textAlign = "left";
  ctx.fillText(sc.visual_label || "ALLOCATION CONTRACT", -cardW / 2 + 22, -cardH / 2 + 34);
  ctx.fillStyle = "#0F172A";
  ctx.font = `900 ${isShort ? 38 : 42}px ${SANS}`;
  ctx.fillText(sc.big_stat || sc.secondary_label || "APPROVED", -cardW / 2 + 22, cardH / 2 - 22);
  if (A.prop) {
    drawPropSpriteOrChip(ctx, A.prop, cardW / 2 - 58, 4, 86, "HBM");
  }
  ctx.restore();

  drawSubtleSourceFootnote(ctx, sc, w, h, u, false);
}

// 3. `copier_blueprint_action`: Exact `amd_micro_0s.jpg` homage (hand places chip -> lid closes -> scan beam!)
function tplCopierBlueprintAction(ctx, A, sc, w, h, t, dur, amp, isShort) {
  const u = t / dur;
  drawCover(ctx, A.bg, w, h, lerp(1.0, 1.04, u), 0, 0, "#E2ECF5");

  const cx = w * 0.5;
  const cy = isShort ? h * 0.58 : h * 0.64;
  const scf = isShort ? 0.88 : 1.08;

  // Copier main body
  fillStrokeBox(ctx, cx - 320 * scf, cy - 40 * scf, 640 * scf, 250 * scf, 18 * scf, "#E5E2DC", INK, 4);
  // Cyan glass platen
  fillStrokeBox(ctx, cx - 250 * scf, cy - 22 * scf, 500 * scf, 82 * scf, 8 * scf, "#BAE6FD", INK, 3);
  // Control panel buttons
  fillStrokeBox(ctx, cx - 180 * scf, cy + 90 * scf, 150 * scf, 52 * scf, 8 * scf, "#93C5FD", INK, 2.8);
  ctx.fillStyle = "#10B981";
  ctx.beginPath();
  ctx.arc(cx + 65 * scf, cy + 116 * scf, 18 * scf, 0, Math.PI * 2);
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = INK;
  ctx.stroke();

  // Chip / blueprint placed onto glass platen (`0..0.30`)
  const placeP = easeOutCubic(u / 0.28);
  const chipX = lerp(cx - 380 * scf, cx, placeP);
  const chipY = cy + 18 * scf;
  drawPinnedChip(ctx, chipX, chipY, 108 * scf, "#1D4ED8", (sc.visual_label || "HBM").slice(0, 10));

  // Cartoon sleeved arm sliding chip in (`0..0.28`) then retracting (`0.28..0.46`)
  if (u < 0.48) {
    const outP = u < 0.28 ? placeP : 1.0 - easeInOutCubic((u - 0.28) / 0.20);
    const hx = lerp(-240 * scf, chipX - 52 * scf, outP);
    ctx.save();
    fillStrokeBox(ctx, hx - 260 * scf, chipY - 30 * scf, 210 * scf, 58 * scf, 10 * scf, "#1E40AF", INK, 3.5);
    fillStrokeBox(ctx, hx - 55 * scf, chipY - 24 * scf, 68 * scf, 46 * scf, 18 * scf, "#F6C39B", INK, 3.5);
    ctx.restore();
  }

  // Green optical scan bar sweeping across glass (`0.66..0.98`)
  if (u > 0.64) {
    const scanP = easeInOutCubic((u - 0.64) / 0.32);
    const sx = lerp(cx - 235 * scf, cx + 235 * scf, scanP);
    ctx.save();
    ctx.fillStyle = "rgba(16, 185, 129, 0.55)";
    ctx.fillRect(sx - 12 * scf, cy - 20 * scf, 24 * scf, 78 * scf);
    ctx.restore();
  }

  // Hinged top scanner lid rotating closed (`0.44..0.68`)!
  const closeP = easeInOutCubic((u - 0.44) / 0.24);
  const lidH = lerp(175 * scf, 22 * scf, closeP);
  const lidY = (cy - 24 * scf) - lidH;
  fillStrokeBox(ctx, cx - 265 * scf, lidY, 530 * scf, lidH, 10 * scf, "#F1EFEA", INK, 3.8);

  if (sc.visual_label) {
    drawCleanSignplate(ctx, cx, isShort ? h * 0.20 : h * 0.15, sc.visual_label, "#1E3A8A", scf);
  }
  drawSubtleSourceFootnote(ctx, sc, w, h, u, false);
}

// 4. `hydraulic_press_crush`: Industrial press slamming down onto chip (`first32s_amd_gXhr` row 4!)
function tplHydraulicPressCrush(ctx, A, sc, w, h, t, dur, amp, isShort) {
  const u = t / dur;
  const impact = u >= 0.52 ? Math.exp(-(u - 0.52) * 18) * Math.sin((u - 0.52) * 90) * 10 : 0;
  drawCover(ctx, A.bg, w, h, 1.03, impact * 0.5, impact, "#E5ECF2");

  const pressX = isShort ? w * 0.42 : w * 0.37;
  const anvilY = isShort ? h * 0.66 : h * 0.74;
  const scf = isShort ? 0.90 : 1.05;

  // Actor watching anxiously on right
  drawDeskActor(ctx, A.leftActor || A.fallbackActor, isShort ? w * 0.78 : w * 0.78, h, isShort ? 380 : 420, amp, t, true);

  // Steel guide columns & base anvil
  fillStrokeBox(ctx, pressX - 210 * scf, 80, 34 * scf, anvilY - 60, 4, "#64748B", INK, 3.5);
  fillStrokeBox(ctx, pressX + 176 * scf, 80, 34 * scf, anvilY - 60, 4, "#64748B", INK, 3.5);
  fillStrokeBox(ctx, pressX - 240 * scf, anvilY, 480 * scf, 95 * scf, 10, "#94A3B8", INK, 4);

  // Target chip on anvil (squashes slightly after impact!)
  const crushP = easeInOutCubic((u - 0.28) / 0.26);
  const sqY = crushP > 0.9 ? 0.72 : 1.0;
  ctx.save();
  ctx.translate(pressX, anvilY - 46 * scf * sqY);
  ctx.scale(1.0 + (1.0 - sqY) * 0.25, sqY);
  drawPinnedChip(ctx, 0, 0, 118 * scf, "#DC2626", (sc.secondary_label || "12-Hi").slice(0, 9));
  ctx.restore();

  // Descending hydraulic head
  const headY = lerp(anvilY - 310 * scf, anvilY - 152 * scf, crushP);
  fillStrokeBox(ctx, pressX - 42 * scf, 40, 84 * scf, headY - 30, 0, "#475569", INK, 3.5);
  fillStrokeBox(ctx, pressX - 195 * scf, headY, 390 * scf, 86 * scf, 12, "#334155", INK, 4);
  ctx.fillStyle = "#FFFFFF";
  ctx.font = `900 ${Math.round(28 * scf)}px ${SANS}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(sc.visual_label || "BOTTLENECK", pressX, headY + 43 * scf);

  drawSubtleSourceFootnote(ctx, sc, w, h, u, false);
}

// 5. `studio_shutter_flash`: Duo presenting prop + crisp white camera shutter flash & zoom cut (`Sony` 35s!)
function tplStudioShutterFlash(ctx, A, sc, w, h, t, dur, amp, isShort) {
  const u = t / dur;
  const flashed = u >= 0.46;
  const camZoom = flashed ? 1.11 : lerp(1.0, 1.03, u / 0.46);

  ctx.save();
  ctx.translate(w * 0.5, h * 0.5);
  ctx.scale(camZoom, camZoom);
  ctx.translate(-w * 0.5, -h * 0.5);

  drawCover(ctx, A.bg, w, h, 1.02, 0, 0, CREAM);
  const deskY = isShort ? h * 0.70 : h * 0.78;
  const charH = isShort ? 360 : 420;
  drawDeskActor(ctx, A.leftActor || A.fallbackActor, w * 0.29, deskY + 24, charH, amp, t, false);
  drawDeskActor(ctx, A.rightActor || A.altActor, w * 0.71, deskY + 24, charH, amp * 0.8, t + 0.8, true);

  // Floating hero prop between characters
  const propY = deskY - charH * 0.48 + Math.sin(t * 3.6) * 7;
  drawPropSpriteOrChip(ctx, A.prop, w * 0.5, propY, isShort ? 165 : 185, sc.big_stat || sc.visual_label || "HBM");

  fillStrokeBox(ctx, -30, deskY, w + 60, h - deskY + 40, 0, "#E6BE8A", INK, 4.2);
  if (sc.visual_label) {
    drawCleanSignplate(ctx, w * 0.5, deskY + (isShort ? 78 : 68), sc.big_stat ? `${sc.visual_label} · ${sc.big_stat}` : sc.visual_label, "#0F172A", 1.0);
  }
  ctx.restore();

  // Crisp studio shutter white flash around u=0.46
  const flashAlpha = Math.max(0, 1.0 - Math.abs(u - 0.46) / 0.045) * 0.82;
  if (flashAlpha > 0.01) {
    ctx.fillStyle = `rgba(255,255,255,${flashAlpha.toFixed(3)})`;
    ctx.fillRect(0, 0, w, h);
  }
  drawSubtleSourceFootnote(ctx, sc, w, h, u, false);
}

// 6. `split_screen_rivalry`: Diagonal comic panel split (`ARM` / `TPU vs GPU` style!)
function tplSplitScreenRivalry(ctx, A, sc, w, h, t, dur, amp, isShort) {
  const u = t / dur;
  const p1 = easeOutCubic(u / 0.25);
  const p2 = easeOutCubic((u - 0.14) / 0.28);

  // Panel 1 (Left / Top)
  ctx.save();
  ctx.beginPath();
  if (isShort) {
    ctx.moveTo(0, 0);
    ctx.lineTo(w, 0);
    ctx.lineTo(w, h * 0.48);
    ctx.lineTo(0, h * 0.54);
  } else {
    ctx.moveTo(0, 0);
    ctx.lineTo(w * 0.54, 0);
    ctx.lineTo(w * 0.46, h);
    ctx.lineTo(0, h);
  }
  ctx.closePath();
  ctx.clip();
  ctx.fillStyle = "#DBEAFE";
  ctx.fillRect(0, 0, w, h);
  drawCover(ctx, A.bg, w, h, 1.04, lerp(-40, 0, p1), 0, "#DBEAFE");
  drawDeskActor(
    ctx,
    A.leftActor || A.fallbackActor,
    isShort ? w * 0.5 : w * 0.25,
    isShort ? h * 0.52 : h * 0.94,
    isShort ? 390 : 450,
    amp,
    t,
    false
  );
  if (sc.visual_label) {
    drawCleanSignplate(ctx, isShort ? w * 0.5 : w * 0.24, isShort ? h * 0.11 : h * 0.13, sc.visual_label, "#1E40AF", 0.96);
  }
  ctx.restore();

  // Panel 2 (Right / Bottom)
  ctx.save();
  ctx.globalAlpha = p2;
  ctx.beginPath();
  if (isShort) {
    ctx.moveTo(0, h * 0.54);
    ctx.lineTo(w, h * 0.48);
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
  } else {
    ctx.moveTo(w * 0.54, 0);
    ctx.lineTo(w, 0);
    ctx.lineTo(w, h);
    ctx.lineTo(w * 0.46, h);
  }
  ctx.closePath();
  ctx.clip();
  ctx.fillStyle = "#FEF3C7";
  ctx.fillRect(0, 0, w, h);
  drawDeskActor(
    ctx,
    A.rightActor || A.altActor,
    isShort ? w * 0.5 : w * 0.75,
    isShort ? h * 0.96 : h * 0.94,
    isShort ? 390 : 450,
    amp * 0.85,
    t + 1.2,
    true
  );
  if (sc.secondary_label) {
    drawCleanSignplate(ctx, isShort ? w * 0.5 : w * 0.76, isShort ? h * 0.59 : h * 0.13, sc.secondary_label, "#B45309", 0.96);
  }
  ctx.restore();

  // Bold comic diagonal divider
  ctx.strokeStyle = INK;
  ctx.lineWidth = 9;
  ctx.beginPath();
  if (isShort) {
    ctx.moveTo(0, h * 0.54);
    ctx.lineTo(w, h * 0.48);
  } else {
    ctx.moveTo(w * 0.54, 0);
    ctx.lineTo(w * 0.46, h);
  }
  ctx.stroke();

  drawSubtleSourceFootnote(ctx, sc, w, h, u, false);
}

// 7. `cleanroom_microscope_zoom`: Lab bench + expanding optical loupe revealing thinned silicon die
function tplCleanroomMicroscopeZoom(ctx, A, sc, w, h, t, dur, amp, isShort) {
  const u = t / dur;
  drawCover(ctx, A.bg, w, h, lerp(1.0, 1.06, u), 0, 0, "#E0F2FE");

  const charX = isShort ? w * 0.30 : w * 0.26;
  drawDeskActor(ctx, A.leftActor || A.fallbackActor, charX, h * 0.95, isShort ? 380 : 450, amp, t, false);

  // Expanding circular magnifying loupe inspecting silicon wafer/die
  const loupeP = springOut((u - 0.12) / 0.38);
  const lx = isShort ? w * 0.56 : w * 0.65;
  const ly = isShort ? h * 0.42 : h * 0.48;
  const rad = (isShort ? 220 : 250) * loupeP;

  if (rad > 10) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(lx, ly, rad, 0, Math.PI * 2);
    ctx.fillStyle = "#0F172A";
    ctx.fill();
    ctx.clip();

    // Glowing microscopic wafer die grid inside loupe
    const step = 36;
    ctx.strokeStyle = "rgba(56, 189, 248, 0.38)";
    ctx.lineWidth = 2;
    const shift = (t * 28) % step;
    for (let gx = lx - rad - step; gx <= lx + rad + step; gx += step) {
      ctx.beginPath();
      ctx.moveTo(gx + shift, ly - rad);
      ctx.lineTo(gx + shift, ly + rad);
      ctx.stroke();
    }
    for (let gy = ly - rad - step; gy <= ly + rad + step; gy += step) {
      ctx.beginPath();
      ctx.moveTo(lx - rad, gy);
      ctx.lineTo(lx + rad, gy);
      ctx.stroke();
    }
    drawPropSpriteOrChip(ctx, A.prop, lx, ly - 18, rad * 1.05, sc.visual_label || "DIE");

    ctx.fillStyle = "#38BDF8";
    ctx.font = `900 ${isShort ? 32 : 36}px ${MONO}`;
    ctx.textAlign = "center";
    ctx.fillText(sc.big_stat || sc.visual_label || "SILICON DIE", lx, ly + rad * 0.62);
    ctx.restore();

    ctx.beginPath();
    ctx.arc(lx, ly, rad, 0, Math.PI * 2);
    ctx.lineWidth = 10;
    ctx.strokeStyle = "#F8FAFC";
    ctx.stroke();
    ctx.lineWidth = 4;
    ctx.strokeStyle = INK;
    ctx.stroke();
  }
  drawSubtleSourceFootnote(ctx, sc, w, h, u, false);
}

// 8. `server_hall_waiting`: AI Data Center server racks pulsing amber (`WAITING`)
function tplServerHallWaiting(ctx, A, sc, w, h, t, dur, amp, isShort) {
  const u = t / dur;
  drawCover(ctx, A.bg, w, h, lerp(1.0, 1.05, u), 0, 0, "#1E293B");
  ctx.fillStyle = "rgba(15, 23, 42, 0.35)";
  ctx.fillRect(0, 0, w, h);

  const nRacks = isShort ? 3 : 4;
  const rackW = isShort ? 240 : 250;
  const rackH = isShort ? 520 : 540;
  const gap = isShort ? 44 : 68;
  const totalW = nRacks * rackW + (nRacks - 1) * gap;
  const startX = (w - totalW) * 0.5;
  const rackY = isShort ? h * 0.33 : h * 0.25;

  for (let r = 0; r < nRacks; r++) {
    const rx = startX + r * (rackW + gap);
    const pop = springOut((u - r * 0.06) / 0.30);
    ctx.save();
    ctx.translate(rx + rackW / 2, rackY + rackH);
    ctx.scale(pop, pop);
    ctx.translate(-(rx + rackW / 2), -(rackY + rackH));

    fillStrokeBox(ctx, rx, rackY, rackW, rackH, 14, "#1E293B", INK, 4);
    for (let b = 0; b < 8; b++) {
      const by = rackY + 26 + b * 60;
      fillStrokeBox(ctx, rx + 20, by, rackW - 40, 44, 6, "#0F172A", "#334155", 2);
      // Blinking amber/red status LEDs showing starved GPU blades
      const blink = Math.sin(t * 6.0 + r + b * 0.5) > 0;
      ctx.fillStyle = blink ? "#F59E0B" : "#EF4444";
      ctx.beginPath();
      ctx.arc(rx + 42, by + 22, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#334155";
      ctx.fillRect(rx + 64, by + 17, rackW - 104, 10);
    }
    ctx.restore();
  }

  if (sc.visual_label || sc.big_stat) {
    drawCleanSignplate(
      ctx,
      w * 0.5,
      isShort ? h * 0.18 : h * 0.13,
      sc.big_stat ? `${sc.big_stat} · ${sc.visual_label}` : sc.visual_label,
      "#DC2626",
      1.08
    );
  }
  drawSubtleSourceFootnote(ctx, sc, w, h, u, true);
}

// 9. `oven_conveyor_bottleneck`: Industrial oven conveyor + swinging pendulum price/cost tag (`short_ref_aicost`!)
function tplOvenConveyorBottleneck(ctx, A, sc, w, h, t, dur, amp, isShort) {
  const u = t / dur;
  ctx.fillStyle = CREAM;
  ctx.fillRect(0, 0, w, h);

  const beltY = isShort ? h * 0.62 : h * 0.68;
  const ovenX = isShort ? w * 0.08 : w * 0.10;
  const ovenW = isShort ? 320 : 380;
  const ovenH = isShort ? 320 : 350;

  // Conveyor belt & rollers
  fillStrokeBox(ctx, ovenX + ovenW - 20, beltY, w - ovenX - ovenW + 40, 46, 20, "#64748B", INK, 3.8);
  for (let k = 0; k < 8; k++) {
    const rx = ovenX + ovenW + 30 + k * 145;
    if (rx > w - 30) break;
    ctx.save();
    ctx.translate(rx, beltY + 23);
    ctx.rotate(t * 4.5);
    ctx.fillStyle = "#CBD5E1";
    ctx.beginPath();
    ctx.arc(0, 0, 14, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = INK;
    ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.restore();
  }

  // Chips moving on belt & bunching up at narrow gate
  const gateX = w * 0.78;
  for (let c = 0; c < 5; c++) {
    const rawX = ovenX + ovenW + 40 + ((u * 520 + c * 115) % (gateX - ovenX - ovenW - 30));
    const cx = Math.min(rawX, gateX - 55 - c * 22);
    drawPinnedChip(ctx, cx, beltY - 42, 74, c % 2 === 0 ? "#2563EB" : "#10B981", "HBM");
  }

  // Industrial Packaging Oven on left
  fillStrokeBox(ctx, ovenX, beltY - ovenH + 30, ovenW, ovenH, 18, "#334155", INK, 4.2);
  fillStrokeBox(ctx, ovenX + 36, beltY - ovenH + 95, ovenW - 72, 130, 10, "#FEF08A", INK, 3.2);
  ctx.fillStyle = "#F97316";
  ctx.fillRect(ovenX + 42, beltY - ovenH + 185, ovenW - 84, 34);
  ctx.fillStyle = "#FFFFFF";
  ctx.font = `900 24px ${SANS}`;
  ctx.textAlign = "center";
  ctx.fillText(sc.visual_label || "HBM LINE", ovenX + ovenW / 2, beltY - ovenH + 66);

  // Swinging hanging tag on wire (`40,000$` homage from `short_ref_aicost.jpg`!)
  if (sc.big_stat) {
    const anchorX = w * 0.68;
    const swing = Math.sin(t * 3.4) * 0.11 * Math.exp(-u * 1.1);
    ctx.save();
    ctx.translate(anchorX, 0);
    ctx.rotate(swing);
    ctx.strokeStyle = INK;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(0, isShort ? 260 : 190);
    ctx.stroke();

    const tagY = isShort ? 250 : 180;
    fillStrokeBox(ctx, -145, tagY, 290, 118, 16, "#FEF9C3", INK, 3.8);
    ctx.fillStyle = "#DC2626";
    ctx.font = `900 44px ${SANS}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(sc.big_stat, 0, tagY + 62);
    ctx.restore();
  }
  drawSubtleSourceFootnote(ctx, sc, w, h, u, false);
}

// 10. `waffle_100_dots`: Exact `sony_micro_35s` (`50¢ / $1`) cream minimalist 100-dot matrix + giant bold stat!
function tplWaffle100Dots(ctx, A, sc, w, h, t, dur, amp, isShort) {
  const u = t / dur;
  ctx.fillStyle = CREAM;
  ctx.fillRect(0, 0, w, h);

  // Parse percentage if present, else default 53
  const m = (sc.big_stat || "").match(/(\d+)/);
  const targetCount = m ? clamp(parseInt(m[1], 10), 12, 92) : 53;
  const revealCount = Math.round(lerp(0, 100, easeOutCubic(u / 0.58)));

  const gridSize = isShort ? 520 : 470;
  const gx = isShort ? (w - gridSize) * 0.5 : w * 0.14;
  const gy = isShort ? h * 0.22 : (h - gridSize) * 0.5;
  const step = gridSize / 10;
  const r = step * 0.34;

  for (let idx = 0; idx < 100; idx++) {
    if (idx >= revealCount) break;
    const row = Math.floor(idx / 10);
    const col = idx % 10;
    const dx = gx + col * step + step * 0.5;
    const dy = gy + row * step + step * 0.5;
    ctx.fillStyle = idx < targetCount ? "#2563EB" : "#F59E0B";
    ctx.beginPath();
    ctx.arc(dx, dy, r, 0, Math.PI * 2);
    ctx.fill();
  }

  // Right / Bottom clean editorial typography (zero ugly boxes!)
  const tx = isShort ? w * 0.5 : w * 0.69;
  const ty = isShort ? h * 0.68 : h * 0.44;
  const pop = springOut((u - 0.16) / 0.34);

  ctx.save();
  ctx.translate(tx, ty);
  ctx.scale(pop, pop);
  ctx.textAlign = "center";
  ctx.fillStyle = INK;
  ctx.font = `900 ${isShort ? 46 : 48}px ${SERIF}`;
  ctx.fillText(sc.visual_label || "SK hynix", 0, -90);

  ctx.fillStyle = "#1D4ED8";
  ctx.font = `900 ${isShort ? 108 : 118}px ${SANS}`;
  ctx.fillText(sc.big_stat || "53%", 0, 22);

  if (sc.secondary_label) {
    ctx.fillStyle = "#475569";
    ctx.font = `800 26px ${SANS}`;
    ctx.fillText(sc.secondary_label, 0, 86);
  }
  ctx.restore();

  drawSubtleSourceFootnote(ctx, sc, w, h, u, false);
}

// 11. `pedestal_rivalry`: Exact `first32s_amd_gXhr` row 3 rising/shrinking stone pedestals + pinned chips!
function tplPedestalRivalry(ctx, A, sc, w, h, t, dur, amp, isShort) {
  const u = t / dur;
  ctx.fillStyle = ICE;
  ctx.fillRect(0, 0, w, h);

  const floorY = isShort ? h * 0.78 : h * 0.84;
  ctx.fillStyle = "#E2E8F0";
  ctx.fillRect(0, floorY, w, h - floorY);
  ctx.strokeStyle = "#CBD5E1";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(0, floorY);
  ctx.lineTo(w, floorY);
  ctx.stroke();

  const shift = easeInOutCubic((u - 0.14) / 0.52);
  const p1H = lerp(isShort ? 320 : 300, isShort ? 140 : 130, shift);
  const p2H = lerp(isShort ? 110 : 100, isShort ? 350 : 330, shift);
  const c1Size = lerp(180, 125, shift);
  const c2Size = lerp(115, 205, shift);

  const x1 = isShort ? w * 0.28 : w * 0.24;
  const x2 = isShort ? w * 0.68 : w * 0.52;
  const pedW = isShort ? 210 : 240;

  // Pedestal 1
  fillStrokeBox(ctx, x1 - pedW / 2, floorY - p1H, pedW, p1H, 6, "#E5E7EB", INK, 3.2);
  drawPinnedChip(ctx, x1, floorY - p1H - c1Size * 0.58, c1Size, "#1D4ED8", sc.visual_label || "RIVAL");

  // Pedestal 2
  fillStrokeBox(ctx, x2 - pedW / 2, floorY - p2H, pedW, p2H, 6, "#E5E7EB", INK, 3.2);
  drawPinnedChip(ctx, x2, floorY - p2H - c2Size * 0.58, c2Size, "#DC2626", sc.secondary_label || "SK hynix");

  if (!isShort) {
    drawStandingActor(ctx, A.leftActor || A.fallbackActor, w * 0.82, floorY + 12, 330, 145, 0, amp, t, true);
  }
  if (sc.big_stat) {
    drawCleanSignplate(ctx, isShort ? w * 0.5 : w * 0.38, isShort ? h * 0.14 : h * 0.12, sc.big_stat, "#0F172A", 1.05);
  }
  drawSubtleSourceFootnote(ctx, sc, w, h, u, false);
}

// 12. `red_cross_stamp`: Exact `first32s_sony_2d88` row 3 midnight navy + bold crimson X cross-out!
function tplRedCrossStamp(ctx, A, sc, w, h, t, dur, amp, isShort) {
  const u = t / dur;
  ctx.fillStyle = MIDNIGHT;
  ctx.fillRect(0, 0, w, h);

  const cx1 = isShort ? w * 0.5 : w * 0.28;
  const cy1 = isShort ? h * 0.28 : h * 0.52;
  const cx2 = isShort ? w * 0.5 : w * 0.68;
  const cy2 = isShort ? h * 0.64 : h * 0.52;

  // Connecting thin cyan trace
  ctx.strokeStyle = "#38BDF8";
  ctx.lineWidth = 3.5;
  ctx.beginPath();
  ctx.moveTo(cx1, cy1);
  ctx.lineTo(cx2, cy2);
  ctx.stroke();

  drawPinnedChip(ctx, cx1, cy1, isShort ? 190 : 210, "#0EA5E9", sc.secondary_label || "CORE");
  drawPinnedChip(ctx, cx2, cy2, isShort ? 240 : 265, "#15803D", sc.visual_label || "BLOCKED", "#FFFFFF", sc.big_stat || "");

  // Bold crimson X stamp slamming onto second module at u=0.50!
  if (u >= 0.48) {
    const sp = springOut((u - 0.48) / 0.22);
    const arm = (isShort ? 135 : 150) * sp;
    ctx.save();
    ctx.translate(cx2, cy2);
    ctx.strokeStyle = "#EF4444";
    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(-arm, -arm);
    ctx.lineTo(arm, arm);
    ctx.moveTo(arm, -arm);
    ctx.lineTo(-arm, arm);
    ctx.stroke();
    ctx.restore();
  }
  drawSubtleSourceFootnote(ctx, sc, w, h, u, true);
}

// 13. `workbench_hbm_stack`: Warm oak desk + cream mat + corner screwdriver + 12-layer DRAM & TSV drilling!
function tplWorkbenchHbmStack(ctx, A, sc, w, h, t, dur, amp, isShort) {
  const u = t / dur;
  // Warm light-oak wood grain surface (`overview_sony_2d88` exact desk!)
  ctx.fillStyle = "#E4BA82";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(161, 108, 48, 0.18)";
  ctx.lineWidth = 3;
  for (let y = 40; y < h; y += 72) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Screwdriver & brass screws on wood desk corners
  ctx.save();
  ctx.translate(70, 95);
  ctx.rotate(-0.65);
  fillStrokeBox(ctx, -90, -16, 120, 32, 12, "#2563EB", INK, 3.2);
  fillStrokeBox(ctx, 30, -5, 95, 10, 3, "#CBD5E1", INK, 2.5);
  ctx.restore();

  // Cream silicone work mat in center
  const matW = isShort ? w * 0.86 : 1120;
  const matH = isShort ? 780 : 690;
  const mx = (w - matW) * 0.5;
  const my = (h - matH) * 0.5;
  fillStrokeBox(ctx, mx, my, matW, matH, 26, "#FAF6EB", INK, 4);

  const cx = w * 0.5;
  const baseY = my + matH * 0.72;
  const scf = isShort ? 0.90 : 1.08;

  // 2.5D Silicon Interposer
  fillStrokeBox(ctx, cx - 320 * scf, baseY, 640 * scf, 54 * scf, 10, "#CBD5E1", INK, 3.4);
  ctx.fillStyle = "#1E293B";
  ctx.font = `800 ${Math.round(19 * scf)}px ${MONO}`;
  ctx.textAlign = "center";
  ctx.fillText("SILICON INTERPOSER", cx, baseY + 34 * scf);

  // Right: Green AI GPU Die
  const gpuX = cx + 155 * scf;
  fillStrokeBox(ctx, gpuX - 118 * scf, baseY - 68 * scf, 236 * scf, 64 * scf, 10, "#10B981", INK, 3.4);
  ctx.fillStyle = "#064E3B";
  ctx.font = `900 ${Math.round(21 * scf)}px ${MONO}`;
  ctx.fillText(sc.secondary_label || "AI GPU DIE", gpuX, baseY - 29 * scf);

  // Left: 12 DRAM layers dropping in with spring + golden TSV vertical shafts!
  const stackX = cx - 145 * scf;
  const layers = 12;
  const stackU = clamp(u / 0.55, 0, 1);
  for (let l = 0; l < layers; l++) {
    const lp = clamp(stackU * layers - l, 0, 1);
    if (lp <= 0.01) continue;
    const ly = lerp(my + 60, baseY - 16 * scf - l * 18 * scf, springOut(lp));
    fillStrokeBox(ctx, stackX - 105 * scf, ly, 210 * scf, 15 * scf, 4, l % 2 === 0 ? "#1D4ED8" : "#3B82F6", INK, 2.4);
  }

  // TSV golden vertical shafts (`u >= 0.48`)
  if (u >= 0.46) {
    const dp = easeOutCubic((u - 0.46) / 0.25);
    const topY = baseY - 16 * scf - (layers - 1) * 18 * scf;
    const botY = baseY + 24 * scf;
    for (const off of [-66, -33, 0, 33, 66]) {
      const vx = stackX + off * scf;
      ctx.strokeStyle = "#FBBF24";
      ctx.lineWidth = 4.2 * scf;
      ctx.beginPath();
      ctx.moveTo(vx, topY);
      ctx.lineTo(vx, lerp(topY, botY, dp));
      ctx.stroke();
    }
  }

  ctx.fillStyle = INK;
  ctx.font = `900 ${Math.round(32 * scf)}px ${SANS}`;
  ctx.textAlign = "center";
  ctx.fillText(sc.big_stat ? `${sc.big_stat} · ${sc.visual_label}` : (sc.visual_label || "3D HBM STACK"), cx, my + 68);
  drawSubtleSourceFootnote(ctx, sc, w, h, u, false);
}

// 14. `thermal_mrmuf_vs_tcncf`: Split physics diagram — TC-NCF trapped heat vs MR-MUF liquid epoxy flow!
function tplThermalMrmufVsTcncf(ctx, A, sc, w, h, t, dur, amp, isShort) {
  const u = t / dur;
  ctx.fillStyle = CREAM;
  ctx.fillRect(0, 0, w, h);

  const panels = isShort
    ? [{ cx: w * 0.5, cy: h * 0.31, hot: true }, { cx: w * 0.5, cy: h * 0.72, hot: false }]
    : [{ cx: w * 0.28, cy: h * 0.54, hot: true }, { cx: w * 0.72, cy: h * 0.54, hot: false }];

  panels.forEach((p) => {
    const pw = isShort ? 760 : 680;
    const ph = isShort ? 560 : 660;
    fillStrokeBox(ctx, p.cx - pw / 2, p.cy - ph / 2, pw, ph, 24, "#FFFFFF", INK, 3.8);

    // 8 stacked DRAM dies with gap channels
    const baseY = p.cy + ph * 0.30;
    for (let l = 0; l < 8; l++) {
      const dy = baseY - l * 46;
      if (!p.hot) {
        // Liquid cyan MR-MUF resin smoothly filling gaps from bottom up!
        const fillP = clamp((u * 1.4) - l * 0.09, 0, 1);
        ctx.fillStyle = `rgba(14, 165, 233, ${(0.82 * fillP).toFixed(2)})`;
        ctx.fillRect(p.cx - 210, dy + 16, 420 * fillP, 30);
      } else {
        // Solid grey TC-NCF film + glowing red hot voids!
        ctx.fillStyle = "#94A3B8";
        ctx.fillRect(p.cx - 200, dy + 20, 400, 18);
        const pulse = 0.55 + 0.45 * Math.sin(t * 8.0 + l);
        ctx.fillStyle = `rgba(239, 68, 68, ${pulse.toFixed(2)})`;
        ctx.beginPath();
        ctx.arc(p.cx + ((l % 3) - 1) * 85, dy + 29, 10, 0, Math.PI * 2);
        ctx.fill();
      }
      fillStrokeBox(ctx, p.cx - 195, dy, 390, 20, 4, p.hot ? "#DC2626" : "#2563EB", INK, 2.6);
    }

    ctx.fillStyle = p.hot ? "#B91C1C" : "#0369A1";
    ctx.font = `900 30px ${SANS}`;
    ctx.textAlign = "center";
    ctx.fillText(
      p.hot ? (sc.visual_label || "TC-NCF SOLID FILM") : (sc.secondary_label || "MR-MUF LIQUID EPOXY"),
      p.cx,
      p.cy - ph * 0.37
    );
  });

  drawSubtleSourceFootnote(ctx, sc, w, h, u, false);
}

// 15. `rain_pipe_analogy`: Narrow 64-bit trickle vs 1,024-bit parallel aqueduct filling glass reservoirs!
function tplRainPipeAnalogy(ctx, A, sc, w, h, t, dur, amp, isShort) {
  const u = t / dur;
  ctx.fillStyle = "#E0F2FE";
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = "#86EFAC";
  ctx.fillRect(0, h * 0.82, w, h * 0.18);

  const leftX = isShort ? w * 0.27 : w * 0.27;
  const rightX = isShort ? w * 0.73 : w * 0.73;
  const topPipeY = isShort ? h * 0.20 : h * 0.18;
  const tankBotY = h * 0.80;
  const tankW = isShort ? 340 : 460;
  const tankH = isShort ? 520 : 450;

  // Left: Tiny single nozzle dripping slowly (starved!)
  fillStrokeBox(ctx, leftX - 90, topPipeY, 180, 26, 8, "#475569", INK, 3.2);
  const dropY = topPipeY + 35 + ((t * 210) % (tankH - 40));
  ctx.fillStyle = "#0284C7";
  ctx.beginPath();
  ctx.arc(leftX, dropY, 8, 0, Math.PI * 2);
  ctx.fill();

  fillStrokeBox(ctx, leftX - tankW / 2, tankBotY - tankH, tankW, tankH, 18, "rgba(255,255,255,0.65)", INK, 3.8);
  const lowH = tankH * lerp(0.06, 0.12, u);
  fillStrokeBox(ctx, leftX - tankW / 2 + 6, tankBotY - lowH, tankW - 12, lowH - 4, 12, "#F97316", null, 0);

  // Right: Wide 8-stream parallel aqueduct rapidly filling reservoir!
  fillStrokeBox(ctx, rightX - tankW / 2, topPipeY, tankW, 34, 10, "#1D4ED8", INK, 3.5);
  const highH = tankH * lerp(0.18, 0.86, easeOutCubic(u));
  for (let s = 0; s < 7; s++) {
    const sx = lerp(rightX - tankW * 0.36, rightX + tankW * 0.36, s / 6);
    ctx.strokeStyle = "#38BDF8";
    ctx.lineWidth = 8;
    ctx.setLineDash([18, 12]);
    ctx.lineDashOffset = -t * 260;
    ctx.beginPath();
    ctx.moveTo(sx, topPipeY + 34);
    ctx.lineTo(sx, tankBotY - highH + 10);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  fillStrokeBox(ctx, rightX - tankW / 2, tankBotY - tankH, tankW, tankH, 18, "rgba(255,255,255,0.65)", INK, 3.8);
  fillStrokeBox(ctx, rightX - tankW / 2 + 6, tankBotY - highH, tankW - 12, highH - 4, 12, "#0EA5E9", null, 0);

  drawCleanSignplate(ctx, leftX, topPipeY - 48, sc.visual_label || "64-BIT BUS", "#B91C1C", isShort ? 0.86 : 0.95);
  drawCleanSignplate(ctx, rightX, topPipeY - 48, sc.secondary_label || "1,024-BIT HBM", "#1D4ED8", isShort ? 0.86 : 0.95);
  drawSubtleSourceFootnote(ctx, sc, w, h, u, false);
}

// 16. `cowos_silicon_bridge`: Midnight top-down TSMC CoWoS interposer + central GPU + 8 HBM3E stacks!
function tplCowosSiliconBridge(ctx, A, sc, w, h, t, dur, amp, isShort) {
  const u = t / dur;
  ctx.fillStyle = MIDNIGHT;
  ctx.fillRect(0, 0, w, h);

  const cx = w * 0.5;
  const cy = h * 0.53;
  const scf = isShort ? 0.82 : 1.05;

  // Silver Silicon Interposer package substrate
  const boardW = 860 * scf;
  const boardH = 620 * scf;
  fillStrokeBox(ctx, cx - boardW / 2, cy - boardH / 2, boardW, boardH, 28 * scf, "#334155", "#94A3B8", 4);

  // Central NVIDIA GPU Die
  const gpuW = 290 * scf;
  const gpuH = 380 * scf;
  fillStrokeBox(ctx, cx - gpuW / 2, cy - gpuH / 2, gpuW, gpuH, 16 * scf, "#10B981", INK, 4);
  ctx.fillStyle = "#064E3B";
  ctx.font = `900 ${Math.round(28 * scf)}px ${SANS}`;
  ctx.textAlign = "center";
  ctx.fillText(sc.secondary_label || "AI GPU", cx, cy + 8);

  // 8 surrounding HBM3E cubes (4 left, 4 right) + streaming cyan micron data pulses!
  for (const side of [-1, 1]) {
    for (let i = 0; i < 4; i++) {
      const hx = cx + side * 275 * scf;
      const hy = cy + (i - 1.5) * 112 * scf;

      // High-density silicon trace lines + animated pulse dot
      ctx.strokeStyle = "#38BDF8";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(hx, hy);
      ctx.lineTo(cx + side * (gpuW / 2), hy);
      ctx.stroke();

      const p = (t * 2.6 + i * 0.23 + (side > 0 ? 0.5 : 0)) % 1.0;
      const px = lerp(hx, cx + side * (gpuW / 2), p);
      ctx.fillStyle = "#FDE047";
      ctx.beginPath();
      ctx.arc(px, hy, 6 * scf, 0, Math.PI * 2);
      ctx.fill();

      fillStrokeBox(ctx, hx - 68 * scf, hy - 40 * scf, 136 * scf, 80 * scf, 10 * scf, "#2563EB", INK, 3.2);
      ctx.fillStyle = "#FFFFFF";
      ctx.font = `800 ${Math.round(18 * scf)}px ${MONO}`;
      ctx.fillText("HBM3E", hx, hy + 6);
    }
  }

  ctx.fillStyle = "#F8FAFC";
  ctx.font = `900 ${isShort ? 38 : 42}px ${SANS}`;
  ctx.textAlign = "center";
  ctx.fillText(sc.big_stat ? `${sc.big_stat} · ${sc.visual_label}` : (sc.visual_label || "TSMC CoWoS 2.5D"), cx, cy - boardH / 2 - 42);
  drawSubtleSourceFootnote(ctx, sc, w, h, u, true);
}

// 17. `wafer_3x_cannibalization`: Side-by-side 300mm silicon wafers showing 3x HBM die-area hunger!
function tplWafer3xCannibalization(ctx, A, sc, w, h, t, dur, amp, isShort) {
  const u = t / dur;
  ctx.fillStyle = CREAM;
  ctx.fillRect(0, 0, w, h);

  const wafers = isShort
    ? [{ cx: w * 0.5, cy: h * 0.30, step: 26, col: "#60A5FA", title: sc.secondary_label || "STANDARD DDR5" },
       { cx: w * 0.5, cy: h * 0.71, step: 48, col: "#F59E0B", title: sc.visual_label || "HBM (3x AREA / GB)" }]
    : [{ cx: w * 0.29, cy: h * 0.54, step: 28, col: "#60A5FA", title: sc.secondary_label || "STANDARD DDR5" },
       { cx: w * 0.71, cy: h * 0.54, step: 52, col: "#F59E0B", title: sc.visual_label || "HBM (3x AREA / GB)" }];

  const rad = isShort ? 225 : 255;
  wafers.forEach((wf, idx) => {
    const pop = springOut((u - idx * 0.14) / 0.36);
    ctx.save();
    ctx.translate(wf.cx, wf.cy);
    ctx.scale(pop, pop);

    ctx.beginPath();
    ctx.arc(0, 0, rad, 0, Math.PI * 2);
    ctx.fillStyle = "#E2E8F0";
    ctx.fill();
    ctx.clip();

    for (let x = -rad; x < rad; x += wf.step) {
      for (let y = -rad; y < rad; y += wf.step) {
        if (Math.hypot(x + wf.step / 2, y + wf.step / 2) < rad - 8) {
          fillStrokeBox(ctx, x + 2, y + 2, wf.step - 4, wf.step - 4, 3, wf.col, "#1E293B", 1.5);
        }
      }
    }
    ctx.restore();

    ctx.save();
    ctx.translate(wf.cx, wf.cy);
    ctx.scale(pop, pop);
    ctx.beginPath();
    ctx.arc(0, 0, rad, 0, Math.PI * 2);
    ctx.lineWidth = 5;
    ctx.strokeStyle = INK;
    ctx.stroke();

    ctx.fillStyle = INK;
    ctx.font = `900 28px ${SANS}`;
    ctx.textAlign = "center";
    ctx.fillText(wf.title, 0, -rad - 28);
    ctx.restore();
  });

  if (sc.big_stat) {
    drawCleanSignplate(ctx, w * 0.5, isShort ? h * 0.505 : h * 0.12, sc.big_stat, "#DC2626", 1.12);
  }
  drawSubtleSourceFootnote(ctx, sc, w, h, u, false);
}

// 18. `timeline_paradox_arc`: Clean cream milestone track (`2013 -> 2015 -> 2019 -> 2023 -> 2026`)
function tplTimelineParadoxArc(ctx, A, sc, w, h, t, dur, amp, isShort) {
  const u = t / dur;
  ctx.fillStyle = CREAM;
  ctx.fillRect(0, 0, w, h);

  const nodes = [
    { yr: "2013", tag: "1st HBM", col: "#2563EB" },
    { yr: "2015", tag: "AMD Fiji", col: "#3B82F6" },
    { yr: "2019", tag: "Rival Quits", col: "#EF4444" },
    { yr: "2023", tag: "AI Explosion", col: "#F59E0B" },
    { yr: "2026", tag: "HBM3E / HBM4", col: "#10B981" },
  ];

  const prog = easeOutCubic(u / 0.72);
  if (isShort) {
    const x = w * 0.32;
    const y1 = h * 0.18;
    const y2 = h * 0.82;
    ctx.strokeStyle = "#CBD5E1";
    ctx.lineWidth = 8;
    ctx.beginPath();
    ctx.moveTo(x, y1);
    ctx.lineTo(x, y2);
    ctx.stroke();

    ctx.strokeStyle = INK;
    ctx.beginPath();
    ctx.moveTo(x, y1);
    ctx.lineTo(x, lerp(y1, y2, prog));
    ctx.stroke();

    nodes.forEach((nd, i) => {
      const ny = lerp(y1, y2, i / (nodes.length - 1));
      const np = springOut((prog * nodes.length - i));
      if (np <= 0.02) return;
      ctx.save();
      ctx.translate(x, ny);
      ctx.scale(np, np);
      ctx.fillStyle = nd.col;
      ctx.beginPath();
      ctx.arc(0, 0, 22, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 4;
      ctx.strokeStyle = INK;
      ctx.stroke();

      fillStrokeBox(ctx, 46, -38, 420, 76, 14, "#FFFFFF", INK, 3.2);
      ctx.fillStyle = INK;
      ctx.font = `900 28px ${SANS}`;
      ctx.textAlign = "left";
      ctx.fillText(`${nd.yr} · ${nd.tag}`, 68, 10);
      ctx.restore();
    });
  } else {
    const y = h * 0.55;
    const x1 = w * 0.13;
    const x2 = w * 0.87;
    ctx.strokeStyle = "#CBD5E1";
    ctx.lineWidth = 8;
    ctx.beginPath();
    ctx.moveTo(x1, y);
    ctx.lineTo(x2, y);
    ctx.stroke();

    ctx.strokeStyle = INK;
    ctx.beginPath();
    ctx.moveTo(x1, y);
    ctx.lineTo(lerp(x1, x2, prog), y);
    ctx.stroke();

    nodes.forEach((nd, i) => {
      const nx = lerp(x1, x2, i / (nodes.length - 1));
      const np = springOut((prog * nodes.length - i));
      if (np <= 0.02) return;
      const up = i % 2 === 0;
      ctx.save();
      ctx.translate(nx, y);
      ctx.scale(np, np);
      ctx.fillStyle = nd.col;
      ctx.beginPath();
      ctx.arc(0, 0, 22, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 4;
      ctx.strokeStyle = INK;
      ctx.stroke();

      const cardY = up ? -145 : 55;
      fillStrokeBox(ctx, -125, cardY, 250, 90, 14, "#FFFFFF", INK, 3.2);
      ctx.fillStyle = nd.col;
      ctx.font = `900 28px ${SANS}`;
      ctx.textAlign = "center";
      ctx.fillText(nd.yr, 0, cardY + 38);
      ctx.fillStyle = "#334155";
      ctx.font = `800 20px ${SANS}`;
      ctx.fillText(nd.tag, 0, cardY + 68);
      ctx.restore();
    });
  }

  if (sc.big_stat) {
    drawCleanSignplate(ctx, w * 0.5, isShort ? h * 0.09 : h * 0.14, `${sc.big_stat} · ${sc.visual_label || ""}`.replace(/ · $/, ""), "#0F172A", 1.08);
  }
  drawSubtleSourceFootnote(ctx, sc, w, h, u, false);
}

const DISPATCH = {
  exterior_hq_walk: tplExteriorHqWalk,
  desk_deal_transfer: tplDeskDealTransfer,
  copier_blueprint_action: tplCopierBlueprintAction,
  hydraulic_press_crush: tplHydraulicPressCrush,
  studio_shutter_flash: tplStudioShutterFlash,
  split_screen_rivalry: tplSplitScreenRivalry,
  cleanroom_microscope_zoom: tplCleanroomMicroscopeZoom,
  server_hall_waiting: tplServerHallWaiting,
  oven_conveyor_bottleneck: tplOvenConveyorBottleneck,
  waffle_100_dots: tplWaffle100Dots,
  pedestal_rivalry: tplPedestalRivalry,
  red_cross_stamp: tplRedCrossStamp,
  workbench_hbm_stack: tplWorkbenchHbmStack,
  thermal_mrmuf_vs_tcncf: tplThermalMrmufVsTcncf,
  rain_pipe_analogy: tplRainPipeAnalogy,
  cowos_silicon_bridge: tplCowosSiliconBridge,
  wafer_3x_cannibalization: tplWafer3xCannibalization,
  timeline_paradox_arc: tplTimelineParadoxArc,
};

async function safeLoad(filePath) {
  try {
    if (filePath && fs.existsSync(filePath)) {
      return await loadImage(filePath);
    }
  } catch (_) {}
  return null;
}

async function main() {
  const jobJsonPath = process.argv[2];
  const job = JSON.parse(fs.readFileSync(jobJsonPath, "utf-8"));
  const { scene, wavPath, visualDir, outMp4, w, h, fps, isShort, envelope } = job;

  const bgBase = scene.bg_key || "set_sunny_office";
  const bgCandidate = isShort
    ? path.join(visualDir, `${bgBase}_9x16.png`)
    : path.join(visualDir, `${bgBase}.png`);
  const bgFallback = path.join(visualDir, `${bgBase}.png`);

  const A = {
    bg: (await safeLoad(bgCandidate)) || (await safeLoad(bgFallback)),
    leftActor: scene.left_actor && scene.left_actor !== "none" ? await safeLoad(path.join(visualDir, `${scene.left_actor}.png`)) : null,
    rightActor: scene.right_actor && scene.right_actor !== "none" ? await safeLoad(path.join(visualDir, `${scene.right_actor}.png`)) : null,
    fallbackActor: await safeLoad(path.join(visualDir, "char_sk_eng_talk.png")),
    altActor: await safeLoad(path.join(visualDir, "char_jensen_talk.png")),
    prop: scene.prop_key && scene.prop_key !== "none" ? await safeLoad(path.join(visualDir, `${scene.prop_key}.png`)) : null,
  };

  const totalFrames = Math.max(1, envelope.length);
  const dur = totalFrames / fps;
  const tplName = scene.template || scene.animation_type || "exterior_hq_walk";
  const fn = DISPATCH[tplName] || tplExteriorHqWalk;

  const canvas = createCanvas(w, h);
  const ctx = canvas.getContext("2d");

  const ff = spawn(
    "ffmpeg",
    [
      "-y",
      "-loglevel", "error",
      "-f", "rawvideo",
      "-pix_fmt", "rgba",
      "-s", `${w}x${h}`,
      "-r", String(fps),
      "-i", "pipe:0",
      "-i", wavPath,
      "-c:v", "libx264",
      "-preset", "veryfast",
      "-crf", "18",
      "-pix_fmt", "yuv420p",
      "-c:a", "aac",
      "-b:a", "192k",
      "-shortest",
      outMp4,
    ],
    { stdio: ["pipe", "inherit", "inherit"] }
  );

  for (let f = 0; f < totalFrames; f++) {
    const t = f / fps;
    const amp = envelope[f] || 0;
    ctx.clearRect(0, 0, w, h);
    fn(ctx, A, scene, w, h, t, Math.max(0.4, dur), amp, isShort);

    // Subtle 4-frame fade-in on chapter openings only (crisp hard cuts everywhere else, just like @TradeoffLayer!)
    if (scene.chapter_title && f < 8) {
      ctx.fillStyle = `rgba(15,23,42,${((8 - f) / 8 * 0.35).toFixed(3)})`;
      ctx.fillRect(0, 0, w, h);
    }

    const buf = canvas.data();
    if (!ff.stdin.write(buf)) {
      await new Promise((res) => ff.stdin.once("drain", res));
    }
  }

  ff.stdin.end();
  await new Promise((resolve, reject) => {
    ff.on("close", (code) => (code === 0 ? resolve() : reject(new Error(`FFmpeg exited with ${code}`))));
    ff.on("error", reject);
  });
}

main().catch((err) => {
  console.error("[skia_animator v3 fatal]", err);
  process.exit(1);
});
