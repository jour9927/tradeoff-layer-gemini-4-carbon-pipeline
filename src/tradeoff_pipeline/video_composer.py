#!/usr/bin/env python3
"""
Tradeoff Layer — 60fps Long-Form (`1920x1080`) & Vertical Shorts (`1080x1920`) Video Compositor.

Faithfully replicates @TradeoffLayer's three visual modes:
  1. `dark_metric_card`        — Matte #0B0B0E canvas, #1A1A1F cards, animated Bezier connector wires,
                                 green/red/blue token & cost meters (`0vZCvW7IzmA` / `bmFUEo6A_1U`).
  2. `vector_whiteboard_scene` — AI cel-shaded architectural background + rounded multi-bar whiteboard
                                 diagram + @TradeoffLayer's signature expressive 2D vector characters
                                 with articulated black stick-figure arms & legs (`yB9SC4HbH5o` / `MnOOAvY_2NQ`).
  3. `editorial_illustration`  — Full-bleed AI flat cel-shaded metaphor scene with smooth 60fps Ken Burns
                                 motion, floating editorial metric cards, and chapter lower-thirds (`2d883UK7p9E`).

Includes:
  - Live on-screen Source Citation Badges (`[S1] Publisher • URL`) for every scraped public fact.
  - Progressive word-by-word highlighted Shorts captions (`white` spoken + `muted grey` upcoming)
    synced to `faster-whisper` word timestamps (`bmFUEo6A_1U`).
  - Parallel scene rendering across multi-core remote workers (RTX PRO 6000 48-vCPU / b850).
  - Strict hardware guard preventing heavy video rendering on local `alan-hub`.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import socket
import subprocess
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


TONE_COLORS = {
    "green": (74, 222, 128),
    "red": (248, 113, 113),
    "blue": (96, 165, 250),
    "amber": (251, 191, 36),
    "neutral": (228, 228, 235),
}


def _guard_not_alan_hub(allow_override: bool = False) -> None:
    host = socket.gethostname().lower()
    if ("alan-hub" in host or "alan_hub" in host) and not allow_override:
        raise RuntimeError(
            "Policy guard: Video rendering on local 'alan-hub' is strictly disabled! "
            "Dispatch rendering to the RTX PRO 6000 Colab worker or b850 node."
        )


def _find_font(bold: bool = False, mono: bool = False, size: int = 36) -> ImageFont.FreeTypeFont:
    if mono:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/System/Library/Fonts/Menlo.ttc",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _ease_out_cubic(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 1.0 - (1.0 - x) ** 3


def _ease_in_out_quad(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 2 * x * x if x < 0.5 else 1 - ((-2 * x + 2) ** 2) / 2


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    cur = words[0]
    for w in words[1:]:
        cand = f"{cur} {w}"
        bbox = draw.textbbox((0, 0), cand, font=font)
        if (bbox[2] - bbox[0]) <= max_w:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _draw_bezier(
    draw: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    progress: float,
    color: tuple[int, int, int],
    width: int = 4,
    steps: int = 36,
) -> None:
    if progress <= 0.01:
        return
    max_step = max(2, int(steps * min(1.0, progress)))
    pts = []
    for i in range(max_step + 1):
        u = i / steps
        om = 1.0 - u
        x = (om**3) * p0[0] + 3 * (om**2) * u * p1[0] + 3 * om * (u**2) * p2[0] + (u**3) * p3[0]
        y = (om**3) * p0[1] + 3 * (om**2) * u * p1[1] + 3 * om * (u**2) * p2[1] + (u**3) * p3[1]
        pts.append((x, y))
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=width, joint="curve")


def _smart_clip(text: str, max_len: int) -> str:
    s = str(text).strip()
    if len(s) <= max_len:
        return s
    cut = s[: max_len - 1].rsplit(" ", 1)[0]
    return (cut if len(cut) >= max_len // 2 else s[: max_len - 1]) + "…"


def _draw_tradeoff_character(
    img: Image.Image,
    cx: int,
    foot_y: int,
    scale: float,
    role: str,
    accent_hex: str,
    t: float,
    speaking_amp: float,
    point_dir: int = -1,
    bust_rgba: Image.Image | None = None,
) -> None:
    """
    Draws @TradeoffLayer's signature 2D vector character:
    expressive cel-shaded head & jacket/hoodie torso with crisp dark ink outlines
    + minimalist articulated black stick-figure arms and legs (`yB9SC4HbH5o` style).
    """
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        hx = accent_hex.lstrip("#")
        accent = (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16), 255)
    except Exception:
        accent = (59, 130, 246, 255)

    ink = (18, 22, 31, 255)
    bob = math.sin(t * 2.6) * 3.0 * scale

    hip_y = int(foot_y - 135 * scale)
    torso_top_y = int(hip_y - 125 * scale + bob)
    head_cy = int(torso_top_y - 58 * scale)

    # Stick legs + half-circle shoes (exact @TradeoffLayer silhouette!)
    leg_dx = int(24 * scale)
    lw = max(3, int(4.5 * scale))
    for side in (-1, 1):
        lx = cx + side * leg_dx
        knee_x = lx + side * int(4 * scale)
        knee_y = int((hip_y + foot_y) * 0.52)
        draw.line([(lx, hip_y), (knee_x, knee_y), (lx, foot_y)], fill=ink, width=lw)
        shoe_w, shoe_h = int(16 * scale), int(9 * scale)
        draw.pieslice(
            [lx - shoe_w, foot_y - shoe_h, lx + shoe_w, foot_y + shoe_h],
            start=180,
            end=360,
            fill=ink,
        )

    # Stick arms (one gesturing toward whiteboard, one resting on hip/side)
    shoulder_y = int(torso_top_y + 28 * scale)
    arm_wave = math.sin(t * 3.8) * 10.0 * scale
    p_sh_x = cx + point_dir * int(46 * scale)
    elbow_x = p_sh_x + point_dir * int(34 * scale)
    elbow_y = int(shoulder_y + 34 * scale)
    hand_x = elbow_x + point_dir * int(42 * scale)
    hand_y = int(shoulder_y - 12 * scale + arm_wave)
    draw.line([(p_sh_x, shoulder_y), (elbow_x, elbow_y), (hand_x, hand_y)], fill=ink, width=lw)
    # Little 3-finger stick hand
    for ang in (-0.35, 0.0, 0.35):
        fx = hand_x + int(math.cos(ang) * point_dir * 10 * scale)
        fy = hand_y + int(math.sin(ang) * 10 * scale)
        draw.line([(hand_x, hand_y), (fx, fy)], fill=ink, width=max(2, lw - 1))

    o_sh_x = cx - point_dir * int(46 * scale)
    o_elb_x = o_sh_x - point_dir * int(26 * scale)
    o_elb_y = int(shoulder_y + 44 * scale)
    o_hand_x = cx - point_dir * int(34 * scale)
    o_hand_y = int(hip_y - 8 * scale)
    draw.line([(o_sh_x, shoulder_y), (o_elb_x, o_elb_y), (o_hand_x, o_hand_y)], fill=ink, width=lw)

    # If AI-generated cel-shaded character bust cutout is available, composite it atop stick limbs!
    if bust_rgba is not None:
        target_h = int(235 * scale * (1.0 + 0.018 * speaking_amp))
        aspect = bust_rgba.width / max(1, bust_rgba.height)
        target_w = max(40, int(target_h * aspect))
        bust_resized = bust_rgba.resize((target_w, target_h), Image.Resampling.LANCZOS)
        bx = cx - target_w // 2
        by = hip_y - target_h + int(8 * scale + bob)
        img.alpha_composite(bust_resized, dest=(max(0, bx), max(0, by)))
        return

    # Cel-shaded upper body jacket/hoodie
    tw = int(52 * scale)
    jacket_col = (36, 41, 51, 255) if role == "ceo" else (196, 205, 218, 255) if role == "engineer" else accent
    shirt_col = (30, 34, 44, 255) if role == "engineer" else (245, 247, 250, 255)
    torso_poly = [
        (cx - tw, hip_y),
        (cx - int(tw * 0.88), torso_top_y),
        (cx + int(tw * 0.88), torso_top_y),
        (cx + tw, hip_y),
    ]
    draw.polygon(torso_poly, fill=jacket_col, outline=ink, width=max(2, int(3.5 * scale)))
    # Inner tee
    inner_w = int(20 * scale)
    draw.polygon(
        [
            (cx - inner_w, hip_y),
            (cx - int(inner_w * 0.8), torso_top_y + int(6 * scale)),
            (cx + int(inner_w * 0.8), torso_top_y + int(6 * scale)),
            (cx + inner_w, hip_y),
        ],
        fill=shirt_col,
        outline=ink,
        width=max(2, int(2.5 * scale)),
    )

    # Neck + Cel-shaded Head
    skin = (246, 205, 172, 255)
    nw = int(13 * scale)
    draw.rectangle([cx - nw, torso_top_y - int(14 * scale), cx + nw, torso_top_y + int(8 * scale)], fill=skin, outline=ink, width=max(2, int(2.5 * scale)))

    hr_x, hr_y = int(42 * scale), int(48 * scale)
    draw.ellipse([cx - hr_x, head_cy - hr_y, cx + hr_x, head_cy + hr_y], fill=skin, outline=ink, width=max(2, int(3.5 * scale)))

    # Hair style by role
    hair_col = (210, 214, 222, 255) if role == "ceo" else (78, 52, 36, 255) if role == "engineer" else (42, 46, 58, 255)
    draw.pieslice(
        [cx - hr_x - int(4 * scale), head_cy - hr_y - int(10 * scale), cx + hr_x + int(4 * scale), head_cy + int(6 * scale)],
        start=175,
        end=365,
        fill=hair_col,
        outline=ink,
        width=max(2, int(3 * scale)),
    )

    # Glasses + Eyes (blinking every ~3.7s)
    blink = (t % 3.7) > 3.56
    eye_y = head_cy - int(4 * scale)
    for ex_off in (-15, 15):
        ex = cx + int(ex_off * scale)
        gw, gh = int(12 * scale), int(9 * scale)
        draw.rounded_rectangle([ex - gw, eye_y - gh, ex + gw, eye_y + gh], radius=max(2, int(3 * scale)), outline=ink, width=max(2, int(2.6 * scale)))
        if blink:
            draw.line([(ex - int(6 * scale), eye_y), (ex + int(6 * scale), eye_y)], fill=ink, width=max(2, int(2.5 * scale)))
        else:
            er = max(2, int(3.5 * scale))
            draw.ellipse([ex - er, eye_y - er, ex + er, eye_y + er], fill=ink)
    draw.line([(cx - int(3 * scale), eye_y), (cx + int(3 * scale), eye_y)], fill=ink, width=max(2, int(2.5 * scale)))

    # Mouth synced to speech envelope!
    mouth_y = head_cy + int(20 * scale)
    mw = int(12 * scale)
    if speaking_amp > 0.18:
        mh = max(3, int(9 * scale * min(1.0, speaking_amp * 1.4)))
        draw.rounded_rectangle([cx - mw, mouth_y - mh // 2, cx + mw, mouth_y + mh], radius=max(2, int(4 * scale)), fill=(165, 58, 58, 255), outline=ink, width=max(2, int(2.2 * scale)))
    else:
        draw.arc([cx - mw, mouth_y - int(6 * scale), cx + mw, mouth_y + int(6 * scale)], start=15, end=165, fill=ink, width=max(2, int(2.5 * scale)))


def _draw_source_badge(
    img: Image.Image,
    sources: list[dict[str, Any]],
    citation_fact: str,
    w: int,
    h: int,
    t: float,
    is_short: bool = False,
) -> None:
    """Renders @TradeoffLayer live public source citation pill on screen."""
    if not sources:
        return
    enter = _ease_out_cubic((t - 0.25) / 0.45)
    if enter <= 0.02:
        return

    src = sources[0]
    draw = ImageDraw.Draw(img, "RGBA")
    f_bold = _find_font(bold=True, size=21 if not is_short else 24)
    f_reg = _find_font(bold=False, mono=True, size=18 if not is_short else 20)

    top_txt = f"SOURCE {src['badge_label']}"
    bot_txt = src.get("short_url") or _smart_clip(src.get("url", ""), 48)
    if citation_fact and not is_short:
        bot_txt = f"{bot_txt}  •  {_smart_clip(citation_fact, 62)}"

    b1 = draw.textbbox((0, 0), top_txt, font=f_bold)
    b2 = draw.textbbox((0, 0), bot_txt, font=f_reg)
    box_w = min(w - 80, max(b1[2] - b1[0], b2[2] - b2[0]) + 44)
    box_h = 66 if not is_short else 76

    if is_short:
        bx = (w - box_w) // 2
        by = int(86 - (1.0 - enter) * 28)
    else:
        bx = 56
        by = int(h - box_h - 38 + (1.0 - enter) * 24)

    alpha = int(230 * enter)
    draw.rounded_rectangle(
        [bx, by, bx + box_w, by + box_h],
        radius=14,
        fill=(16, 20, 28, alpha),
        outline=(74, 222, 128, int(190 * enter)),
        width=2,
    )
    draw.ellipse([bx + 16, by + 18, bx + 26, by + 28], fill=(74, 222, 128, alpha))
    draw.text((bx + 34, by + 10), top_txt, font=f_bold, fill=(244, 246, 250, alpha))
    draw.text((bx + 18, by + 37), bot_txt, font=f_reg, fill=(156, 168, 188, alpha))


def _render_dark_metric_scene(
    scene: dict[str, Any],
    w: int,
    h: int,
    t: float,
    dur: float,
    is_short: bool,
) -> Image.Image:
    """Renders Mode C (`0vZCvW7IzmA` / `bmFUEo6A_1U`): matte #0B0B0E dark UI card + Bezier token/flow wires."""
    img = Image.new("RGBA", (w, h), (11, 11, 14, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    # Subtle dot matrix grid
    step = 64
    for gx in range(48, w, step):
        for gy in range(48, h, step):
            draw.point((gx, gy), fill=(36, 36, 44, 255))

    enter = _ease_out_cubic(t / 0.55)
    headline = scene.get("headline") or ""
    subhead = scene.get("subheadline") or ""
    nodes = (scene.get("diagram_nodes") or [])[:5]
    metrics = (scene.get("metrics") or [])[:4]

    f_head = _find_font(bold=True, size=48 if not is_short else 52)
    f_sub = _find_font(bold=False, size=27 if not is_short else 30)
    f_mono_b = _find_font(bold=True, mono=True, size=34 if not is_short else 38)
    f_mono = _find_font(bold=False, mono=True, size=28 if not is_short else 30)

    top_y = 108 if not is_short else 210
    if is_short:
        lines = _wrap_text(draw, headline, f_head, w - 120)
        cy = top_y
        for ln in lines[:2]:
            bb = draw.textbbox((0, 0), ln, font=f_head)
            draw.text(((w - (bb[2] - bb[0])) // 2, cy), ln, font=f_head, fill=(248, 250, 252, int(255 * enter)))
            cy += 62
        if subhead:
            s_lines = _wrap_text(draw, subhead, f_sub, w - 140)
            for sln in s_lines[:2]:
                bb = draw.textbbox((0, 0), sln, font=f_sub)
                draw.text(((w - (bb[2] - bb[0])) // 2, cy + 6), sln, font=f_sub, fill=(156, 163, 175, int(240 * enter)))
                cy += 38
    else:
        draw.text((110, top_y), headline, font=f_head, fill=(248, 250, 252, int(255 * enter)))
        if subhead:
            draw.text((112, top_y + 64), subhead, font=f_sub, fill=(148, 163, 184, int(240 * enter)))

    # Central dark rounded card (`Your AI Bill` / `The meter` style!)
    card_w = int(w * 0.62) if not is_short else int(w * 0.84)
    card_h = max(290, 128 + len(metrics) * 74)
    card_x = (w - card_w) // 2
    card_y = int((h * 0.48 if not is_short else h * 0.46) - card_h // 2 + (220 if nodes else 60))
    if is_short and nodes:
        card_y = min(card_y, h - card_h - 490)

    # Node pills + animated Bezier wires into card (`0vZCvW7IzmA_t0032s.jpg`!)
    if nodes:
        node_y = card_y - (175 if not is_short else 165)
        gap = 22 if not is_short else 14
        pill_w = min(250, (card_w - gap * (len(nodes) - 1)) // max(1, len(nodes)))
        total_nw = pill_w * len(nodes) + gap * (len(nodes) - 1)
        start_nx = (w - total_nw) // 2
        for idx, nd in enumerate(nodes):
            n_prog = _ease_out_cubic((t - 0.18 - idx * 0.14) / 0.45)
            if n_prog <= 0.01:
                continue
            px = start_nx + idx * (pill_w + gap)
            py = int(node_y - (1.0 - n_prog) * 18)
            ph = 68
            draw.rounded_rectangle(
                [px, py, px + pill_w, py + ph],
                radius=12,
                fill=(28, 30, 36, int(245 * n_prog)),
                outline=(68, 74, 88, int(220 * n_prog)),
                width=2,
            )
            lbl = nd[:16]
            bb = draw.textbbox((0, 0), lbl, font=f_mono)
            draw.text(
                (px + (pill_w - (bb[2] - bb[0])) // 2, py + (ph - (bb[3] - bb[1])) // 2 - 2),
                lbl,
                font=f_mono,
                fill=(96, 185, 252, int(255 * n_prog)),
            )
            p0 = (px + pill_w / 2, py + ph)
            p3 = (card_x + card_w * (0.25 + 0.5 * (idx / max(1, len(nodes) - 1 if len(nodes) > 1 else 1))), card_y)
            p1 = (p0[0], p0[1] + 65)
            p2 = (p3[0], p3[1] - 65)
            wire_prog = _ease_out_cubic((t - 0.35 - idx * 0.14) / 0.55)
            _draw_bezier(draw, p0, p1, p2, p3, wire_prog, color=(138, 146, 162), width=3)

    draw.rounded_rectangle(
        [card_x, card_y, card_x + card_w, card_y + card_h],
        radius=24,
        fill=(26, 27, 32, int(250 * enter)),
        outline=(68, 72, 84, int(230 * enter)),
        width=2,
    )
    card_title = (scene.get("chapter_title") or headline or "System Breakdown")[:38]
    draw.text((card_x + 42, card_y + 30), card_title, font=f_mono_b, fill=(244, 244, 248, 255))
    draw.line([(card_x + 42, card_y + 88), (card_x + card_w - 42, card_y + 88)], fill=(62, 65, 76, 255), width=2)

    for m_idx, m in enumerate(metrics):
        m_prog = _ease_out_cubic((t - 0.35 - m_idx * 0.28) / 0.5)
        if m_prog <= 0.02:
            continue
        ry = card_y + 114 + m_idx * 70
        lbl = str(m.get("label", ""))[:32]
        val = str(m.get("value", ""))[:18]
        tone = TONE_COLORS.get(str(m.get("tone", "neutral")).lower(), TONE_COLORS["neutral"])
        draw.text((card_x + 44, ry), lbl, font=f_mono, fill=(212, 216, 224, int(255 * m_prog)))
        vb = draw.textbbox((0, 0), val, font=f_mono_b)
        draw.text(
            (card_x + card_w - 44 - (vb[2] - vb[0]), ry - 3),
            val,
            font=f_mono_b,
            fill=(*tone, int(255 * m_prog)),
        )

    return img


def _crop_ken_burns(bg: Image.Image, w: int, h: int, t: float, dur: float, zoom_max: float = 1.06) -> Image.Image:
    prog = _ease_in_out_quad(t / max(0.5, dur))
    scale = 1.0 + (zoom_max - 1.0) * prog
    sw, sh = int(w * scale), int(h * scale)
    resized = bg.resize((sw, sh), resample=Image.Resampling.BILINEAR)
    ox = int((sw - w) * (0.35 + 0.3 * prog))
    oy = int((sh - h) * 0.45)
    return resized.crop((ox, oy, ox + w, oy + h)).convert("RGBA")


def _render_whiteboard_or_editorial_scene(
    scene: dict[str, Any],
    bg_img: Image.Image | None,
    char_bust: Image.Image | None,
    w: int,
    h: int,
    t: float,
    dur: float,
    speaking_amp: float,
    is_short: bool,
) -> Image.Image:
    vmode = scene.get("visual_mode", "vector_whiteboard_scene")
    if bg_img is not None:
        img = _crop_ken_burns(bg_img, w, h, t, dur, zoom_max=1.065 if vmode == "editorial_illustration" else 1.035)
    else:
        img = Image.new("RGBA", (w, h), (222, 236, 250, 255))

    enter = _ease_out_cubic(t / 0.5)
    headline = scene.get("headline") or ""
    subhead = scene.get("subheadline") or ""
    nodes = (scene.get("diagram_nodes") or [])[:6]
    metrics = (scene.get("metrics") or [])[:3]
    chars = scene.get("characters") or [{"role": "engineer", "accent": "#3B82F6"}]

    if vmode == "vector_whiteboard_scene":
        # Proper RGBA alpha-composite subtle 10% clarity veil (preserves crisp cel-shaded background!)
        img = Image.alpha_composite(img, Image.new("RGBA", (w, h), (245, 249, 255, 24)))
        draw = ImageDraw.Draw(img, "RGBA")

        if not is_short:
            wb_x, wb_y, wb_w, wb_h = 96, 98, 1280, 770
        else:
            wb_x, wb_y, wb_w, wb_h = 68, 200, w - 136, 760

        # Whiteboard legs
        leg_ink = (130, 144, 164, 255)
        draw.line([(wb_x + int(wb_w * 0.28), wb_y + wb_h), (wb_x + int(wb_w * 0.25), wb_y + wb_h + 70)], fill=leg_ink, width=8)
        draw.line([(wb_x + int(wb_w * 0.72), wb_y + wb_h), (wb_x + int(wb_w * 0.75), wb_y + wb_h + 70)], fill=leg_ink, width=8)

        # Rounded whiteboard panel
        draw.rounded_rectangle(
            [wb_x, wb_y, wb_x + wb_w, wb_y + wb_h],
            radius=24,
            fill=(250, 252, 255, 248),
            outline=(118, 132, 152, 255),
            width=4,
        )
        # Signature 4-color top accent strip (`yB9SC4HbH5o_t0120s.jpg`!)
        bar_cols = [(59, 130, 246, 255), (239, 68, 68, 255), (245, 158, 11, 255), (16, 185, 129, 255)]
        seg_w = (wb_w - 80) // 4
        for bi, bc in enumerate(bar_cols):
            sx = wb_x + 40 + bi * seg_w
            draw.rectangle([sx, wb_y + 16, sx + seg_w - 6, wb_y + 26], fill=bc)

        f_head = _find_font(bold=True, size=42 if not is_short else 44)
        f_sub = _find_font(bold=False, size=25 if not is_short else 26)
        f_node = _find_font(bold=True, size=26 if not is_short else 28)
        f_val = _find_font(bold=True, mono=True, size=34)

        h_lines = _wrap_text(draw, headline, f_head, wb_w - 96)
        cy = wb_y + 52
        for ln in h_lines[:2]:
            draw.text((wb_x + 48, cy), ln, font=f_head, fill=(18, 24, 38, int(255 * enter)))
            cy += 50
        if subhead:
            s_lines = _wrap_text(draw, subhead, f_sub, wb_w - 96)
            for sln in s_lines[:2]:
                draw.text((wb_x + 48, cy + 4), sln, font=f_sub, fill=(82, 95, 118, int(240 * enter)))
                cy += 34

        # Animated diagram flow tiles inside whiteboard
        if nodes:
            cols = min(3, len(nodes)) if not is_short else min(2, len(nodes))
            rows = math.ceil(len(nodes) / max(1, cols))
            grid_top = cy + 32
            tile_w = (wb_w - 96 - (cols - 1) * 28) // max(1, cols)
            tile_h = min(130, max(92, (wb_h - (grid_top - wb_y) - 180) // max(1, rows)))
            for n_i, nd in enumerate(nodes):
                r, c = divmod(n_i, cols)
                n_prog = _ease_out_cubic((t - 0.25 - n_i * 0.18) / 0.45)
                if n_prog <= 0.02:
                    continue
                tx = wb_x + 48 + c * (tile_w + 28)
                ty = grid_top + r * (tile_h + 24)
                active = n_i == min(len(nodes) - 1, int((t / max(1.0, dur * 0.8)) * len(nodes)))
                fill_c = (59, 130, 246, int(245 * n_prog)) if active else (238, 242, 249, int(245 * n_prog))
                txt_c = (255, 255, 255, int(255 * n_prog)) if active else (30, 41, 59, int(255 * n_prog))
                draw.rounded_rectangle(
                    [tx, ty, tx + tile_w, ty + tile_h],
                    radius=16,
                    fill=fill_c,
                    outline=(148, 163, 184, int(220 * n_prog)),
                    width=2,
                )
                n_lines = _wrap_text(draw, str(nd), f_node, tile_w - 28)
                ny = ty + (tile_h - len(n_lines[:2]) * 32) // 2
                for nln in n_lines[:2]:
                    bb = draw.textbbox((0, 0), nln, font=f_node)
                    draw.text((tx + (tile_w - (bb[2] - bb[0])) // 2, ny), nln, font=f_node, fill=txt_c)
                    ny += 32
                if c < cols - 1 and (n_i + 1) < len(nodes):
                    ax1, ay1 = tx + tile_w + 4, ty + tile_h // 2
                    ax2 = tx + tile_w + 24
                    draw.line([(ax1, ay1), (ax2, ay1)], fill=(59, 130, 246, int(220 * n_prog)), width=3)

        # Bottom metric badges inside whiteboard
        if metrics:
            my = wb_y + wb_h - 118
            mw = (wb_w - 96 - (len(metrics) - 1) * 20) // len(metrics)
            f_mlbl = _find_font(bold=False, size=20)
            for m_i, m in enumerate(metrics):
                mp = _ease_out_cubic((t - 0.5 - m_i * 0.2) / 0.45)
                if mp <= 0.02:
                    continue
                mx = wb_x + 48 + m_i * (mw + 20)
                draw.rounded_rectangle([mx, my, mx + mw, my + 90], radius=14, fill=(18, 24, 36, int(238 * mp)))
                tone = TONE_COLORS.get(str(m.get("tone", "blue")).lower(), TONE_COLORS["blue"])
                val_str = _smart_clip(str(m.get("value", "")), 24)
                use_val_font = _find_font(bold=True, mono=True, size=27 if len(val_str) > 16 else 34)
                draw.text((mx + 20, my + 12), _smart_clip(str(m.get("label", "")), 28), font=f_mlbl, fill=(185, 196, 214, int(255 * mp)))
                draw.text((mx + 20, my + 42), val_str, font=use_val_font, fill=(*tone, int(255 * mp)))

        # Draw @TradeoffLayer character(s) beside/below whiteboard!
        char_cfg = chars[0]
        if not is_short:
            _draw_tradeoff_character(
                img,
                cx=1635,
                foot_y=955,
                scale=1.22,
                role=str(char_cfg.get("role", "engineer")),
                accent_hex=str(char_cfg.get("accent", "#3B82F6")),
                t=t,
                speaking_amp=speaking_amp,
                point_dir=-1,
                bust_rgba=char_bust,
            )
        else:
            _draw_tradeoff_character(
                img,
                cx=835,
                foot_y=1355,
                scale=0.95,
                role=str(char_cfg.get("role", "engineer")),
                accent_hex=str(char_cfg.get("accent", "#3B82F6")),
                t=t,
                speaking_amp=speaking_amp,
                point_dir=-1,
                bust_rgba=char_bust,
            )
    else:
        draw = ImageDraw.Draw(img, "RGBA")
        # Mode B: `editorial_illustration` (`2d883UK7p9E` clean cel-shaded scene + floating dark glass header & metric pills)
        f_head = _find_font(bold=True, size=44 if not is_short else 46)
        f_sub = _find_font(bold=False, size=25 if not is_short else 26)
        f_val = _find_font(bold=True, mono=True, size=36)
        f_mlbl = _find_font(bold=True, size=21)

        panel_x = 76 if not is_short else 56
        panel_y = 68 if not is_short else 190
        panel_w = 880 if not is_short else (w - 112)
        h_lines = _wrap_text(draw, headline, f_head, panel_w - 64)
        s_lines = _wrap_text(draw, subhead, f_sub, panel_w - 64) if subhead else []
        panel_h = 52 + len(h_lines[:2]) * 52 + (len(s_lines[:2]) * 34 + 10 if s_lines else 0)

        draw.rounded_rectangle(
            [panel_x, panel_y, panel_x + panel_w, panel_y + panel_h],
            radius=20,
            fill=(15, 20, 30, int(222 * enter)),
            outline=(255, 255, 255, int(70 * enter)),
            width=2,
        )
        cy = panel_y + 24
        for ln in h_lines[:2]:
            draw.text((panel_x + 32, cy), ln, font=f_head, fill=(248, 250, 252, int(255 * enter)))
            cy += 52
        for sln in s_lines[:2]:
            draw.text((panel_x + 32, cy + 4), sln, font=f_sub, fill=(186, 200, 220, int(245 * enter)))
            cy += 34

        if metrics:
            base_y = (h - 235) if not is_short else 1080
            card_w = 340 if not is_short else (w - 112)
            for m_i, m in enumerate(metrics[:2]):
                mp = _ease_out_cubic((t - 0.35 - m_i * 0.22) / 0.45)
                if mp <= 0.02:
                    continue
                mx = (w - 76 - (m_i + 1) * (card_w + 20) + 20) if not is_short else 56
                my = base_y if not is_short else (base_y + m_i * 124)
                draw.rounded_rectangle(
                    [mx, my, mx + card_w, my + 108],
                    radius=18,
                    fill=(15, 20, 30, int(232 * mp)),
                    outline=(96, 165, 250, int(180 * mp)),
                    width=2,
                )
                tone = TONE_COLORS.get(str(m.get("tone", "amber")).lower(), TONE_COLORS["amber"])
                draw.text((mx + 24, my + 16), str(m.get("label", ""))[:26], font=f_mlbl, fill=(203, 213, 225, int(255 * mp)))
                draw.text((mx + 24, my + 48), str(m.get("value", ""))[:18], font=f_val, fill=(*tone, int(255 * mp)))

    return img


def _draw_chapter_ribbon_and_progress(
    img: Image.Image,
    chapter_title: str,
    global_t: float,
    total_dur: float,
    scene_t: float,
    w: int,
    h: int,
) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    # Subtle 4px progress bar along bottom edge
    prog = max(0.0, min(1.0, global_t / max(1.0, total_dur)))
    draw.rectangle([0, h - 5, w, h], fill=(30, 34, 44, 180))
    draw.rectangle([0, h - 5, int(w * prog), h], fill=(74, 222, 128, 235))

    # Chapter pill at top-right during first 3.6 seconds of a new chapter
    if chapter_title and scene_t < 3.8:
        a = _ease_out_cubic(scene_t / 0.4) * _ease_out_cubic((3.8 - scene_t) / 0.5)
        if a > 0.03:
            f_ch = _find_font(bold=True, size=22)
            txt = f"CHAPTER  •  {chapter_title.upper()[:38]}"
            bb = draw.textbbox((0, 0), txt, font=f_ch)
            pw = (bb[2] - bb[0]) + 44
            px = w - pw - 48
            py = 36
            draw.rounded_rectangle(
                [px, py, px + pw, py + 46],
                radius=23,
                fill=(15, 23, 42, int(228 * a)),
                outline=(96, 165, 250, int(210 * a)),
                width=2,
            )
            draw.text((px + 22, py + 11), txt, font=f_ch, fill=(241, 245, 249, int(255 * a)))


def _draw_short_progressive_captions(
    img: Image.Image,
    words: list[dict[str, Any]],
    t: float,
    w: int,
    h: int,
) -> None:
    """
    Exact @TradeoffLayer Shorts caption style (`bmFUEo6A_1U_t0005s.jpg`):
    5-word phrase groups at lower-third where spoken words (`start <= t`) are crisp
    #FFFFFF Bold and upcoming words in the active group are muted #71717A Bold!
    """
    if not words:
        return
    group_size = 5
    groups = [words[i : i + group_size] for i in range(0, len(words), group_size)]
    active_group = None
    for g in groups:
        g_start = g[0]["start"] - 0.08
        g_end = g[-1]["end"] + 0.25
        if g_start <= t <= g_end:
            active_group = g
            break
    if active_group is None:
        # Check if between groups
        for i in range(len(groups) - 1):
            if groups[i][-1]["end"] <= t < groups[i + 1][0]["start"]:
                active_group = groups[i + 1]
                break
    if not active_group:
        return

    draw = ImageDraw.Draw(img, "RGBA")
    f_cap = _find_font(bold=True, size=52)

    # Split active_group into at most 2 centered lines (~520px max width)
    lines: list[list[dict[str, Any]]] = [[]]
    cur_w = 0
    for wd in active_group:
        wb = draw.textbbox((0, 0), wd["word"] + " ", font=f_cap)
        ww = wb[2] - wb[0]
        if lines[-1] and (cur_w + ww > w - 150) and len(lines) < 2:
            lines.append([wd])
            cur_w = ww
        else:
            lines[-1].append(wd)
            cur_w += ww

    box_h = len(lines) * 68 + 34
    base_y = h - 410
    draw.rounded_rectangle(
        [56, base_y - 18, w - 56, base_y + box_h],
        radius=22,
        fill=(10, 11, 15, 212),
    )

    for l_idx, ln_words in enumerate(lines):
        full_str = " ".join(x["word"] for x in ln_words)
        lb = draw.textbbox((0, 0), full_str, font=f_cap)
        cursor_x = (w - (lb[2] - lb[0])) // 2
        ly = base_y + l_idx * 68
        for w_idx, wd in enumerate(ln_words):
            token_str = wd["word"] + (" " if w_idx < len(ln_words) - 1 else "")
            spoken = t >= (wd["start"] - 0.03)
            col = (255, 255, 255, 255) if spoken else (113, 113, 122, 255)
            draw.text((cursor_x, ly), token_str, font=f_cap, fill=col)
            tb = draw.textbbox((0, 0), token_str, font=f_cap)
            cursor_x += tb[2] - tb[0]


def _compute_audio_envelope(wav_path: Path, fps: int) -> np.ndarray:
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        samples = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    total_frames = max(1, int(math.ceil((n / sr) * fps)))
    hop = max(1, int(sr / fps))
    env = np.zeros(total_frames, dtype=np.float32)
    for f in range(total_frames):
        chunk = samples[f * hop : (f + 1) * hop]
        if len(chunk) > 0:
            env[f] = min(1.0, float(np.sqrt(np.mean(chunk**2))) * 4.5)
    return env


def _add_subtle_sound_design(voice_wav: Path, out_wav: Path, is_short: bool = False) -> Path:
    """Mixes a clean studio transition chime + ultra-subtle -33dB warm ambient bed under the cloned voice."""
    with wave.open(str(voice_wav), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        voice = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0

    t = np.arange(n, dtype=np.float32) / sr
    # Warm ultra-clean -34dB minimalist synth pad (root + fifth gentle shimmer, never muddies speech)
    f0 = 110.0 if not is_short else 130.81
    pad = (
        0.55 * np.sin(2 * np.pi * f0 * t)
        + 0.30 * np.sin(2 * np.pi * (f0 * 1.5) * t)
        + 0.15 * np.sin(2 * np.pi * (f0 * 2.0) * t)
    ) * (0.014 * (0.75 + 0.25 * np.sin(2 * np.pi * 0.12 * t)))

    # Subtle soft transition rise at t=0..0.22s
    sw_len = min(n, int(sr * 0.22))
    if sw_len > 10:
        st = np.linspace(0.0, 1.0, sw_len, dtype=np.float32)
        whoosh = np.sin(2 * np.pi * (180 + 260 * st) * (st * 0.22)) * np.hanning(sw_len) * 0.045
        pad[:sw_len] += whoosh

    mixed = np.clip(voice + pad, -0.98, 0.98)
    pcm = (mixed * 32767.0).astype(np.int16)
    with wave.open(str(out_wav), "wb") as out_wf:
        out_wf.setnchannels(1)
        out_wf.setsampwidth(2)
        out_wf.setframerate(sr)
        out_wf.writeframes(pcm.tobytes())
    return out_wav


def _render_single_scene_worker(job: dict[str, Any]) -> str:
    scene = job["scene"]
    tts_meta = job["tts_meta"]
    wav_path = Path(job["wav_path"])
    bg_path = Path(job["bg_path"]) if job.get("bg_path") else None
    out_mp4 = Path(job["out_mp4"])
    w, h, fps = int(job["w"]), int(job["h"]), int(job["fps"])
    is_short = bool(job["is_short"])
    total_dur = float(job["total_dur"])

    bg_img = Image.open(bg_path).convert("RGB") if (bg_path and bg_path.exists()) else None
    char_path = Path(job["char_path"]) if job.get("char_path") else None
    char_bust = Image.open(char_path).convert("RGBA") if (char_path and char_path.exists()) else None
    env = _compute_audio_envelope(wav_path, fps=fps)
    n_frames = len(env)
    dur = n_frames / float(fps)

    scored_wav = out_mp4.with_suffix(".scored.wav")
    _add_subtle_sound_design(wav_path, scored_wav, is_short=is_short)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-i", str(scored_wav),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-shortest",
        str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None

    vmode = scene.get("visual_mode", "vector_whiteboard_scene")
    words = tts_meta.get("words") or []
    sources = scene.get("resolved_sources") or []
    citation_fact = scene.get("citation_fact") or ""
    chapter_title = scene.get("chapter_title") or ""
    scene_global_start = float(tts_meta.get("start_sec", 0.0))

    #Fade-in / fade-out duration (8 frames = 0.13s crisp studio cut/dissolve)
    fade_frames = 8

    for f_idx in range(n_frames):
        t = f_idx / float(fps)
        amp = float(env[f_idx])

        if vmode == "dark_metric_card":
            frame_rgba = _render_dark_metric_scene(scene, w, h, t, dur, is_short=is_short)
        else:
            frame_rgba = _render_whiteboard_or_editorial_scene(
                scene, bg_img, char_bust, w, h, t, dur, speaking_amp=amp, is_short=is_short
            )

        _draw_source_badge(frame_rgba, sources, citation_fact, w, h, t, is_short=is_short)

        if is_short:
            _draw_short_progressive_captions(frame_rgba, words, t, w, h)
        else:
            _draw_chapter_ribbon_and_progress(
                frame_rgba, chapter_title, scene_global_start + t, total_dur, t, w, h
            )

        rgb = np.asarray(frame_rgba.convert("RGB"), dtype=np.uint8)
        if f_idx < fade_frames:
            alpha = (f_idx + 1) / float(fade_frames)
            rgb = (rgb.astype(np.float32) * alpha).astype(np.uint8)
        elif f_idx >= n_frames - fade_frames:
            alpha = max(0.0, (n_frames - f_idx) / float(fade_frames))
            rgb = (rgb.astype(np.float32) * alpha).astype(np.uint8)

        proc.stdin.write(rgb.tobytes())

    proc.stdin.close()
    rc = proc.wait()
    scored_wav.unlink(missing_ok=True)
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed on scene {scene['id']} (rc={rc})")
    return str(out_mp4)


def _format_srt_ts(sec: float) -> str:
    ms = int(round(max(0.0, sec) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt_subtitles(tts_report: dict[str, Any], out_srt: Path) -> None:
    entries: list[str] = []
    idx = 1
    for sc in tts_report.get("scenes", []):
        words = sc.get("words") or []
        for i in range(0, len(words), 8):
            chunk = words[i : i + 8]
            st = chunk[0]["global_start"]
            et = chunk[-1]["global_end"]
            txt = " ".join(w["word"] for w in chunk)
            entries.append(f"{idx}\n{_format_srt_ts(st)} --> {_format_srt_ts(et)}\n{txt}\n")
            idx += 1
    out_srt.write_text("\n".join(entries), encoding="utf-8")


def generate_thumbnail(plan: dict[str, Any], visual_dir: Path, out_png: Path) -> Path:
    w, h = 1280, 720
    bg_p = visual_dir / "thumbnail_bg.png"
    if bg_p.exists():
        img = Image.open(bg_p).convert("RGB").resize((w, h), Image.Resampling.LANCZOS).convert("RGBA")
    else:
        img = Image.new("RGBA", (w, h), (18, 24, 38, 255))

    # Dramatic left gradient for high-CTR text legibility
    grad = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(grad, "RGBA")
    for x in range(w):
        a = int(215 * max(0.0, 1.0 - (x / (w * 0.72))))
        gdraw.line([(x, 0), (x, h)], fill=(10, 14, 24, a))
    img = Image.alpha_composite(img, grad)

    draw = ImageDraw.Draw(img, "RGBA")
    thumb = plan.get("thumbnail") or {}
    headline = (thumb.get("headline") or plan.get("title", "THE TRADEOFF"))[:28].upper()
    subbadge = (thumb.get("subbadge") or "")[:18]

    f_big = _find_font(bold=True, size=92)
    f_badge = _find_font(bold=True, mono=True, size=44)
    lines = _wrap_text(draw, headline, f_big, 760)
    cy = 155
    for ln in lines[:3]:
        draw.text((68, cy + 4), ln, font=f_big, fill=(0, 0, 0, 200))
        draw.text((64, cy), ln, font=f_big, fill=(255, 255, 255, 255))
        cy += 102

    if subbadge:
        bb = draw.textbbox((0, 0), subbadge, font=f_badge)
        bw = (bb[2] - bb[0]) + 48
        draw.rounded_rectangle([64, cy + 24, 64 + bw, cy + 98], radius=16, fill=(239, 68, 68, 245))
        draw.text((88, cy + 36), subbadge, font=f_badge, fill=(255, 255, 255, 255))

    ceo_bust_p = visual_dir / "char_ceo.png"
    ceo_bust = Image.open(ceo_bust_p).convert("RGBA") if ceo_bust_p.exists() else None
    _draw_tradeoff_character(img, cx=1055, foot_y=695, scale=1.18, role="ceo", accent_hex="#10B981", t=0.4, speaking_amp=0.35, point_dir=-1, bust_rgba=ceo_bust)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(str(out_png), quality=95)
    return out_png


def build_youtube_metadata_and_citations(
    plan: dict[str, Any],
    long_tts: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Generates ready-to-publish YouTube description (with CHAPTERS + SOURCES) and citation audit JSON."""
    seo = plan.get("seo") or {}
    tts_by_id = {s["id"]: s for s in long_tts.get("scenes", [])}

    chapters_lines = []
    for i, sc in enumerate(plan.get("long_scenes", [])):
        ch = (sc.get("chapter_title") or "").strip()
        if i == 0 and not ch:
            ch = "The hook"
        if ch and sc["id"] in tts_by_id:
            st = int(tts_by_id[sc["id"]]["start_sec"])
            mm, ss = divmod(st, 60)
            chapters_lines.append(f"{mm}:{ss:02d} {ch}")

    sources = plan.get("research_metadata", {}).get("sources", [])
    source_lines = [
        f"[{s['id']}] {s['publisher']} ({s['date']}) — {s['title']}\n    {s['url']}"
        for s in sources
    ]

    bullets = "\n".join(f"- {b}" for b in (seo.get("learning_bullets") or []))
    description = (
        f"{seo.get('summary_hook', plan.get('core_paradox', ''))}\n\n"
        f"You'll learn:\n{bullets}\n\n"
        f"CHAPTERS\n" + "\n".join(chapters_lines) + "\n\n"
        f"SOURCES & PUBLIC DATA REFERENCES\n" + "\n".join(source_lines) + "\n"
    )

    shorts_meta = []
    for sh in plan.get("shorts", []):
        shorts_meta.append({
            "id": sh["id"],
            "title": sh.get("title"),
            "description": (
                f"{sh.get('hook_angle', '')}\n\n"
                f"Full animated breakdown on @TradeoffLayer: \"{plan.get('title')}\"\n\n"
                f"Sources:\n" + "\n".join(source_lines[:4])
            ),
            "related_long_video_title": plan.get("title"),
        })

    meta = {
        "channel": "@TradeoffLayer",
        "long_video": {
            "title": plan.get("title"),
            "description": description,
            "tags": seo.get("tags") or [],
            "chapters": chapters_lines,
        },
        "shorts": shorts_meta,
    }
    (output_dir / "youtube_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "description.txt").write_text(description, encoding="utf-8")

    audit = {
        "title": plan.get("title"),
        "retrieved_at": plan.get("research_metadata", {}).get("retrieved_at"),
        "sources": sources,
        "scene_citations": [
            {
                "scene_id": sc["id"],
                "headline": sc.get("headline"),
                "citation_fact": sc.get("citation_fact"),
                "sources": sc.get("resolved_sources", []),
            }
            for sc in plan.get("long_scenes", [])
        ],
    }
    (output_dir / "citations_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def concat_scene_mp4s(scene_mp4s: list[Path], out_mp4: Path) -> Path:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    concat_txt = out_mp4.with_suffix(".concat.txt")
    concat_txt.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in scene_mp4s) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-c:v", "copy",
            "-af", "loudnorm=I=-14.0:TP=-1.5:LRA=6.5",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(out_mp4),
        ],
        check=True,
    )
    concat_txt.unlink(missing_ok=True)
    return out_mp4


def render_all(
    plan_path: Path,
    visual_dir: Path,
    long_tts_dir: Path,
    shorts_tts_root: Path,
    output_dir: Path,
    fps: int = 60,
    max_workers: int = 12,
    allow_local_override: bool = False,
) -> dict[str, Any]:
    _guard_not_alan_hub(allow_override=allow_local_override)
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_scenes_dir = output_dir / "_scene_segments"
    tmp_scenes_dir.mkdir(parents=True, exist_ok=True)

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    long_tts = json.loads((long_tts_dir / "tts_alignment_report.json").read_text(encoding="utf-8"))
    long_tts_map = {s["id"]: s for s in long_tts["scenes"]}
    long_total_dur = float(long_tts["total_duration_sec"])

    workers = min(max_workers, max(2, (os.cpu_count() or 4) // 2))
    print(f"[video_composer] Rendering Long-Form (1920x1080@{fps}fps) across {workers} parallel workers...", flush=True)

    long_jobs: list[dict[str, Any]] = []
    long_part_paths: list[Path] = []
    for sc in plan.get("long_scenes", []):
        sid = sc["id"]
        tmeta = long_tts_map[sid]
        seg_mp4 = tmp_scenes_dir / f"long_{sid}.mp4"
        long_part_paths.append(seg_mp4)
        bg_cand = visual_dir / f"long_{sid}.png"
        c_role = ((sc.get("characters") or [{}])[0].get("role") or "engineer").lower()
        char_cand = visual_dir / f"char_{c_role}.png"
        long_jobs.append({
            "scene": sc,
            "tts_meta": tmeta,
            "wav_path": str(long_tts_dir / tmeta["wav_file"]),
            "bg_path": str(bg_cand) if bg_cand.exists() else None,
            "char_path": str(char_cand) if char_cand.exists() else None,
            "out_mp4": str(seg_mp4),
            "w": 1920,
            "h": 1080,
            "fps": fps,
            "is_short": False,
            "total_dur": long_total_dur,
        })

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_render_single_scene_worker, long_jobs))

    long_mp4 = output_dir / "long_video_1080p60.mp4"
    concat_scene_mp4s(long_part_paths, long_mp4)
    write_srt_subtitles(long_tts, output_dir / "long_video.srt")

    short_outputs: list[str] = []
    for sh in plan.get("shorts", []):
        sh_id = sh["id"]
        sh_tts_dir = shorts_tts_root / sh_id
        sh_tts = json.loads((sh_tts_dir / "tts_alignment_report.json").read_text(encoding="utf-8"))
        sh_map = {s["id"]: s for s in sh_tts["scenes"]}
        sh_dur = float(sh_tts["total_duration_sec"])

        print(f"[video_composer] Rendering Vertical Short '{sh_id}' (1080x1920@{fps}fps)...", flush=True)
        sh_jobs: list[dict[str, Any]] = []
        sh_parts: list[Path] = []
        for sc in sh.get("scenes", []):
            sid = sc["id"]
            tmeta = sh_map[sid]
            seg_mp4 = tmp_scenes_dir / f"{sh_id}_{sid}.mp4"
            sh_parts.append(seg_mp4)
            bg_cand = visual_dir / f"{sh_id}_{sid}.png"
            c_role = ((sc.get("characters") or [{}])[0].get("role") or "engineer").lower()
            char_cand = visual_dir / f"char_{c_role}.png"
            sh_jobs.append({
                "scene": sc,
                "tts_meta": tmeta,
                "wav_path": str(sh_tts_dir / tmeta["wav_file"]),
                "bg_path": str(bg_cand) if bg_cand.exists() else None,
                "char_path": str(char_cand) if char_cand.exists() else None,
                "out_mp4": str(seg_mp4),
                "w": 1080,
                "h": 1920,
                "fps": fps,
                "is_short": True,
                "total_dur": sh_dur,
            })

        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_render_single_scene_worker, sh_jobs))

        sh_mp4 = output_dir / f"{sh_id}_1080x1920_60fps.mp4"
        concat_scene_mp4s(sh_parts, sh_mp4)
        write_srt_subtitles(sh_tts, output_dir / f"{sh_id}.srt")
        short_outputs.append(sh_mp4.name)

    thumb_png = generate_thumbnail(plan, visual_dir, output_dir / "thumbnail_1280x720.png")
    build_youtube_metadata_and_citations(plan, long_tts, output_dir)

    # Clean temporary per-scene segments
    for p in tmp_scenes_dir.glob("*.mp4"):
        p.unlink(missing_ok=True)
    tmp_scenes_dir.rmdir()

    summary = {
        "ok": True,
        "elapsed_sec": round(time.time() - t0, 2),
        "long_video": long_mp4.name,
        "long_duration_sec": long_total_dur,
        "shorts": short_outputs,
        "thumbnail": thumb_png.name,
    }
    (output_dir / "render_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Tradeoff Layer 60fps Remote Video & Shorts Compositor")
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--visual-dir", required=True, type=Path)
    ap.add_argument("--long-tts-dir", required=True, type=Path)
    ap.add_argument("--shorts-tts-root", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--max-workers", type=int, default=12)
    ap.add_argument("--allow-local-override", action="store_true")
    args = ap.parse_args()

    res = render_all(
        plan_path=args.plan,
        visual_dir=args.visual_dir,
        long_tts_dir=args.long_tts_dir,
        shorts_tts_root=args.shorts_tts_root,
        output_dir=args.output_dir,
        fps=args.fps,
        max_workers=args.max_workers,
        allow_local_override=args.allow_local_override,
    )
    print(f"[video_composer] Finished rendering in {res['elapsed_sec']}s -> {res['long_video']} + {res['shorts']}")


if __name__ == "__main__":
    main()
