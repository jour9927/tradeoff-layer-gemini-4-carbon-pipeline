#!/usr/bin/env python3
"""
Tradeoff Layer v3 — Frame-Accurate Foley & UI Sound Effects Engine.

Replaces the v1/v2 continuous synth drone with @TradeoffLayer's authentic studio sound design:
  - Ultra-clean pause floor (<-66 dBFS) so spoken pauses carry dramatic weight
  - Crisp, frame-accurate physical Foley & UI micro-transients synced to shot cuts and keyframe events:
      * `whoosh_soft`: Bandpass-swept air swish on major scene/camera transitions
      * `pop_clean`: Tactile pitch-envelope pop on icon/node/pedestal reveals
      * `tick_counter`: Micro-mechanical click on waffle-grid dots & numeric counters
      * `stamp_heavy`: Tactile thud + crisp slap on copier lid, hydraulic press, or red X cross-out
      * `shutter_flash`: Dual-click mechanical camera shutter on flash transitions
      * `paper_slide`: Crisp desk slide for checks, contracts, and silicon wafers
      * `chime_reveal`: Subtle warm harmonic bell on act-opening/closing paradoxes
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _butter_like_env(n: int, attack_frac: float = 0.08, decay_exp: float = 5.0) -> np.ndarray:
    if n <= 2:
        return np.ones(max(1, n), dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    att = np.clip(t / max(1e-4, attack_frac), 0.0, 1.0)
    dec = np.exp(-decay_exp * np.maximum(0.0, t - attack_frac))
    env = att * dec
    env[-min(24, n // 4) :] *= np.linspace(1.0, 0.0, min(24, n // 4), dtype=np.float32)
    return env


def synth_sfx(kind: str, sr: int = 24000, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if kind == "whoosh_soft":
        dur = 0.145
        n = int(sr * dur)
        t = np.linspace(0.0, dur, n, dtype=np.float32)
        noise = rng.standard_normal(n).astype(np.float32)
        # Moving resonant low-pass / bandpass sweep via windowed modulation
        kernel = np.hanning(31).astype(np.float32)
        kernel /= kernel.sum()
        smooth = np.convolve(noise, kernel, mode="same")
        sweep = np.sin(2.0 * math.pi * (240.0 * t + 1900.0 * (t ** 2) * (1.0 - 2.2 * t)))
        sig = (0.65 * smooth + 0.35 * smooth * sweep) * (np.maximum(0.0, np.sin(math.pi * (t / dur))) ** 1.6)
        return (sig * 0.055).astype(np.float32)

    if kind == "pop_clean":
        dur = 0.052
        n = int(sr * dur)
        t = np.linspace(0.0, dur, n, dtype=np.float32)
        # Rapid pitch drop 660Hz -> 210Hz
        phase = 2.0 * math.pi * (210.0 * t + (450.0 / 38.0) * (1.0 - np.exp(-38.0 * t)))
        sig = np.sin(phase) * _butter_like_env(n, attack_frac=0.05, decay_exp=7.5)
        return (sig * 0.085).astype(np.float32)

    if kind == "tick_counter":
        dur = 0.015
        n = int(sr * dur)
        t = np.linspace(0.0, dur, n, dtype=np.float32)
        sig = (
            0.65 * np.sin(2.0 * math.pi * 2150.0 * t)
            + 0.35 * np.sin(2.0 * math.pi * 4300.0 * t)
        ) * _butter_like_env(n, attack_frac=0.04, decay_exp=12.0)
        return (sig * 0.042).astype(np.float32)

    if kind == "stamp_heavy":
        dur = 0.135
        n = int(sr * dur)
        t = np.linspace(0.0, dur, n, dtype=np.float32)
        phase = 2.0 * math.pi * (44.0 * t + (95.0 / 26.0) * (1.0 - np.exp(-26.0 * t)))
        thud = np.sin(phase) * _butter_like_env(n, attack_frac=0.03, decay_exp=8.5)
        slap_n = min(n, int(sr * 0.032))
        slap = rng.standard_normal(slap_n).astype(np.float32) * _butter_like_env(slap_n, 0.04, 10.0)
        out = thud * 0.13
        out[:slap_n] += slap * 0.055
        return out.astype(np.float32)

    if kind == "shutter_flash":
        dur = 0.075
        n = int(sr * dur)
        out = np.zeros(n, dtype=np.float32)
        c1 = int(sr * 0.010)
        c2_start = int(sr * 0.034)
        c2 = int(sr * 0.012)
        out[:c1] = rng.standard_normal(c1).astype(np.float32) * _butter_like_env(c1, 0.05, 11.0) * 0.08
        if c2_start + c2 <= n:
            out[c2_start : c2_start + c2] = (
                rng.standard_normal(c2).astype(np.float32) * _butter_like_env(c2, 0.05, 9.0) * 0.065
            )
        return out

    if kind == "paper_slide":
        dur = 0.155
        n = int(sr * dur)
        t = np.linspace(0.0, dur, n, dtype=np.float32)
        raw = rng.standard_normal(n).astype(np.float32)
        diff = np.diff(raw, prepend=raw[0])
        env = np.maximum(0.0, np.sin(math.pi * (t / dur))) ** 1.3
        return (diff * env * 0.028).astype(np.float32)

    if kind == "chime_reveal":
        dur = 0.42
        n = int(sr * dur)
        t = np.linspace(0.0, dur, n, dtype=np.float32)
        sig = (
            0.62 * np.sin(2.0 * math.pi * 523.25 * t)
            + 0.30 * np.sin(2.0 * math.pi * 783.99 * t)
            + 0.08 * np.sin(2.0 * math.pi * 1046.50 * t)
        ) * _butter_like_env(n, attack_frac=0.03, decay_exp=5.8)
        return (sig * 0.052).astype(np.float32)

    return np.zeros(int(sr * 0.02), dtype=np.float32)


def _mix_at(buf: np.ndarray, clip: np.ndarray, offset_sec: float, sr: int, gain: float = 1.0) -> None:
    start = max(0, int(round(offset_sec * sr)))
    if start >= len(buf):
        return
    end = min(len(buf), start + len(clip))
    buf[start:end] += clip[: end - start] * float(gain)


def build_scene_sfx_events(scene: dict[str, Any], dur_sec: float, index: int) -> list[tuple[float, str, float]]:
    """
    Deterministically derives frame-accurate Foley cues from a shot's `template` / `sfx_cues`
    so audio transients land right on visual motion keyframes.
    """
    events: list[tuple[float, str, float]] = []
    # Explicit cues take priority if authored in shot spec
    explicit = scene.get("sfx_cues") or []
    if explicit:
        for cue in explicit:
            frac = float(cue.get("at_frac", 0.08))
            events.append((max(0.0, min(dur_sec - 0.05, frac * dur_sec)), str(cue.get("kind", "pop_clean")), float(cue.get("gain", 1.0))))
        return events

    tpl = str(scene.get("template") or scene.get("animation_type") or "")
    if scene.get("chapter_title"):
        events.append((0.02, "chime_reveal", 0.85))
    elif index > 0 and index % 2 == 0:
        events.append((0.01, "whoosh_soft", 0.72))

    if tpl in ("copier_clone_action", "hydraulic_press_action", "red_cross_stamp"):
        events.append((0.12 * dur_sec, "paper_slide", 0.85))
        events.append((0.54 * dur_sec, "stamp_heavy", 0.95))
    elif tpl in ("studio_shutter_flash",):
        events.append((0.46 * dur_sec, "shutter_flash", 1.0))
    elif tpl in ("desk_deal_transfer", "split_phone_call"):
        events.append((0.18 * dur_sec, "pop_clean", 0.75))
        events.append((0.48 * dur_sec, "paper_slide", 0.90))
    elif tpl in ("waffle_100_dots", "pedestal_rivalry", "timeline_milestones"):
        events.append((0.10 * dur_sec, "pop_clean", 0.80))
        for k in range(4):
            events.append(((0.24 + k * 0.12) * dur_sec, "tick_counter", 0.75))
    elif tpl in ("workbench_hbm_stack", "thermal_mrmuf_vs_tcncf", "cowos_interposer_flow"):
        events.append((0.15 * dur_sec, "pop_clean", 0.75))
        events.append((0.52 * dur_sec, "tick_counter", 0.85))
        events.append((0.68 * dur_sec, "pop_clean", 0.80))
    else:
        events.append((0.14 * dur_sec, "pop_clean", 0.65))

    return events


def mix_foley_with_voice(voice_mono: np.ndarray, sr: int, scene: dict[str, Any], index: int = 0) -> np.ndarray:
    """
    Mixes subtle, crisp Foley transients onto clean mastered voiceover WITHOUT adding any
    continuous background noise/hum, preserving <-66 dB natural speech pauses.
    """
    dur_sec = len(voice_mono) / float(sr)
    fx_bus = np.zeros_like(voice_mono, dtype=np.float32)
    events = build_scene_sfx_events(scene, dur_sec=dur_sec, index=index)
    for ev_idx, (t_sec, kind, gain) in enumerate(events):
        clip = synth_sfx(kind, sr=sr, seed=101 + index * 19 + ev_idx)
        _mix_at(fx_bus, clip, offset_sec=t_sec, sr=sr, gain=gain)

    mixed = np.nan_to_num(voice_mono.astype(np.float32) + fx_bus, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    if peak > 0.96:
        mixed *= 0.96 / peak
    return mixed
