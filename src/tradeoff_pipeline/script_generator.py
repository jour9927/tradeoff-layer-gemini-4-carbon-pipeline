#!/usr/bin/env python3
"""
Tradeoff Layer v3 — Shot-by-Shot Industrial Thriller Scriptwriter & Native Hook-Excerpt Short Engine.

Matches @TradeoffLayer's authentic benchmark metrics:
  - Duration: 8–10 minutes (~1,250–1,450 words)
  - Shot density: 85–110 rapid 3.5s–5.2s bespoke visual shots (zero 20-second static holds)
  - Visual grammar: Alternates illustrated multi-layer cartoon story/action shots with clean
    cream (#F6EFE0) and midnight (#131C31) minimalist micro-infographics (18 shot templates)
  - Paired Short: Authentic @TradeoffLayer traffic-driver (`hook_prefix_excerpt`) — natively
    re-composes the first ~26–30 shots (~120–145s Cold Open + Act I cliffhanger) in 9:16 vertical
    with zero burned-in KTV subtitle clutter, cutting cleanly on the cliffhanger question.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

V3_TEMPLATES = [
    "exterior_hq_walk",
    "desk_deal_transfer",
    "copier_blueprint_action",
    "hydraulic_press_crush",
    "studio_shutter_flash",
    "split_screen_rivalry",
    "cleanroom_microscope_zoom",
    "server_hall_waiting",
    "oven_conveyor_bottleneck",
    "waffle_100_dots",
    "pedestal_rivalry",
    "red_cross_stamp",
    "workbench_hbm_stack",
    "thermal_mrmuf_vs_tcncf",
    "rain_pipe_analogy",
    "cowos_silicon_bridge",
    "wafer_3x_cannibalization",
    "timeline_paradox_arc",
]

SYSTEM_PROMPT = f"""You are the head showrunner and storyboard director of the YouTube channel "Tradeoff Layer" (@TradeoffLayer), famous for 1M–2M view animated semiconductor, AI infrastructure, and business-thriller explainers ("Why No One Can Copy ASML?", "Can Google Beat Nvidia? TPU vs GPU Explained", "Why Intel Couldn't Kill AMD?", "Why Every Phone Has a Sony Camera Sensor?").

STRICT @TRADEOFFLAYER v3 PRODUCTION RULES:
1. Length & Shot Density: Write an 8-to-9 minute chronological engineering & business thriller (~1,300–1,450 words) broken into 85 to 100 rapid shots (`long_scenes`), where EVERY shot is 1–2 punchy spoken sentences (11–18 words, ~3.8–5.2 seconds) with its OWN bespoke visual action. NEVER let one visual hold for 15+ seconds.
2. Storytelling DNA:
   - Cold Open (`0:00–2:05`, first ~26 shots): Start inside a shocking deal, paradox, or rivalry twist. End the Cold Open on a sharp cliffhanger question — because the first `short_excerpt_shots` (~26 shots) ALSO serve verbatim as the paired 9:16 traffic-driving YouTube Short!
   - Chronological History + Concrete People/Companies: Use real years, contracts, engineering gambles, and rival miscalculations.
   - Everyday Physical Analogy: Explain the core bottleneck with an intuitive physical metaphor before introducing technical acronyms.
   - Final Act ("The Tradeoff"): Show why the exact architectural choice that created dominance also creates a painful cost or future vulnerability.
3. Visual Templates (`template` field — rotate dynamically so consecutive shots never repeat the exact same static layout):
   Choose from: {", ".join(V3_TEMPLATES)}.
4. Clean Screen Policy + Scraped Source Citations:
   - NEVER put long sentences on screen. `visual_label` / `prop_text` / `big_stat` must be 1–3 crisp words or a single bold metric (`53%`, `$15B`, `3x WAFER`, `1,024-BIT`).
   - Ground factual claims in `[S1..Sn]` via `source_ids` and `citation_note` (rendered only as a subtle 2.4-second frameless micro-footnote on metric shots + full URLs in YouTube description).

Return STRICTLY valid JSON matching:
{{
  "version": "v3",
  "title": "Why ...? ... Explained",
  "slug": "kebab-slug",
  "core_paradox": "One sentence summary of the central tradeoff",
  "thumbnail": {{
    "headline": "2-3 GIANT COMIC WORDS (e.g. SOLD OUT / PAY UP / LOCKED IN)",
    "accent_word": "Optional brand/tag (e.g. SK hynix)",
    "visual_prompt": "2D flat cel-shaded comic close-up scene prompt"
  }},
  "seo": {{
    "summary_hook": "2-paragraph YouTube description opening",
    "learning_bullets": ["4-5 crisp takeaways"],
    "tags": ["20 SEO tags"]
  }},
  "short_config": {{
    "id": "short_01",
    "mode": "hook_prefix_excerpt",
    "excerpt_shot_count": 26,
    "title": "Punchy Short title #ai #semiconductors #tech",
    "hook_angle": "Counter-intuitive hook summary"
  }},
  "long_scenes": [
    {{
      "id": "L001",
      "chapter_title": "Set on first shot of each chapter else empty string",
      "voice_style": "hook | narrative | analytical",
      "speed": 1.02,
      "template": "one of the 18 V3_TEMPLATES",
      "bg_key": "set_nvidia_hq | set_sk_cleanroom | set_samsung_office | set_amd_2013_lab | set_tsmc_fab | set_datacenter_hall | set_sunny_office | set_courtyard_road",
      "left_actor": "char_jensen_talk | char_jensen_shock | char_sk_eng_talk | char_sk_eng_proud | char_samsung_dismiss | char_samsung_sweat | char_amd_eng_talk | char_tsmc_eng_talk | none",
      "right_actor": "actor key or none",
      "prop_key": "prop_hbm_tower | prop_gpu_package | prop_silicon_wafer | prop_server_rack | none",
      "narration": "11 to 18 words of crisp spoken narration.",
      "big_stat": "Optional bold number like 53% or 3x or $30,000",
      "visual_label": "1-3 word in-scene label (e.g. SK hynix / MR-MUF / 2019)",
      "secondary_label": "Optional rival/comparison label (e.g. SAMSUNG / DDR5)",
      "source_ids": ["S1"],
      "citation_note": "Optional brief factual citation note (only on data/metric shots, else empty)"
    }}
  ]
}}
"""


def _extract_json_object(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM output did not contain a JSON object:\n{raw[:600]}")
    return json.loads(cleaned[start : end + 1])


def _build_user_prompt(dossier: dict[str, Any], target_long_scenes: int = 90) -> str:
    sources_txt = []
    for s in dossier.get("sources", []):
        facts = "; ".join(s.get("key_facts") or [])
        sources_txt.append(
            f"[{s['id']}] Publisher: {s['publisher']} | Date: {s['date']} | Title: {s['title']}\n"
            f"    URL: {s['url']}\n"
            f"    Summary: {s['summary'][:700]}\n"
            + (f"    Numeric Facts: {facts}\n" if facts else "")
        )
    joined_sources = "\n".join(sources_txt)
    return (
        f"TARGET TOPIC: {dossier['topic']}\n"
        f"TARGET SHOT COUNT: {target_long_scenes} rapid shots (~1,320–1,450 words total, ~8–9 minutes, ~4.5s per shot).\n"
        f"PAIRED SHORT: First 26 shots (~125s) form the self-contained Cold Open + Act I cliffhanger hook for the 9:16 Short.\n\n"
        f"SCRAPED PUBLIC SOURCES:\n{joined_sources}\n\n"
        f"Output ONLY the valid v3 JSON production plan."
    )


def call_gemini_api(system_prompt: str, user_prompt: str, api_key: str, model: str = "gemini-2.5-pro") -> dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.55,
            "responseMimeType": "application/json",
            "maxOutputTokens": 32768,
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return _extract_json_object(text)


def call_openai_compatible_api(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4.1",
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.55,
        "max_tokens": 24000,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = data["choices"][0]["message"]["content"]
    return _extract_json_object(text)


def call_local_transformers_llm(
    system_prompt: str,
    user_prompt: str,
    model_id: str = "Qwen/Qwen2.5-14B-Instruct",
) -> dict[str, Any]:
    """Runs Qwen2.5-14B-Instruct BF16 directly on the 96GB RTX PRO 6000 worker when no cloud API key is set."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_cuda = False
    chosen_model = model_id
    if torch.cuda.is_available():
        free_b, _ = torch.cuda.mem_get_info(0)
        free_gib = free_b / (1024**3)
        if free_gib >= 32.0:
            use_cuda = True
        elif free_gib >= 12.0:
            use_cuda = True
            if "14B" in chosen_model:
                chosen_model = "Qwen/Qwen2.5-7B-Instruct"
    if not use_cuda and "14B" in chosen_model:
        chosen_model = "Qwen/Qwen2.5-3B-Instruct"
        torch.set_num_threads(min(36, max(4, (os.cpu_count() or 8) - 6)))

    tok = AutoTokenizer.from_pretrained(chosen_model)
    mdl = AutoModelForCausalLM.from_pretrained(
        chosen_model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if use_cuda else None,
        low_cpu_mem_usage=True,
    )
    if not use_cuda:
        mdl.to("cpu")
    mdl.eval()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([prompt_str], return_tensors="pt")
    if use_cuda:
        inputs = inputs.to("cuda")

    with torch.inference_mode():
        gen_ids = mdl.generate(
            **inputs,
            max_new_tokens=18000,
            do_sample=True,
            temperature=0.45,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    out_ids = gen_ids[0][inputs.input_ids.shape[1] :]
    raw_out = tok.decode(out_ids, skip_special_tokens=True)
    return _extract_json_object(raw_out)


def enrich_plan_with_sources(plan: dict[str, Any], dossier: dict[str, Any]) -> dict[str, Any]:
    src_map = {s["id"]: s for s in dossier.get("sources", [])}
    default_sid = dossier["sources"][0]["id"] if dossier.get("sources") else None

    for sc in plan.get("long_scenes", []):
        sids = [sid for sid in (sc.get("source_ids") or []) if sid in src_map]
        if not sids and default_sid:
            sids = [default_sid]
        sc["source_ids"] = sids
        sc["resolved_sources"] = [
            {
                "id": sid,
                "badge_label": src_map[sid]["badge_label"],
                "short_url": src_map[sid]["short_url"],
                "title": src_map[sid]["title"],
                "url": src_map[sid]["url"],
                "publisher": src_map[sid]["publisher"],
                "date": src_map[sid]["date"],
            }
            for sid in sids
        ]

    # Ensure authentic `shorts` structure using `hook_prefix_excerpt` if not explicitly populated
    short_cfg = plan.get("short_config") or {
        "id": "short_01",
        "mode": "hook_prefix_excerpt",
        "excerpt_shot_count": min(26, len(plan.get("long_scenes", []))),
        "title": f"{plan.get('title', 'The AI Bottleneck')} #ai #semiconductors #tech",
        "hook_angle": plan.get("core_paradox", ""),
    }
    plan["short_config"] = short_cfg
    if not plan.get("shorts"):
        n_ex = min(int(short_cfg.get("excerpt_shot_count", 26)), len(plan.get("long_scenes", [])))
        plan["shorts"] = [
            {
                "id": short_cfg.get("id", "short_01"),
                "mode": "hook_prefix_excerpt",
                "excerpt_shot_count": n_ex,
                "title": short_cfg.get("title"),
                "hook_angle": short_cfg.get("hook_angle"),
                "cta_target_title": plan.get("title"),
                "scenes": plan.get("long_scenes", [])[:n_ex],
            }
        ]

    plan["research_metadata"] = {
        "topic": dossier.get("topic"),
        "retrieved_at": dossier.get("retrieved_at"),
        "source_count": len(src_map),
        "sources": dossier.get("sources", []),
    }
    return plan


def generate_production_plan(
    dossier_path: Path,
    output_path: Path,
    backend: str = "auto",
    model: str = "",
    target_long_scenes: int = 90,
) -> dict[str, Any]:
    dossier = json.loads(dossier_path.read_text(encoding="utf-8"))
    user_prompt = _build_user_prompt(dossier, target_long_scenes=target_long_scenes)

    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    openai_key = os.environ.get("OPENAI_API_KEY") or ""
    openai_base = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    if backend == "gemini" or (backend == "auto" and gemini_key):
        plan = call_gemini_api(SYSTEM_PROMPT, user_prompt, api_key=gemini_key, model=model or "gemini-2.5-pro")
    elif backend == "openai" or (backend == "auto" and openai_key):
        plan = call_openai_compatible_api(
            SYSTEM_PROMPT, user_prompt, api_key=openai_key, base_url=openai_base, model=model or "gpt-4.1"
        )
    else:
        plan = call_local_transformers_llm(
            SYSTEM_PROMPT, user_prompt, model_id=model or "Qwen/Qwen2.5-14B-Instruct"
        )

    plan = enrich_plan_with_sources(plan, dossier)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(description="Tradeoff Layer v3 Shot-by-Shot Thriller Scriptwriter")
    ap.add_argument("--dossier", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--backend", choices=["auto", "gemini", "openai", "transformers"], default="auto")
    ap.add_argument("--model", default="")
    ap.add_argument("--long-scenes", type=int, default=90)
    args = ap.parse_args()

    plan = generate_production_plan(
        dossier_path=args.dossier,
        output_path=args.output,
        backend=args.backend,
        model=args.model,
        target_long_scenes=args.long_scenes,
    )
    print(
        f"[script_generator] v3 plan '{plan.get('title')}' "
        f"({len(plan.get('long_scenes', []))} rapid shots, Short excerpt={plan['short_config']['excerpt_shot_count']} shots) -> {args.output}"
    )


if __name__ == "__main__":
    main()
