#!/usr/bin/env python3
"""
Tradeoff Layer — AI Scriptwriter & Paired Shorts Traffic-Driver Generator.
Reconstructs the exact narrative DNA of @TradeoffLayer:
  - Immediate paradox / origin hook (zero channel-intro fluff)
  - Tangible everyday analogy ("buckets in the rain", "cake factory", "toll booth")
  - Quantitative engineering & economic breakdown grounded in scraped public sources [S1..Sn]
  - Signature closing chapter: "The Tradeoff" (why their biggest strength is also their risk)
  - Paired 60–75s vertical Short(s) engineered as a curiosity funnel back to the Long-Form video

Supports unattended execution via:
  1. Google Gemini API (`GEMINI_API_KEY` / `GOOGLE_API_KEY`)
  2. OpenAI-compatible API (`OPENAI_API_KEY` + optional `OPENAI_BASE_URL`)
  3. Local/Remote HuggingFace `Qwen/Qwen2.5-7B-Instruct` on the GPU/48-vCPU worker (zero API keys needed)
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


SYSTEM_PROMPT = """You are the head writer and showrunner of the YouTube channel "Tradeoff Layer" (@TradeoffLayer), famous for 1M–2M view animated tech, semiconductor, AI infrastructure, and cybersecurity explainers (such as "Can Google Beat Nvidia? TPU vs GPU Explained", "Why No One Can Copy ASML?", "Why Every Phone Has a Sony Camera Sensor?", and "AI Costs Explained: Why Output Tokens Cost 5x More").

STRICT EDITORIAL & VOICE RULES OF @TradeoffLayer:
1. NO YouTube clichés: NEVER say "Hey guys", "Welcome back", "Smash like", or "In today's video". Start sentence 1 directly inside a concrete paradox, historical moment, or surprising bill/number.
2. Everyday physical analogy: Early in the script, explain the core technical mechanism using one vivid everyday analogy (like buckets catching rain, a cake factory, a hotel front desk, or an assembly line vs a master chef) before layering in the real engineering terms.
3. Short, punchy spoken sentences: Write for spoken rhythm at ~165 words per minute. Mix 6–14 word sentences with crisp contrast ("Same tokens. Same chip. Five times the price.").
4. Ground EVERY factual metric, date, dollar amount, or technical claim in the provided SCRAPED PUBLIC SOURCES [S1..Sn]. Attach `source_ids` (e.g. ["S1", "S3"]) and a concise `citation_fact` to scenes that use scraped facts so the video renderer displays live source badges on screen.
5. Visual modes: Every scene MUST choose one of @TradeoffLayer's 3 signature visual modes:
   - `"dark_metric_card"`: Matte black #0B0B0D canvas with sleek #1C1C1F cards, green/red/blue metrics, or token/pipeline flow boxes connected by smooth curves.
   - `"vector_whiteboard_scene"`: Bright studio/campus/datacenter AI-illustrated background + Tradeoff Layer's signature flat-vector characters (expressive cel-shaded upper body + minimalist black stick arms/legs) beside a clean rounded whiteboard diagram.
   - `"editorial_illustration"`: Full-bleed flat cel-shaded editorial illustration with crisp dark-navy ink outlines + animated kinetic callout labels and counters.
6. Always end the Long-Form video with a final chapter titled around "The Tradeoff" — showing how the exact design or business choice that created dominance also creates a structural vulnerability or cost.
7. Paired Short (`shorts` array): Create at least 1 vertical YouTube Short (60–75 seconds, 6–7 fast scenes, ~175–210 words total) that isolates the single sharpest counter-intuitive hook from the Long-Form video, cites `[S1]`/`[S2]`, and ends with a natural 1-sentence cliffhanger pointing viewers to the full breakdown on Tradeoff Layer.

Return STRICTLY valid JSON (no markdown fences outside JSON) matching this schema:
{
  "title": "Compelling @TradeoffLayer question/paradox title (under 62 chars)",
  "slug": "kebab-case-slug",
  "core_paradox": "One sentence summarizing the central tradeoff",
  "thumbnail": {
    "headline": "2-4 BOLD WORDS",
    "subbadge": "$XXB / 10x metric",
    "visual_prompt": "Flat cel-shaded vector editorial illustration prompt (no text in image)"
  },
  "seo": {
    "summary_hook": "2-paragraph YouTube description opening in @TradeoffLayer style",
    "learning_bullets": ["4-5 bullet points under You'll learn:"],
    "tags": ["20 multilingual + English SEO tags"]
  },
  "long_scenes": [
    {
      "id": "L01",
      "chapter_title": "Chapter title (set on first scene of each chapter, empty string if continuing same chapter)",
      "voice_style": "hook | narrative | analytical",
      "visual_mode": "dark_metric_card | vector_whiteboard_scene | editorial_illustration",
      "narration": "45 to 75 words of spoken narration in exact @TradeoffLayer voice.",
      "headline": "Short 2-6 word on-screen title",
      "subheadline": "Concise subtitle or physical analogy tagline",
      "metrics": [
        {"label": "Metric name", "value": "$15 / 73x / 92%", "tone": "green | red | blue | amber | neutral"}
      ],
      "diagram_nodes": ["Step/Node 1", "Step/Node 2", "Step/Node 3"],
      "characters": [
        {"role": "engineer | ceo | analyst", "label": "Optional label", "accent": "#3B82F6"}
      ],
      "ai_image_prompt": "Clean flat cel-shaded 2D vector background/scene illustration prompt, crisp dark ink outlines, warm daylight or sleek datacenter palette, NO text or letters",
      "source_ids": ["S1"],
      "citation_fact": "Short factual note shown in the on-screen source citation badge"
    }
  ],
  "shorts": [
    {
      "id": "short_01",
      "title": "Punchy Short title with #ai #semiconductors #tech",
      "hook_angle": "Single counter-intuitive question",
      "cta_target_title": "Matches Long-Form title",
      "scenes": [
        {
          "id": "S01",
          "voice_style": "short_punchy",
          "speed": 1.04,
          "visual_mode": "dark_metric_card | vector_whiteboard_scene | editorial_illustration",
          "narration": "22 to 34 words of crisp fast-paced Short narration.",
          "headline": "2-5 word bold header",
          "subheadline": "Punchy contrast line",
          "metrics": [{"label": "Key stat", "value": "5x", "tone": "red"}],
          "diagram_nodes": ["Node A", "Node B", "Node C"],
          "characters": [{"role": "engineer", "label": "", "accent": "#10B981"}],
          "ai_image_prompt": "Vertical 9:16 flat cel-shaded vector scene illustration prompt, clean outlines, no text",
          "source_ids": ["S1"],
          "citation_fact": "Verified metric from scraped source"
        }
      ]
    }
  ]
}
Aim for 10 to 12 `long_scenes` (~550–750 words total, ~3.5 to 5 minutes of dense broadcast narration) and 1 paired Short of 6 to 7 `scenes` (~165–210 words, ~60–70 seconds).
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


def _build_user_prompt(dossier: dict[str, Any], target_long_scenes: int = 10) -> str:
    sources_txt = []
    for s in dossier.get("sources", []):
        facts = "; ".join(s.get("key_facts") or [])
        sources_txt.append(
            f"[{s['id']}] Publisher: {s['publisher']} | Date: {s['date']} | Title: {s['title']}\n"
            f"    URL: {s['url']}\n"
            f"    Summary: {s['summary'][:550]}\n"
            + (f"    Numeric Facts: {facts}\n" if facts else "")
        )
    joined_sources = "\n".join(sources_txt)
    return (
        f"TARGET TOPIC: {dossier['topic']}\n"
        f"TARGET LONG-FORM SCENE COUNT: {target_long_scenes} scenes (each 50–70 words, crisp chapters, ending with 'The Tradeoff').\n"
        f"PAIRED SHORTS: 1 vertical Short (6–7 scenes, ~60–70 seconds) driving viewers to the full Long-Form video.\n\n"
        f"SCRAPED PUBLIC SOURCES (use ONLY verifiable facts from these sources and cite their IDs like S1, S2 in `source_ids`):\n"
        f"{joined_sources}\n\n"
        f"Output ONLY the valid JSON production plan."
    )


def call_gemini_api(system_prompt: str, user_prompt: str, api_key: str, model: str = "gemini-2.5-flash") -> dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.55,
            "responseMimeType": "application/json",
            "maxOutputTokens": 8192,
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
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
        "max_tokens": 7500,
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
    with urllib.request.urlopen(req, timeout=240) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = data["choices"][0]["message"]["content"]
    return _extract_json_object(text)


def call_local_transformers_llm(
    system_prompt: str,
    user_prompt: str,
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
) -> dict[str, Any]:
    """Zero-API-key fallback running directly on the remote RTX PRO 6000 / b850 worker."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_cuda = False
    if torch.cuda.is_available():
        free_b, _ = torch.cuda.mem_get_info(0)
        if free_b / (1024**3) >= 16.0:
            use_cuda = True

    if not use_cuda:
        torch.set_num_threads(min(36, max(4, (os.cpu_count() or 8) - 6)))

    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
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
            max_new_tokens=5500,
            do_sample=True,
            temperature=0.45,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    out_ids = gen_ids[0][inputs.input_ids.shape[1] :]
    raw_out = tok.decode(out_ids, skip_special_tokens=True)
    return _extract_json_object(raw_out)


def enrich_plan_with_sources(plan: dict[str, Any], dossier: dict[str, Any]) -> dict[str, Any]:
    """Attach full scraped source objects to every scene and build YouTube description + audit file."""
    src_map = {s["id"]: s for s in dossier.get("sources", [])}
    default_sid = dossier["sources"][0]["id"] if dossier.get("sources") else None

    for collection in [plan.get("long_scenes", []), *[sh.get("scenes", []) for sh in plan.get("shorts", [])]]:
        for sc in collection:
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
    target_long_scenes: int = 10,
) -> dict[str, Any]:
    dossier = json.loads(dossier_path.read_text(encoding="utf-8"))
    user_prompt = _build_user_prompt(dossier, target_long_scenes=target_long_scenes)

    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    openai_key = os.environ.get("OPENAI_API_KEY") or ""
    openai_base = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    if backend == "gemini" or (backend == "auto" and gemini_key):
        plan = call_gemini_api(SYSTEM_PROMPT, user_prompt, api_key=gemini_key, model=model or "gemini-2.5-flash")
    elif backend == "openai" or (backend == "auto" and openai_key):
        plan = call_openai_compatible_api(
            SYSTEM_PROMPT, user_prompt, api_key=openai_key, base_url=openai_base, model=model or "gpt-4.1"
        )
    else:
        plan = call_local_transformers_llm(
            SYSTEM_PROMPT, user_prompt, model_id=model or "Qwen/Qwen2.5-3B-Instruct"
        )

    plan = enrich_plan_with_sources(plan, dossier)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(description="Tradeoff Layer AI Scriptwriter + Shorts Generator")
    ap.add_argument("--dossier", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--backend", choices=["auto", "gemini", "openai", "transformers"], default="auto")
    ap.add_argument("--model", default="")
    ap.add_argument("--long-scenes", type=int, default=10)
    args = ap.parse_args()

    plan = generate_production_plan(
        dossier_path=args.dossier,
        output_path=args.output,
        backend=args.backend,
        model=args.model,
        target_long_scenes=args.long_scenes,
    )
    print(
        f"[script_generator] Generated '{plan.get('title')}' "
        f"({len(plan.get('long_scenes', []))} long scenes, {len(plan.get('shorts', []))} shorts) -> {args.output}"
    )


if __name__ == "__main__":
    main()
