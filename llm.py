"""LLM client for stage light effect prediction using OpenAI API."""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are an expert stage lighting designer for virtual concert experiences.
You follow industry conventions used by professional lighting designers, visual directors, and pyro techs.

# LIGHT GROUP TYPES
- groupLightKey 0, 1 → VFX_GROUP: Special effects (fireworks, smoke, CO2, pyro)
- groupLightKey 2, 3 → SINGLE_LIGHT_GROUP: Individual moving head lights
- groupLightKey 4, 5, 6 → MULTI_LIGHT_GROUP: Synchronized beam arrays, lasers

# EFFECT ENUMS
MotionEffectType: 0=None, 1=LaserCone, 2=LaserFan, 3=Wave, 4=Rotate, 5=Circle_Rotate, 6=PingPong
ColorEffectType: 0=None, 1=StaticColor, 2=RandomPerBeam, 3=PingPongColor
IntensityEffectType: 0=None, 1=SpectrumBased, 2=PingPongIntensity, 3=AlternatingBeams, 4=WaveIntensity
VfxEffectType: 0=None, 1=VFX_Simultaneous, 2=VFX_Wave

# HARD CONSTRAINTS (MUST enforce)
1. VFX_GROUP (key 0,1): ONLY use VfxEffect (1 or 2). MotionEffect, ColorEffect, IntensityEffect MUST be 0.
2. SINGLE_LIGHT_GROUP (key 2,3): Use Motion 3-6 (Wave, Rotate, Circle_Rotate, PingPong). VfxEffect MUST be 0.
3. MULTI_LIGHT_GROUP (key 4,5,6): Use Motion 1-2 (LaserCone, LaserFan). VfxEffect MUST be 0.
4. Value 0 = effect OFF.
5. Light groups (key 2-6) should have ColorEffect and IntensityEffect set when MotionEffect is active.

# PROFESSIONAL LIGHTING RULES (follow these conventions)

## Beat & Bar Rules
- Sync on beat 1 (downbeat) = strongest moment. Use for dramatic changes.
- Beat 1 & 3: strong (on-beat). Beat 2 & 4: lighter, build-up (off-beat/snare).
- Follow 4-bar / 8-bar phrasing. Major changes at phrase boundaries.
- Do NOT sync every single beat. Aim for 60-70% sync rate. Leave 30-40% without effects for breathing room.

## Song Structure Awareness (use lyrics/energy cues to detect)
- Intro: dark, subtle, cool colors. Build tension. No strobe.
- Verse: gentle wash, warm/neutral colors. Support vocals, don't overwhelm.
- Pre-chorus / Build-up: gradually increase intensity, add movement, speed up.
- Drop / Chorus: FULL ON. Fast beams, VFX, bright/vivid colors. Peak energy.
- Bridge / Breakdown: sudden contrast. Dark or spotlight only. Create rest.
- Outro: fade intensity gradually. Return to calm.

## Key Pitfalls to AVOID
- NO strobe/flash on every beat → looks robotic, causes fatigue.
- NO heavy VFX (pyro/CO2) on calm/ballad sections → destroys emotion.
- NO constant max intensity → need contrast (light vs dark, fast vs slow).
- Keep VFX for drops, beat 1, or key lyric moments ONLY. Never random.

## Best Practices
- "Less is more" at the start → build gradually, don't overwhelm from beat 1.
- Contrast is key → alternate bright/dark, fast/slow, warm/cool colors.
- Lasers sync with high-frequency melody lines.
- Bass hits → Wave, Rotate (strong motion).

# OUTPUT FORMAT
Return ONLY a valid JSON object with TWO fields:
1. "reasoning": A short explanation (2-3 sentences) of WHY you chose these effects or why this beat should be empty. Reference the song position, audio cues, and professional rules.
2. "effects": The array of groupLight objects, or [] for breathing room.

Example with effects:
{"reasoning": "At ~50% this is likely chorus/drop with high energy. Using LaserCone on multi-light group for peak impact, with vivid colors.", "effects": [{"groupLightKey": 4, "MotionEffect": 1, "ColorEffect": 2, "IntensityEffect": 3, "VfxEffect": 0}]}

Example empty (breathing room):
{"reasoning": "Beat 2 of a verse section with low energy. Following the 60-70% sync rule, this beat provides breathing room.", "effects": []}

Return ONLY the JSON object, no markdown."""


def create_client():
    """Create OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in .env")
    base_url = os.getenv("OPENAI_BASE_URL", None)
    return OpenAI(api_key=api_key, base_url=base_url)


def _build_beat_entry(beat, audio_features, beat_lyrics, idx):
    """Build a single beat entry dict for the prompt."""
    entry = {
        "time": round(beat["time"], 2),
        "duration": round(beat["duration"], 3),
    }
    if audio_features and idx < len(audio_features):
        af = audio_features[idx]
        entry["energy"] = af["energy_label"]
        entry["dominant_band"] = af["dominant_band"]
        entry["brightness"] = af["brightness"]
    if beat_lyrics and idx < len(beat_lyrics) and beat_lyrics[idx]:
        entry["lyrics"] = beat_lyrics[idx]
    return entry


def build_single_beat_prompt(beat_idx, beats, audio_features=None, beat_lyrics=None,
                              window_size=2):
    """Build prompt for predicting effects of a single beat with sliding window context."""
    total = len(beats)

    # Context window: previous and next beats
    ctx_before = []
    for i in range(max(0, beat_idx - window_size), beat_idx):
        entry = _build_beat_entry(beats[i], audio_features, beat_lyrics, i)
        entry["position"] = "before"
        ctx_before.append(entry)

    ctx_after = []
    for i in range(beat_idx + 1, min(total, beat_idx + window_size + 1)):
        entry = _build_beat_entry(beats[i], audio_features, beat_lyrics, i)
        entry["position"] = "after"
        ctx_after.append(entry)

    # Target beat
    target = _build_beat_entry(beats[beat_idx], audio_features, beat_lyrics, beat_idx)

    # Calculate time gaps and position info
    time_prev = beats[beat_idx]["time"] - beats[beat_idx - 1]["time"] if beat_idx > 0 else 999
    time_next = beats[beat_idx + 1]["time"] - beats[beat_idx]["time"] if beat_idx < total - 1 else 999
    is_rapid = time_prev < 0.6 or time_next < 0.6

    target["gap_prev"] = round(time_prev, 3)
    target["gap_next"] = round(time_next, 3)
    target["beat_position"] = f"{beat_idx + 1}/{total}"

    # Estimate position in song (rough %)
    pct = beat_idx / max(total - 1, 1) * 100
    if pct < 10:
        section_hint = "likely INTRO (build tension, subtle)"
    elif pct < 30:
        section_hint = "likely VERSE (support vocals, moderate)"
    elif pct < 45:
        section_hint = "likely PRE-CHORUS / BUILD-UP (rising intensity)"
    elif pct < 65:
        section_hint = "likely CHORUS / DROP (peak energy, go big)"
    elif pct < 80:
        section_hint = "likely VERSE 2 or BRIDGE (contrast, breathing room)"
    elif pct < 90:
        section_hint = "likely FINAL CHORUS (climax, full effects)"
    else:
        section_hint = "likely OUTRO (fading, calm down)"

    prompt = (
        f"Predict light effects for beat {beat_idx + 1}/{total} in a song.\n"
        f"Song position: ~{pct:.0f}% → {section_hint}\n\n"
        f"Audio cues:\n"
        f"- HIGH energy → dramatic effects, multiple groups, VFX on drops\n"
        f"- LOW energy → empty [] or subtle wash. Do NOT force effects on calm moments.\n"
        f"- Bass-dominant → strong motion (Wave, Rotate)\n"
        f"- High-dominant → lasers/beams (LaserCone, LaserFan, PingPong)\n"
        f"- Has lyrics → color/intensity to support vocals, don't overwhelm\n"
        f"- No lyrics → more freedom for motion and VFX\n\n"
        f"Pro rules:\n"
        f"- Sync ~60-70% of beats. The rest should be empty [] for breathing room.\n"
        f"- VFX (key 0,1) ONLY on drops/climax/key moments. Never random.\n"
        f"- Build gradually. Less at start, peak at chorus.\n"
    )

    if is_rapid:
        prompt += f"- RAPID FIRE: gap < 0.6s, keep same group as neighbors or use PingPong sequence.\n"

    prompt += "\n"

    if ctx_before:
        prompt += f"Previous beats: {json.dumps(ctx_before)}\n"
    prompt += f">>> TARGET: {json.dumps(target)} <<<\n"
    if ctx_after:
        prompt += f"Next beats: {json.dumps(ctx_after)}\n"

    prompt += (
        f'\nReturn {{"effects": [...]}} with groupLight objects, or {{"effects": []}} if this beat should be silent.'
    )
    return prompt


import re
from validate import validate_group_light

WINDOW_SIZE = 10


def _call_llm_single(client, model, beat_idx, beats, audio_features, beat_lyrics,
                      window_size=2, max_retries=3):
    """Call LLM to predict effects for a single beat with sliding window context."""
    base_prompt = build_single_beat_prompt(
        beat_idx, beats, audio_features, beat_lyrics, window_size
    )

    error_feedback = None

    for attempt in range(max_retries):
        try:
            user_prompt = base_prompt
            if error_feedback:
                user_prompt += (
                    f"\n\nYour previous response had these constraint ERRORS, fix them:\n"
                    f"{error_feedback}\n"
                    f"VFX groups (key 0,1): ONLY VfxEffect. "
                    f"Single (key 2,3): Motion 3-6. Multi (key 4,5,6): Motion 1-2."
                )

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
            )

            content = response.choices[0].message.content
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r'\{[\s\S]*\}', content)
                if match:
                    parsed = json.loads(match.group())
                else:
                    continue

            # Extract reasoning and effects
            reasoning = ""
            if isinstance(parsed, dict):
                reasoning = parsed.get("reasoning", "")
                effects = parsed.get("effects", [])
                if not isinstance(effects, list):
                    for v in parsed.values():
                        if isinstance(v, list):
                            effects = v
                            break
                    else:
                        effects = []
            elif isinstance(parsed, list):
                effects = parsed
            else:
                effects = []

            # Normalize: ensure all items are dicts
            normalized = []
            for item in effects:
                if isinstance(item, dict):
                    normalized.append(item)
                elif isinstance(item, list):
                    normalized.extend(x for x in item if isinstance(x, dict))
            effects = normalized

            # Validate constraints
            all_errors = []
            for gl in effects:
                errors = validate_group_light(gl, beat_idx)
                all_errors.extend(errors)

            if all_errors:
                error_feedback = "\n".join(all_errors[:5])
                continue

            return effects, reasoning

        except Exception:
            continue

    # Fallback: return empty effects if all retries fail
    print(f"    Beat {beat_idx}: failed after {max_retries} attempts, using empty []")
    return [], ""


def predict_effects(beats, model=None, max_retries=3,
                     audio_features=None, beat_lyrics=None,
                     max_concurrent=5, window_size=2):
    """Call LLM to predict groupLights per beat with sliding window context.

    Args:
        beats: List of beat dicts with 'time' and 'duration'
        model: Model name override
        max_retries: Number of retries on parse failure per beat
        audio_features: Per-beat audio features from audio_analysis
        beat_lyrics: Per-beat lyrics text
        max_concurrent: Max concurrent LLM requests
        window_size: Number of neighboring beats for context (each side)

    Returns:
        tuple: (effects_list, reasonings_list) — one per beat
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Semaphore

    client = create_client()
    model = model or os.getenv("MODEL_NAME", "gpt-4o-mini")
    total = len(beats)
    print(f"  Predicting effects for {total} beats (window={window_size}, concurrent={max_concurrent})")

    semaphore = Semaphore(max_concurrent)
    completed = [0]  # mutable counter for progress

    def process_beat(beat_idx):
        with semaphore:
            effects, reasoning = _call_llm_single(
                client, model, beat_idx, beats, audio_features, beat_lyrics,
                window_size=window_size, max_retries=max_retries,
            )
            completed[0] += 1
            has_fx = "✓" if effects else "·"
            print(f"    [{completed[0]}/{total}] Beat {beat_idx} {has_fx}")
            return beat_idx, effects, reasoning

    # Process all beats concurrently
    all_effects: list = [None] * total
    all_reasonings: list = [None] * total
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [executor.submit(process_beat, i) for i in range(total)]
        for future in as_completed(futures):
            beat_idx, effects, reasoning = future.result()
            all_effects[beat_idx] = effects
            all_reasonings[beat_idx] = reasoning

    # Replace any None with defaults (safety)
    all_effects = [r if r is not None else [] for r in all_effects]
    all_reasonings = [r if r is not None else "" for r in all_reasonings]

    with_fx = sum(1 for r in all_effects if r)
    print(f"\n  Done! {with_fx}/{total} beats have effects ({with_fx/total*100:.1f}%)")
    return all_effects, all_reasonings
