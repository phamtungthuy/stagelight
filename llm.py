"""LLM client for stage light effect prediction using OpenAI API."""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are an expert stage lighting designer for virtual concert experiences.
Given a list of music beats (with timestamp and duration), you must assign light effects to create a visually stunning and musically coherent light show.

# LIGHT GROUP TYPES
- groupLightKey 0, 1 → VFX_GROUP: Special effects (fireworks, smoke, etc.)
- groupLightKey 2, 3 → SINGLE_LIGHT_GROUP: Individual lights, flexible movement
- groupLightKey 4, 5, 6 → MULTI_LIGHT_GROUP: Multiple lights, synchronized patterns

# EFFECT ENUMS

MotionEffectType: 0=None, 1=LaserCone, 2=LaserFan, 3=Wave, 4=Rotate, 5=Circle_Rotate, 6=PingPong
ColorEffectType: 0=None, 1=StaticColor, 2=RandomPerBeam, 3=PingPongColor
IntensityEffectType: 0=None, 1=SpectrumBased, 2=PingPongIntensity, 3=AlternatingBeams, 4=WaveIntensity
VfxEffectType: 0=None, 1=VFX_Simultaneous, 2=VFX_Wave

# CONSTRAINTS (MUST follow)
1. VFX_GROUP (key 0,1): ONLY use VfxEffect (1 or 2). MotionEffect, ColorEffect, IntensityEffect MUST be 0.
2. SINGLE_LIGHT_GROUP (key 2,3): Use Motion 3-6 (Wave, Rotate, Circle_Rotate, PingPong). VfxEffect MUST be 0.
3. MULTI_LIGHT_GROUP (key 4,5,6): Use Motion 1-2 (LaserCone, LaserFan). VfxEffect MUST be 0.
4. Value 0 in any field = effect is OFF.
5. Light groups (key 2-6) should have ColorEffect and IntensityEffect set when MotionEffect is active.

# ARTISTIC GUIDELINES
- EVERY beat MUST have at least one groupLight with ACTIVE effects. NEVER leave a beat blank or with all-zero values.
- Short beats (<1s): simpler effects, fewer groups (1-2 groups max)
- Long beats (>3s): can use more groups, more complex combinations
- Vary the effects — avoid repeating the same combination for consecutive beats.
- Use VFX groups (key 0,1) sparingly for dramatic moments / accents.
- Build intensity: start simpler, escalate over time, peak at climactic moments.
- Create patterns: alternating between different light groups adds visual interest.
- Low energy beats: use subtle effects like gentle Wave or soft StaticColor, NOT blank.

# OUTPUT FORMAT
Return ONLY a valid JSON object with key "effects" containing an array of groupLight objects.
Each groupLight must have at least one non-zero effect value (MotionEffect, ColorEffect, IntensityEffect, or VfxEffect).

Example: {"effects": [{"groupLightKey": 4, "MotionEffect": 1, "ColorEffect": 2, "IntensityEffect": 3, "VfxEffect": 0}]}

Return ONLY the JSON object, no markdown, no explanation."""


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

    prompt = (
        f"Predict light effects for beat {beat_idx + 1}/{total} in a song.\n\n"
        f"Use the audio features to guide your choice:\n"
        f"- HIGH energy → more groups, dramatic effects, VFX accents\n"
        f"- LOW energy → subtle effects (gentle Wave, soft StaticColor)\n"
        f"- Bass-dominant → strong motion (Wave, Rotate)\n"
        f"- High-dominant → sharp effects (LaserCone, LaserFan, PingPong)\n"
        f"- Vocal (has lyrics) → complement with color/intensity\n"
        f"- Instrumental (no lyrics) → more motion and VFX freedom\n"
        f"- EVERY beat MUST have active effects. No blank beats.\n\n"
    )

    if ctx_before:
        prompt += f"Previous beats (context): {json.dumps(ctx_before)}\n"
    prompt += f">>> TARGET BEAT to predict: {json.dumps(target)} <<<\n"
    if ctx_after:
        prompt += f"Next beats (context): {json.dumps(ctx_after)}\n"

    prompt += (
        f"\nReturn a JSON object: {{\"effects\": [...]}} with at least 1 groupLight with ACTIVE effects. "
        f"Every groupLight must have at least one non-zero effect. NEVER return blank/empty."
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

            # Extract effects array
            if isinstance(parsed, dict):
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

            return effects

        except Exception:
            continue

    # Fallback: return empty effects if all retries fail
    print(f"    Beat {beat_idx}: failed after {max_retries} attempts, using empty []")
    return []


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
        List of groupLights arrays (one per beat)
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
            result = _call_llm_single(
                client, model, beat_idx, beats, audio_features, beat_lyrics,
                window_size=window_size, max_retries=max_retries,
            )
            completed[0] += 1
            has_fx = "✓" if result else "·"
            print(f"    [{completed[0]}/{total}] Beat {beat_idx} {has_fx}")
            return beat_idx, result

    # Process all beats concurrently
    all_results = [None] * total
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [executor.submit(process_beat, i) for i in range(total)]
        for future in as_completed(futures):
            beat_idx, result = future.result()
            all_results[beat_idx] = result

    # Replace any None with empty list (safety)
    all_results = [r if r is not None else [] for r in all_results]

    with_fx = sum(1 for r in all_results if r)
    print(f"\n  Done! {with_fx}/{total} beats have effects ({with_fx/total*100:.1f}%)")
    return all_results
