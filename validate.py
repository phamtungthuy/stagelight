"""Validate predicted effects against constraints and compare with groundtruth."""

import json
import sys
import os
from collections import Counter

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


# Valid enum ranges
VALID_RANGES = {
    "MotionEffect": range(0, 7),
    "ColorEffect": range(0, 4),
    "IntensityEffect": range(0, 5),
    "VfxEffect": range(0, 3),
    "groupLightKey": range(0, 7),
}

# Constraint rules per group type
VFX_KEYS = {0, 1}
SINGLE_KEYS = {2, 3}
MULTI_KEYS = {4, 5, 6}
SINGLE_MOTIONS = {0, 3, 4, 5, 6}  # None + Wave, Rotate, Circle_Rotate, PingPong
MULTI_MOTIONS = {0, 1, 2}          # None + LaserCone, LaserFan


def validate_group_light(gl, beat_idx):
    """Validate a single groupLight object. Returns list of error messages."""
    errors = []
    key = gl.get("groupLightKey")

    # Check required fields
    for field in ["groupLightKey", "MotionEffect", "ColorEffect", "IntensityEffect", "VfxEffect"]:
        if field not in gl:
            errors.append(f"Beat {beat_idx}: missing field '{field}'")
            return errors

    # Check enum ranges
    for field, valid in VALID_RANGES.items():
        if gl[field] not in valid:
            errors.append(f"Beat {beat_idx}, key {key}: {field}={gl[field]} out of range")

    # VFX group rules
    if key in VFX_KEYS:
        if gl["MotionEffect"] != 0:
            errors.append(f"Beat {beat_idx}, VFX key {key}: MotionEffect should be 0, got {gl['MotionEffect']}")
        if gl["ColorEffect"] != 0:
            errors.append(f"Beat {beat_idx}, VFX key {key}: ColorEffect should be 0, got {gl['ColorEffect']}")
        if gl["IntensityEffect"] != 0:
            errors.append(f"Beat {beat_idx}, VFX key {key}: IntensityEffect should be 0, got {gl['IntensityEffect']}")
        if gl["VfxEffect"] == 0:
            errors.append(f"Beat {beat_idx}, VFX key {key}: VfxEffect should not be 0")

    # Single light group rules
    if key in SINGLE_KEYS:
        if gl["MotionEffect"] not in SINGLE_MOTIONS:
            errors.append(f"Beat {beat_idx}, Single key {key}: MotionEffect={gl['MotionEffect']} not recommended (use 3-6)")
        if gl["VfxEffect"] != 0:
            errors.append(f"Beat {beat_idx}, Single key {key}: VfxEffect should be 0, got {gl['VfxEffect']}")

    # Multi light group rules
    if key in MULTI_KEYS:
        if gl["MotionEffect"] not in MULTI_MOTIONS:
            errors.append(f"Beat {beat_idx}, Multi key {key}: MotionEffect={gl['MotionEffect']} not recommended (use 1-2)")
        if gl["VfxEffect"] != 0:
            errors.append(f"Beat {beat_idx}, Multi key {key}: VfxEffect should be 0, got {gl['VfxEffect']}")

    return errors


def validate_file(filepath):
    """Validate an effects JSON file. Returns (errors, stats)."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    beats = data["beats"]
    all_errors = []
    stats = {
        "total_beats": len(beats),
        "beats_with_effects": 0,
        "beats_empty": 0,
        "groupKey_usage": Counter(),
        "motion_usage": Counter(),
        "color_usage": Counter(),
        "intensity_usage": Counter(),
        "vfx_usage": Counter(),
    }

    for i, beat in enumerate(beats):
        gls = beat.get("groupLights", [])
        if gls:
            stats["beats_with_effects"] += 1
        else:
            stats["beats_empty"] += 1

        for gl in gls:
            errors = validate_group_light(gl, i)
            all_errors.extend(errors)
            if not errors:
                stats["groupKey_usage"][gl["groupLightKey"]] += 1
                stats["motion_usage"][gl["MotionEffect"]] += 1
                stats["color_usage"][gl["ColorEffect"]] += 1
                stats["intensity_usage"][gl["IntensityEffect"]] += 1
                stats["vfx_usage"][gl["VfxEffect"]] += 1

    return all_errors, stats


def print_stats(label, stats):
    """Print formatted stats."""
    total = stats["total_beats"]
    with_fx = stats["beats_with_effects"]
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Total beats:        {total}")
    print(f"  With effects:       {with_fx} ({with_fx/total*100:.1f}%)")
    print(f"  Empty:              {stats['beats_empty']}")
    print(f"  GroupKey usage:     {dict(sorted(stats['groupKey_usage'].items()))}")
    print(f"  Motion usage:       {dict(sorted(stats['motion_usage'].items()))}")
    print(f"  Color usage:        {dict(sorted(stats['color_usage'].items()))}")
    print(f"  Intensity usage:    {dict(sorted(stats['intensity_usage'].items()))}")
    print(f"  VFX usage:          {dict(sorted(stats['vfx_usage'].items()))}")


def compare(predicted_path, groundtruth_path):
    """Compare predicted effects with groundtruth."""
    errors_p, stats_p = validate_file(predicted_path)
    errors_g, stats_g = validate_file(groundtruth_path)

    print_stats("PREDICTED", stats_p)
    print_stats("GROUNDTRUTH", stats_g)

    if errors_p:
        print(f"\n⚠ Predicted has {len(errors_p)} constraint violations:")
        for e in errors_p[:10]:
            print(f"  - {e}")
        if len(errors_p) > 10:
            print(f"  ... and {len(errors_p) - 10} more")
    else:
        print("\n✅ Predicted passes all constraint checks!")

    # Coverage comparison
    p_pct = stats_p["beats_with_effects"] / stats_p["total_beats"] * 100
    g_pct = stats_g["beats_with_effects"] / stats_g["total_beats"] * 100
    print(f"\n  Effect coverage: predicted={p_pct:.1f}% vs groundtruth={g_pct:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate.py <effects.json> [groundtruth.json]")
        sys.exit(1)

    filepath = sys.argv[1]
    errors, stats = validate_file(filepath)
    print_stats(filepath, stats)

    if errors:
        print(f"\n⚠ {len(errors)} constraint violations:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\n✅ All constraints satisfied!")

    if len(sys.argv) >= 3:
        print("\n--- COMPARISON ---")
        compare(sys.argv[1], sys.argv[2])
