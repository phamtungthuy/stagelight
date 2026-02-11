"""Stage Light Effect Prediction Pipeline.

Usage:
    python main.py --song 24KMagic         # Predict effects for one song
    python main.py --all                    # Predict effects for all songs
    python main.py --song 24KMagic --compare  # Predict + compare with groundtruth
    python main.py --validate effects/TimeLine_24KMagic.json  # Validate only
    python main.py --song 24KMagic --no-lyrics  # Skip lyrics transcription
"""

import argparse
import json
import os
import sys

from llm import predict_effects
from validate import validate_file, print_stats, compare
from audio_analysis import analyze_audio, transcribe_lyrics, get_lyrics_for_beats


BEATS_DIR = "beats"
EFFECTS_DIR = "effects"
MUSICS_DIR = "musics"


def load_beats(song_name):
    """Load beat data from beats/ folder."""
    path = os.path.join(BEATS_DIR, f"TimeLine_{song_name}.json")
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_mp3(song_name):
    """Find mp3 file for a song."""
    path = os.path.join(MUSICS_DIR, f"{song_name}.mp3")
    if os.path.exists(path):
        return path
    print(f"Warning: {path} not found, skipping audio analysis")
    return None


def save_effects(song_name, data):
    """Save predicted effects to effects/ folder."""
    os.makedirs(EFFECTS_DIR, exist_ok=True)
    path = os.path.join(EFFECTS_DIR, f"TimeLine_{song_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {path}")
    return path


def process_song(song_name, model=None, skip_lyrics=False, max_concurrent=5, window_size=2):
    """Predict effects for a single song."""
    print(f"\n{'='*50}")
    print(f"  Processing: {song_name}")
    print(f"{'='*50}")

    # Load beats
    data = load_beats(song_name)
    beats = data["beats"]
    print(f"  Loaded {len(beats)} beats")

    # Audio analysis
    audio_features = None
    beat_lyrics = None
    mp3_path = find_mp3(song_name)

    if mp3_path:
        audio_features = analyze_audio(mp3_path, beats)

        if not skip_lyrics:
            lyrics_segments = transcribe_lyrics(mp3_path)
            if lyrics_segments:
                beat_lyrics = get_lyrics_for_beats(beats, lyrics_segments)
                vocal_beats = sum(1 for l in beat_lyrics if l)
                print(f"  {vocal_beats}/{len(beats)} beats have lyrics")

    # Predict effects
    predicted_group_lights, reasonings = predict_effects(
        beats, model=model,
        audio_features=audio_features,
        beat_lyrics=beat_lyrics,
        max_concurrent=max_concurrent,
        window_size=window_size,
    )

    # Merge predictions into beat data (clean — no reasoning)
    for i, beat in enumerate(beats):
        beat["groupLights"] = predicted_group_lights[i]

    # Save clean effects file (for system)
    output_path = save_effects(song_name, data)

    # Save reasoning file (duplicate + reasoning field, for partner review)
    import copy
    reasoning_data = copy.deepcopy(data)
    for i, beat in enumerate(reasoning_data["beats"]):
        beat["reasoning"] = reasonings[i]
    reasoning_path = os.path.join(EFFECTS_DIR, f"TimeLine_{song_name}_reasoning.json")
    with open(reasoning_path, "w", encoding="utf-8") as f:
        json.dump(reasoning_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved reasoning: {reasoning_path}")

    # Validate
    errors, stats = validate_file(output_path)
    print_stats(f"Results: {song_name}", stats)

    if errors:
        print(f"\n⚠ {len(errors)} constraint violations:")
        for e in errors[:5]:
            print(f"  - {e}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    else:
        print("\n✅ All constraints satisfied!")

    return output_path


def get_all_songs():
    """Get list of all song names from beats/ folder."""
    songs = []
    for f in os.listdir(BEATS_DIR):
        if f.startswith("TimeLine_") and f.endswith(".json"):
            name = f.replace("TimeLine_", "").replace(".json", "")
            songs.append(name)
    return sorted(songs)


def main():
    parser = argparse.ArgumentParser(description="Stage Light Effect Prediction")
    parser.add_argument("--song", type=str, help="Song name (e.g. 24KMagic)")
    parser.add_argument("--all", action="store_true", help="Process all songs")
    parser.add_argument("--compare", action="store_true", help="Compare with groundtruth")
    parser.add_argument("--validate", type=str, help="Validate an effects file")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--no-lyrics", action="store_true", help="Skip lyrics transcription")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent LLM requests (default: 5)")
    parser.add_argument("--window-size", type=int, default=2, help="Context window size (default: 2 beats per side)")

    args = parser.parse_args()

    if args.validate:
        errors, stats = validate_file(args.validate)
        print_stats(args.validate, stats)
        if errors:
            print(f"\n⚠ {len(errors)} constraint violations:")
            for e in errors:
                print(f"  - {e}")
        else:
            print("\n✅ All constraints satisfied!")
        return

    if args.all:
        songs = get_all_songs()
        print(f"Processing {len(songs)} songs: {songs}")
        for song in songs:
            process_song(
                song, model=args.model, skip_lyrics=args.no_lyrics,
                max_concurrent=args.max_concurrent, window_size=args.window_size
            )
        return

    if args.song:
        output_path = process_song(
            args.song, model=args.model, skip_lyrics=args.no_lyrics,
            max_concurrent=args.max_concurrent, window_size=args.window_size
        )

        if args.compare:
            gt_path = f"TimeLine_{args.song}_groundtruth.json"
            if os.path.exists(gt_path):
                print(f"\n--- COMPARISON WITH GROUNDTRUTH ---")
                compare(output_path, gt_path)
            else:
                print(f"\n⚠ Groundtruth file not found: {gt_path}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
