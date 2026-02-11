"""Audio analysis: extract per-beat features from mp3 and transcribe lyrics.

Results are cached in cache/ folder to avoid re-processing.
"""

import os
import json
import numpy as np
import librosa
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = "cache"


def _get_cache_path(mp3_path, suffix):
    """Get cache file path for a given mp3 and data type."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    song_name = os.path.splitext(os.path.basename(mp3_path))[0]
    return os.path.join(CACHE_DIR, f"{song_name}_{suffix}.json")


def _load_cache(cache_path):
    """Load cached data if exists."""
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_cache(cache_path, data):
    """Save data to cache."""
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def analyze_audio(mp3_path, beats):
    """Extract audio features for each beat segment (cached).

    Returns list of dicts with per-beat features:
        - energy: RMS energy (0-1 normalized)
        - energy_label: "low" / "medium" / "high"
        - bass: low frequency energy ratio
        - mid: mid frequency energy ratio
        - high: high frequency energy ratio
        - dominant_band: "bass" / "mid" / "high"
        - brightness: spectral centroid (normalized)
    """
    cache_path = _get_cache_path(mp3_path, "features")
    cached = _load_cache(cache_path)
    if cached and len(cached) == len(beats):
        print(f"  Audio features loaded from cache ({len(cached)} beats)")
        return cached

    print(f"  Analyzing audio: {mp3_path}")
    y, sr = librosa.load(mp3_path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)

    # Compute features over the full track
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # Band boundaries
    low_mask = freqs < 250
    mid_mask = (freqs >= 250) & (freqs < 2000)
    high_mask = freqs >= 2000

    # RMS energy per frame
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    rms_max = rms.max() if rms.max() > 0 else 1.0

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
    centroid_max = centroid.max() if centroid.max() > 0 else 1.0

    beat_features = []
    for beat in beats:
        t_start = beat["time"]
        t_end = min(t_start + beat["duration"], duration)

        # Convert time to frame indices
        f_start = librosa.time_to_frames(t_start, sr=sr, hop_length=512)
        f_end = librosa.time_to_frames(t_end, sr=sr, hop_length=512)
        f_end = max(f_end, f_start + 1)  # at least 1 frame

        # RMS energy for this beat
        beat_rms = rms[f_start:f_end].mean() if f_start < len(rms) else 0
        energy_norm = float(beat_rms / rms_max)

        # Band energies
        beat_spec = S[:, f_start:f_end]
        if beat_spec.size > 0:
            total_e = beat_spec.sum() + 1e-10
            bass_e = float(beat_spec[low_mask].sum() / total_e)
            mid_e = float(beat_spec[mid_mask].sum() / total_e)
            high_e = float(beat_spec[high_mask].sum() / total_e)
        else:
            bass_e, mid_e, high_e = 0.33, 0.33, 0.34

        # Dominant band
        band_vals = {"bass": bass_e, "mid": mid_e, "high": high_e}
        dominant = max(band_vals, key=lambda k: band_vals[k])

        # Energy label
        if energy_norm < 0.3:
            energy_label = "low"
        elif energy_norm < 0.65:
            energy_label = "medium"
        else:
            energy_label = "high"

        # Spectral centroid (brightness)
        beat_centroid = centroid[f_start:f_end].mean() if f_start < len(centroid) else 0
        brightness = float(beat_centroid / centroid_max)

        beat_features.append({
            "energy": round(energy_norm, 3),
            "energy_label": energy_label,
            "bass": round(bass_e, 2),
            "mid": round(mid_e, 2),
            "high": round(high_e, 2),
            "dominant_band": dominant,
            "brightness": round(brightness, 2),
        })

    _save_cache(cache_path, beat_features)
    print(f"  Extracted & cached features for {len(beat_features)} beats")
    return beat_features


def transcribe_lyrics(mp3_path):
    """Transcribe lyrics from mp3 using OpenAI Whisper API (cached).

    Returns list of segments: [{"start": float, "end": float, "text": str}, ...]
    """
    cache_path = _get_cache_path(mp3_path, "lyrics")
    cached = _load_cache(cache_path)
    if cached is not None:
        print(f"  Lyrics loaded from cache ({len(cached)} segments)")
        return cached

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  Skipping lyrics: no OPENAI_API_KEY")
        return []

    print(f"  Transcribing lyrics: {mp3_path}")
    client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")

    with open(mp3_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    segments = []
    if hasattr(response, "segments") and response.segments:
        for seg in response.segments:
            segments.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            })

    _save_cache(cache_path, segments)
    print(f"  Transcribed & cached {len(segments)} lyric segments")
    return segments


def get_lyrics_for_beats(beats, lyrics_segments):
    """Map lyrics to beats based on time overlap."""
    beat_lyrics = []
    for beat in beats:
        t_start = beat["time"]
        t_end = t_start + beat["duration"]
        texts = []
        for seg in lyrics_segments:
            # Check overlap
            if seg["end"] > t_start and seg["start"] < t_end:
                texts.append(seg["text"])
        beat_lyrics.append(" ".join(texts) if texts else "")
    return beat_lyrics
