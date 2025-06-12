












#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_diarize.py

This script extracts audio, performs speaker diarization, and splits speaker segments into fixed 0.2s intervals.
Modified to assign speaker IDs in order of first appearance time.
"""

import os
import sys
import subprocess
import json
import torch
from pyannote.audio import Pipeline

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

INPUT_VIDEO   = r"D:\LLMS\PROJECT_2\modified_zoom\output_grid_with_audio_1_edited.mp4"
SEGMENTS_FILE = "updated_segments_output_grid.json"

# HF_TOKEN for PyAnnote (set as env var or hard-code here)
HF_TOKEN = os.getenv("HF_TOKEN", "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")
if not HF_TOKEN:
    print("❌ ERROR: please set HF_TOKEN as an environment variable or hard-code it above.", file=sys.stderr)
    sys.exit(1)

CHUNK_DURATION = 0.5  # seconds (fixed chunk length for SyncNet)
NUM_SPEAKERS = 3      # expected number of speakers in the video

# ── STEP 1: Extract audio from the video (16 kHz mono) ────────────────────────
print("🔊 Extracting audio from video…")
EXTRACTED_AUDIO = "extracted_audio.wav"
try:
    subprocess.run([
        "ffmpeg", "-y",
        "-fflags", "+genpts", "-copyts",
        "-i", INPUT_VIDEO,
        "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-start_at_zero",
        EXTRACTED_AUDIO
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    print(f"✔ Audio saved → {EXTRACTED_AUDIO}")
except subprocess.CalledProcessError as e:
    print("❌ FFmpeg failed to extract audio:", e, file=sys.stderr)
    sys.exit(1)

# ── STEP 2: Run PyAnnote speaker diarization ─────────────────────────────────
print("🔍 Loading PyAnnote speaker-diarization pipeline…")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)
# If GPU is available, use it for faster processing
if torch.cuda.is_available():
    # Create a torch.device object from the string "cuda"
    device = torch.device("cuda")
    pipeline.to(device)
    print("✔ Using GPU for diarization.")
else:
    print("⚠️  GPU not available, running on CPU.")

print(f"🔍 Running diarization with expected {NUM_SPEAKERS} speakers…")
annotation = pipeline(EXTRACTED_AUDIO, num_speakers=NUM_SPEAKERS)

# Collect coarse segments with original speaker labels
coarse_segments = []
for segment, _, label in annotation.itertracks(yield_label=True):
    coarse_segments.append([round(segment.start, 3), round(segment.end, 3), label])

# **Sort coarse segments by start time to ensure chronological order**
coarse_segments.sort(key=lambda x: x[0])

# **Assign new speaker IDs based on first appearance time**
speaker_id_map = {}
sorted_coarse_segments = []
for (start, end, label) in coarse_segments:
    if label not in speaker_id_map:
        # Assign the next available ID (0,1,2,...) when a new label is encountered
        speaker_id_map[label] = len(speaker_id_map)
    new_id = speaker_id_map[label]
    sorted_coarse_segments.append([start, end, new_id])

# Replace coarse_segments with the sorted version containing new IDs
coarse_segments = sorted_coarse_segments
detected_speakers = len(speaker_id_map)
print(f"✔ Detected {len(coarse_segments)} coarse segments across {detected_speakers} speakers (IDs sorted by first appearance).")

if detected_speakers != NUM_SPEAKERS:
    print(f"⚠️  WARNING: Expected {NUM_SPEAKERS} speakers but detected {detected_speakers}.")
    print("   Check the audio quality or adjust NUM_SPEAKERS if this seems incorrect.")

# ── STEP 3: Split each coarse segment into fixed 0.2 s chunks ────────────────
def split_into_fixed_chunks(segments, chunk_duration):
    """
    Split each [t_start, t_end, speaker_id] segment into sub-intervals of length chunk_duration.
    The last segment is padded to exactly chunk_duration if there's a leftover.
    """
    fixed_segments = []
    for (t_start, t_end, spk_id) in segments:
        total_dur = t_end - t_start
        if total_dur <= 0:
            continue
        num_full = int(total_dur // chunk_duration)
        # Add all full-duration chunks
        for i in range(num_full):
            s = t_start + i * chunk_duration
            e = s + chunk_duration
            fixed_segments.append([round(s, 3), round(e, 3), spk_id])
        # Handle the leftover tail (if any) by padding it to chunk_duration
        tail_start = t_start + num_full * chunk_duration
        if tail_start < t_end:
            s = tail_start
            e = s + chunk_duration
            fixed_segments.append([round(s, 3), round(e, 3), spk_id])
    return fixed_segments

print("🔄 Splitting coarse segments into fixed 0.2 s chunks…")
fine_segments = split_into_fixed_chunks(coarse_segments, CHUNK_DURATION)
print(f"✔ Generated {len(fine_segments)} fine-grained segments (each {CHUNK_DURATION:.1f} s).")

# ── STEP 4: Save fine-grained segments to JSON ───────────────────────────────
with open(SEGMENTS_FILE, "w") as f:
    json.dump(fine_segments, f, indent=2)
print(f"✔ Wrote fine-grained segments → {SEGMENTS_FILE}")
