


# #!/usr/bin/env python3
# """
# spotlight_ffmpeg_dynamic.py

# Centers each speaker in its own cell using column-major ordering:
# cell IDs increase down each column, then move right.
# Grid size computes to accommodate all speakers.
# """
# import os
# import json
# import subprocess
# import argparse
# import math
# import cv2
# import numpy as np

# # ── DEBUG PARAMETERS ─────────────────────────────────────────────────────
# DEBUG_DIR = "debug_frames"


# def load_segments(path):
#     with open(path, "r") as f:
#         return json.load(f)


# def compute_grid(n):
#     # Choose smallest grid (rows x cols) s.t. rows*cols >= n
#     cols = math.ceil(math.sqrt(n))
#     rows = math.ceil(n / cols)
#     return rows, cols


# def get_speaker_cell_map(segments):
#     # Unique sorted speaker IDs
#     speakers = sorted({seg[2] for seg in segments})
#     # Static mapping: speaker N -> cell N
#     return {spk: idx for idx, spk in enumerate(speakers)}, len(speakers)


# def build_filter_complex(segments, mapping, width, height, rows, cols):
#     cell_w = width  // cols
#     cell_h = height // rows
#     filters, tags = [], []
#     cnt = 0

#     for st, en, spk in segments:
#         if spk not in mapping:
#             continue
#         cell = mapping[spk]
#         # COLUMN-MAJOR: down then right
#         col = cell // rows
#         row = cell % rows
#         x = col * cell_w
#         y = row * cell_h

#         # Debug output
#         print(f"segment #{cnt}: speaker={spk}, cell={cell} -> row={row},col={col},crop=(x={x},y={y},w={cell_w},h={cell_h})")

#         vtag = f"[v{cnt}]"
#         filters.append(
#             f"[0:v]trim=start={st:.3f}:end={en:.3f},"
#             f"setpts=PTS-STARTPTS,crop={cell_w}:{cell_h}:{x}:{y},"
#             f"scale={width}:{height}:flags=lanczos,setsar=1{vtag}"
#         )
#         tags.append(vtag)

#         atag = f"[a{cnt}]"
#         filters.append(
#             f"[0:a]atrim=start={st:.3f}:end={en:.3f},asetpts=PTS-STARTPTS{atag}"
#         )
#         tags.append(atag)
#         cnt += 1

#     concat_in = "".join(tags)
#     filters.append(f"{concat_in}concat=n={cnt}:v=1:a=1[outv][outa]")
#     return ";".join(filters), cnt


# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("-i", "--input", default="output_grid_with_audio_1_edited.mp4")
#     p.add_argument("-s", "--segments", default="updated_segments_output_grid.json")
#     p.add_argument("-o", "--output", default="Zoom_centered_output_grid.mp4")
#     args = p.parse_args()

#     segments = load_segments(args.segments)
#     cap = cv2.VideoCapture(args.input)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cap.release()

#     mapping, n_speakers = get_speaker_cell_map(segments)
#     rows, cols = compute_grid(n_speakers)
#     print(f"Static column-major mapping: {mapping}, Grid: {rows} rows x {cols} cols")

#     fc, processed = build_filter_complex(segments, mapping, width, height, rows, cols)
#     print(f"Processing {processed}/{len(segments)} segments")

#     os.makedirs(DEBUG_DIR, exist_ok=True)
#     with open("filters.txt", "w") as f:
#         f.write(fc)
#     print("Wrote filter graph to filters.txt")

#     cmd = [
#         "ffmpeg", "-y", "-i", args.input,
#         "-filter_complex_script", "filters.txt",
#         "-map", "[outv]", "-map", "[outa]",
#         "-c:v", "libx264", "-preset", "medium", "-crf", "23",
#         "-c:a", "aac", "-b:a", "192k", "-vsync", "vfr", args.output
#     ]
#     print("Running FFmpeg...")
#     res = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
#     if res.returncode:
#         print("❌ FFmpeg error:\n", res.stderr)
#     else:
#         print(f"✔ Saved to {args.output}")

# if __name__ == "__main__":
#     main()



#final fully GPU based version fully working with ffmpeg

#!/usr/bin/env python3
"""
spotlight_ffmpeg_gpu.py

Process each speaker segment: CPU decode & crop → GPU upload & scale → GPU encode → concat.
"""
import os, sys, json, math, argparse, subprocess, tempfile, cv2

def load_segments(path):
    return json.load(open(path, "r"))

def compute_grid(n):
    cols = math.ceil(math.sqrt(n))
    return math.ceil(n/cols), cols

def get_map(segs):
    spk = sorted({s[2] for s in segs})
    return {sp: i for i, sp in enumerate(spk)}, len(spk)

def process_segment(ffmpeg, input_vid, st, en, x, y, cw, ch, W, H, idx, tmpdir):
    out_path = os.path.join(tmpdir, f"seg_{idx}.mp4")
    # CPU decode & crop, then GPU upload & scale, then GPU encode
    cmd = [
        ffmpeg, "-y",
        # seek and limit segment
        "-ss", f"{st}", "-to", f"{en}",
        "-i", input_vid,
        # crop on CPU, then upload to GPU and scale on GPU
        "-vf", f"crop={cw}:{ch}:{x}:{y},hwupload_cuda,scale_cuda=w={W}:h={H}",
        # encode video on GPU
        "-c:v", "h264_nvenc", "-preset", "p1", "-cq", "23",
        # audio passthrough or re-encode if needed
        "-c:a", "aac", "-b:a", "192k",
        out_path
    ]
    subprocess.run(cmd, check=True)
    return out_path

def run_ffmpeg_concat(ffmpeg, segment_files, output):
    list_file = os.path.join(os.path.dirname(output), "concat_list.txt")
    with open(list_file, "w") as f:
        for p in segment_files:
            f.write(f"file '{p}'\n")
    cmd = [
        ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_file,
        "-c", "copy", output
    ]
    subprocess.run(cmd, check=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i","--input", required=True)
    p.add_argument("-s","--segments", required=True)
    p.add_argument("-o","--output", required=True)
    args = p.parse_args()

    FFMPEG = r"D:\LLMS\PROJECT_2\modified_zoom\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe"
    if not os.path.isfile(FFMPEG):
        sys.exit("❌ ffmpeg not found at " + FFMPEG)

    # Load segments and video properties
    segs = load_segments(args.segments)
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened(): sys.exit("❌ cannot open input video")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Compute grid layout
    mapping, nsp = get_map(segs)
    rows, cols = compute_grid(nsp)
    cw, ch = W // cols, H // rows
    print(f"Grid layout: {rows}×{cols} for {nsp} speakers")

    # Process each segment separately
    tmpdir = tempfile.mkdtemp(prefix="ffmpeg_gpu_")
    seg_files = []
    for idx, (st, en, sp) in enumerate(segs):
        col, row = divmod(mapping[sp], rows)
        x, y = col * cw, row * ch
        print(f"Processing segment {idx+1}/{len(segs)}: speaker {sp} @ {st}-{en}s")
        seg_files.append(
            process_segment(FFMPEG, args.input, st, en, x, y, cw, ch, W, H, idx, tmpdir)
        )

    # Concatenate all segments into final output
    print("Merging segments…")
    run_ffmpeg_concat(FFMPEG, seg_files, args.output)
    print("✔ Done: output at", args.output)

if __name__ == "__main__": main()
