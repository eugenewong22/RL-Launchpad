"""Assemble the <=2 minute demo video from committed clips and generated
title cards (R4: every clip is rendered by record_video.py from the same
checkpoint the reported numbers come from, on the eval seeds).

Usage: uv run python scripts/make_demo.py [--eval-table results/final_eval_push.md]
Writes results/demo_final.mp4. Regenerating is idempotent - the cards read
their numbers from the eval table, so there is no second source of truth.

Cards are drawn with matplotlib rather than ffmpeg's drawtext because
drawtext needs a font path that differs per OS; matplotlib ships DejaVu,
so this produces byte-identical cards on a judge's machine.
"""

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

W = H = 480  # every clip is 480x480; cards match so concat needs no scaling
FPS = 25
BG = "#101418"
FG = "#f2f4f6"
ACCENT = "#4da3ff"
MUTED = "#8b98a5"

# Short display names sized for a 480px card - deliberately terser than the
# legend labels in make_plots.py, which have a full figure width to use.
CARD_LABELS = {
    "push_td3_her": "TD3+HER (ours)",
    "push_sb3_sac": "SAC+HER (SB3)",
    "push_sb3_her": "TD3+HER (SB3)",
    "push_td3_noher": "TD3, no HER (ablation)",
}


def render_card(path: Path, lines: list[tuple]):
    """Lay a block of lines out top-to-bottom, vertically centered.

    A 3-tuple (text, size, color) is one centered line. A 4-tuple
    (label, value, size, color) is a two-column row: label right-aligned
    and value left-aligned about the centre gutter, so the numbers line
    up in a column even though the arm names differ in length.
    """
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(BG)
    n = len(lines)
    # Spread the block around the vertical centre with even spacing.
    top, step = 0.5 + (n - 1) * 0.055, 0.11
    for i, line in enumerate(lines):
        y = top - i * step
        common = dict(va="center", family="DejaVu Sans")
        if len(line) == 4:
            label, value, size, color = line
            fig.text(0.48, y, label, ha="right", fontsize=size, color=color, **common)
            fig.text(0.52, y, value, ha="left", fontsize=size, color=FG, **common)
        else:
            text, size, color = line
            fig.text(0.5, y, text, ha="center", fontsize=size, color=color, **common)
    fig.savefig(path, facecolor=BG, dpi=100)
    plt.close(fig)


def parse_eval_table(path: Path) -> list[tuple[str, str]]:
    """Pull (arm, 'mean ± std') out of the final_eval.md markdown table."""
    rows = []
    for line in path.read_text().splitlines():
        m = re.match(r"\|\s*(\S+)\s*\|\s*\d+\s*\|\s*([\d.]+ ± [\d.]+)\s*\|", line)
        if m:
            rows.append((m.group(1), m.group(2)))
    # Best arm first so the card reads as a ranking.
    return sorted(rows, key=lambda r: -float(r[1].split()[0]))


def ffmpeg(args: list[str]):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def seg_from_card(png: Path, out: Path, dur: float):
    ffmpeg(["-loop", "1", "-i", str(png), "-t", f"{dur}",
            "-vf", f"scale={W}:{H},format=yuv420p", "-r", str(FPS),
            "-c:v", "libx264", "-preset", "veryfast", str(out)])


def seg_from_clip(clip: Path, out: Path, slowdown: float = 1.0):
    # setpts stretches presentation timestamps; 2.0 = half speed. Used on the
    # failure clip, which is only 2 s and unreadable at native rate.
    vf = f"setpts={slowdown}*PTS,scale={W}:{H},format=yuv420p"
    ffmpeg(["-i", str(clip), "-vf", vf, "-r", str(FPS),
            "-c:v", "libx264", "-preset", "veryfast", "-an", str(out)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--eval-table", default=None,
                        help="defaults to <results-dir>/final_eval_push.md")
    parser.add_argument("--out", default=None,
                        help="defaults to <results-dir>/demo_final.mp4")
    args = parser.parse_args()

    res = Path(args.results_dir)
    table_path = Path(args.eval_table) if args.eval_table else res / "final_eval_push.md"
    out_path = Path(args.out) if args.out else res / "demo_final.mp4"

    if not table_path.exists():
        raise SystemExit(
            f"{table_path} not found - run scripts/final_eval.py first. The results "
            "card reads its numbers from that table so they cannot drift."
        )
    rows = parse_eval_table(table_path)

    results_lines: list[tuple] = [("FetchPush-v4", 17, ACCENT),
                                  ("50 episodes x 3 seeds", 10, MUTED)]
    for arm, val in rows:
        # Our arm is the claim being made, so it is the one tinted.
        color = ACCENT if arm == "push_td3_her" else MUTED
        results_lines.append((CARD_LABELS.get(arm, arm), val, 11, color))

    # (kind, payload, duration_or_slowdown)
    storyboard = [
        ("card", [("From-Scratch TD3 + HER", 20, FG),
                  ("on Fetch Manipulation", 20, FG),
                  ("LaunchPad 2026 - Griffin Labs", 11, ACCENT),
                  ('"RL From Scratch" track', 10, MUTED)], 3.5),
        ("card", [("FetchPush-v4", 20, ACCENT),
                  ("TD3+HER, written from scratch", 12, FG),
                  ("held-out eval seeds 10000+", 10, MUTED)], 2.5),
        ("clip", res / "push_demo.mp4", 1.0),
        # The reported 50-episode eval contains no seed-0 failures, so this
        # clip necessarily comes from outside it. Saying so on the card keeps
        # a judge from reading the video as contradicting the results table.
        ("card", [("Failure mode", 20, "#ff8f6b"),
                  ("0 failures in the reported 50 eps -", 10, MUTED),
                  ("swept 200 held-out states to find one", 10, MUTED),
                  ("seed 2081: reaches the block,", 12, FG),
                  ("then pushes only 17% of the way", 12, FG),
                  ("(half speed)", 9, MUTED)], 4.5),
        ("clip", res / "push_failure_seed2081.mp4", 2.0),
        ("card", [("FetchPickAndPlace-v4", 18, ACCENT),
                  ("stretch task, same agent code", 12, FG),
                  ("successes and failures", 10, MUTED)], 2.5),
        ("clip", res / "pickplace_demo.mp4", 1.0),
        ("card", results_lines, 6.0),
    ]

    missing = [str(p) for k, p, _ in storyboard if k == "clip" and not Path(p).exists()]
    if missing:
        raise SystemExit("missing clips: " + ", ".join(missing))

    tmp = Path(tempfile.mkdtemp(prefix="demo_"))
    try:
        segs = []
        for i, (kind, payload, param) in enumerate(storyboard):
            seg = tmp / f"seg{i:02d}.mp4"
            if kind == "card":
                png = tmp / f"card{i:02d}.png"
                render_card(png, payload)
                seg_from_card(png, seg, param)
            else:
                seg_from_clip(Path(payload), seg, param)
            segs.append(seg)

        listing = tmp / "concat.txt"
        listing.write_text("".join(f"file '{s}'\n" for s in segs))
        # All segments share codec/resolution/fps, so stream-copy concat is
        # lossless and avoids a second generation of h264 artefacts.
        ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing),
                "-c", "copy", str(out_path)])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    dur = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(out_path)],
        capture_output=True, text=True, check=True).stdout.strip()
    print(f"wrote {out_path}: {float(dur):.1f}s (limit 120s)")
    if float(dur) > 120:
        raise SystemExit("OVER the 2-minute limit")


if __name__ == "__main__":
    main()
