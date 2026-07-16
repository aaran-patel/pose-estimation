"""
inspect_links.py

First-pass sanity analysis of pose_link_logger.py output. Does NOT do fall
detection -- just checks that the logged geometry behaves the way real
bones should, and gives a quick visual look at how torso angle differs
across session types (standing, bending, falling, etc).

Run:
    python inspect_links.py session_bend.jsonl session_fall_sim.jsonl ...
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_session(path):
    """Flatten a JSONL session into a per-(frame, link) dataframe."""
    rows = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            for link_name, geom in rec["links"].items():
                if geom is None:
                    continue
                rows.append({
                    "session": Path(path).stem,
                    "frame_id": rec["frame_id"],
                    "timestamp": rec["timestamp"],
                    "link": link_name,
                    "length_m": geom["length_m"],
                    "angle_from_vertical_deg": geom["angle_from_vertical_deg"],
                    "confidence": geom["confidence"],
                })
    return pd.DataFrame(rows)


def report_length_stability(df):
    """Bone length should be roughly constant. High std = depth noise."""
    print("\n--- Link length stability (should be low std if depth is clean) ---")
    stats = (
        df.groupby(["session", "link"])["length_m"]
        .agg(["mean", "std", "count"])
        .sort_values("std", ascending=False)
    )
    print(stats.round(3).to_string())


def plot_torso_angle(df, out_path="torso_angle_comparison.png"):
    """Overlay torso angle-from-vertical across sessions for left_torso."""
    torso = df[df["link"] == "left_torso"]
    if torso.empty:
        print("No left_torso data found -- check keypoint confidence/depth.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for session, group in torso.groupby("session"):
        ax.plot(group["frame_id"], group["angle_from_vertical_deg"],
                label=session, marker=".", linewidth=1)

    ax.set_xlabel("Frame")
    ax.set_ylabel("left_torso angle from vertical (deg)")
    ax.set_title("Torso angle across sessions (0 = upright, 90 = horizontal)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_links.py session1.jsonl [session2.jsonl ...]")
        sys.exit(1)

    dfs = [load_session(p) for p in sys.argv[1:]]
    df = pd.concat(dfs, ignore_index=True)

    print(f"Loaded {len(df)} link-observations across {df['session'].nunique()} session(s)")
    print(f"Sessions: {sorted(df['session'].unique())}")

    report_length_stability(df)
    plot_torso_angle(df)


if __name__ == "__main__":
    main()
