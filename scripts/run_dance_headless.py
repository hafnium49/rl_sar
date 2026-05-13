#!/usr/bin/env python3
"""Headless G1 dance_102 run for rl_sar on a DGX Spark (no real display).

Pipeline:
  Xvfb :99  →  rl_sim_mujoco g1 scene_29dof (via PTY)  →  ffmpeg x11grab → MP4

The binary's FSM keyboard input is delivered through stdin (termios non-canonical
mode at src/rl_sar/library/core/rl_sdk/rl_sdk.cpp:383). Per fsm_g1.hpp the path
to dance_102 requires THREE keypresses with a wait between each:

  Passive --'0'--> GetUp(interp) --'1'--> RLRoboMimicLocomotion --'3'--> dance_102

This script polls run.log to detect each FSM transition before sending the next
key, which makes it robust against the very low real-time-factor under software
rendering (Xvfb llvmpipe on aarch64 runs the sim at ~1% real-time).

Run with the venv interpreter:
  /tmp/rl_sar_uv_env/bin/python scripts/run_dance_headless.py
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pexpect

REPO = Path("/home/h_fujiwara/projects/rl_sar")
BIN = REPO / "cmake_build/bin/rl_sim_mujoco"
LIBTORCH = REPO / "library/inference_runtime/libtorch/lib"
LIBMUJOCO = REPO / "library/mujoco/lib"
MP4 = REPO / "dance_102.mp4"
LOG = REPO / "run.log"
DISPLAY = ":99"

# Screen size — smaller helps the llvmpipe software renderer keep up
W, H = 640, 480

# Per-transition wall-clock timeout (sim runs slow under Xvfb llvmpipe)
TRANSITION_TIMEOUT_S = 900   # 15 min upper bound per state transition
POLL_INTERVAL_S = 1.0
# After reaching dance_102, how long to capture motion frames before shutdown
DANCE_CAPTURE_S = 180
# Hard ceiling on whole run + ffmpeg recording duration
MAX_RECORD_S = 1800  # 30 min


def wait_for_xvfb(display: str, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if subprocess.run(["xdpyinfo", "-display", display],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0:
            return True
        time.sleep(0.2)
    return False


def wait_for_log_pattern(log_path: Path, pattern: str, timeout_s: float,
                         label: str) -> bool:
    """Tail the log file until `pattern` (regex) appears or timeout."""
    print(f"[orchestrator] waiting for: {label}  (regex={pattern!r}, ≤{timeout_s:.0f}s)")
    deadline = time.monotonic() + timeout_s
    rx = re.compile(pattern)
    pos = 0
    while time.monotonic() < deadline:
        try:
            with open(log_path, "r", errors="replace") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
        except FileNotFoundError:
            chunk = ""
        if chunk and rx.search(chunk):
            elapsed = timeout_s - (deadline - time.monotonic())
            print(f"[orchestrator] matched: {label}  (after {elapsed:.1f}s)")
            return True
        time.sleep(POLL_INTERVAL_S)
    print(f"[orchestrator] TIMEOUT waiting for: {label}", file=sys.stderr)
    return False


def main() -> int:
    if not BIN.exists():
        print(f"ERROR: binary missing: {BIN}", file=sys.stderr)
        return 2

    # Reset log so wait_for_log_pattern doesn't see stale matches
    if LOG.exists():
        LOG.unlink()

    env = os.environ.copy()
    env["DISPLAY"] = DISPLAY
    env["LD_LIBRARY_PATH"] = f"{LIBTORCH}:{LIBMUJOCO}:" + env.get("LD_LIBRARY_PATH", "")

    procs: list[tuple[str, subprocess.Popen]] = []
    print(f"[orchestrator] log -> {LOG}")
    print(f"[orchestrator] video -> {MP4}")
    print(f"[orchestrator] DISPLAY={DISPLAY} screen={W}x{H}")

    try:
        # 1. Xvfb
        print("[orchestrator] starting Xvfb")
        xvfb = subprocess.Popen(
            ["Xvfb", DISPLAY, "-screen", "0", f"{W}x{H}x24"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        procs.append(("Xvfb", xvfb))
        if not wait_for_xvfb(DISPLAY):
            print("ERROR: Xvfb did not come up", file=sys.stderr)
            return 3

        # 2. ffmpeg recording (long ceiling; we stop it when done)
        print(f"[orchestrator] starting ffmpeg x11grab -> {MP4}")
        ffmpeg = subprocess.Popen(
            [
                "ffmpeg", "-y", "-framerate", "15",
                "-video_size", f"{W}x{H}",
                "-f", "x11grab", "-i", DISPLAY,
                "-t", str(MAX_RECORD_S),
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                str(MP4),
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        procs.append(("ffmpeg", ffmpeg))

        # 3. Spawn rl_sim_mujoco under PTY
        log_fh = open(LOG, "w")
        print(f"[orchestrator] spawning {BIN.name} g1 scene_29dof")
        proc = pexpect.spawn(
            str(BIN), ["g1", "scene_29dof"],
            env=env, timeout=None,
            encoding="utf-8", codec_errors="replace",
            logfile=log_fh,
        )

        # 4. Sequenced key delivery, each gated on a log transition
        time.sleep(8)  # FSM init + viewer warm-up
        print("[orchestrator] sending '0' (Num0 → GetUp)")
        proc.send("0")
        if not wait_for_log_pattern(
            LOG,
            r"Switch from RLFSMStatePassive to RLFSMStateGetUp",
            TRANSITION_TIMEOUT_S,
            "Passive → GetUp",
        ):
            return 10

        print("[orchestrator] sending '1' (Num1 → RoboMimicLocomotion when GetUp completes)")
        proc.send("1")
        if not wait_for_log_pattern(
            LOG,
            r"Switch from RLFSMStateGetUp to RLFSMStateRLRoboMimicLocomotion",
            TRANSITION_TIMEOUT_S,
            "GetUp → RoboMimicLocomotion",
        ):
            return 11

        print("[orchestrator] sending '3' (Num3 → WholeBodyTrackingDance102)")
        proc.send("3")
        if not wait_for_log_pattern(
            LOG,
            r"Switch from RLFSMStateRLRoboMimicLocomotion to RLFSMStateRLWholeBodyTrackingDance102",
            TRANSITION_TIMEOUT_S,
            "Locomotion → WholeBodyTrackingDance102",
        ):
            return 12

        # 5. Capture some dance frames
        print(f"[orchestrator] dance_102 active — capturing {DANCE_CAPTURE_S}s")
        time.sleep(DANCE_CAPTURE_S)

        # 6. Clean shutdown
        print("[orchestrator] sending SIGINT to rl_sim_mujoco")
        proc.kill(signal.SIGINT)
        try:
            proc.expect(pexpect.EOF, timeout=15)
        except pexpect.TIMEOUT:
            print("[orchestrator] EOF timeout; SIGTERM")
            proc.terminate(force=True)
        return 0

    finally:
        for name, p in procs:
            if p.poll() is None:
                print(f"[orchestrator] stopping {name}")
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
        try:
            log_fh.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
