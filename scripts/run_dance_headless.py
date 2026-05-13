#!/usr/bin/env python3
"""Headless G1 dance_102 run for rl_sar on a DGX Spark (no real display).

Pipeline:
  Xvfb :99  →  rl_sim_mujoco g1 scene_29dof (via PTY)  →  ffmpeg x11grab → MP4

The binary's FSM keyboard input is delivered through stdin (termios non-canonical
mode at src/rl_sar/library/core/rl_sdk/rl_sdk.cpp:383). We send '0' then '3' to
trigger GetUp then RLWholeBodyTrackingDance102.

Run with the venv interpreter:
  /tmp/rl_sar_uv_env/bin/python scripts/run_dance_headless.py
"""

from __future__ import annotations

import os
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

# Tunables
XVFB_WIDTH, XVFB_HEIGHT = 1280, 720
RECORD_SECONDS = 90
STARTUP_WAIT = 5     # FSM + viewer init before sending Num0
GETUP_WAIT = 5       # GetUp interpolation (~2 s) + settle margin before Num3
DANCE_WATCH = 75     # how long to keep the dance running after Num3


def wait_for_xvfb(display: str, timeout: float = 5.0) -> bool:
    """Return True when xdpyinfo can reach the X server."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if subprocess.run(
            ["xdpyinfo", "-display", display],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0:
            return True
        time.sleep(0.2)
    return False


def main() -> int:
    if not BIN.exists():
        print(f"ERROR: binary missing: {BIN}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["DISPLAY"] = DISPLAY
    env["LD_LIBRARY_PATH"] = f"{LIBTORCH}:{LIBMUJOCO}:" + env.get("LD_LIBRARY_PATH", "")

    procs: list = []
    log_fh = open(LOG, "w")
    print(f"[orchestrator] log -> {LOG}")
    print(f"[orchestrator] video -> {MP4}")

    try:
        # 1. Xvfb
        print(f"[orchestrator] starting Xvfb on {DISPLAY} ({XVFB_WIDTH}x{XVFB_HEIGHT})")
        xvfb = subprocess.Popen(
            ["Xvfb", DISPLAY, "-screen", "0", f"{XVFB_WIDTH}x{XVFB_HEIGHT}x24"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        procs.append(("Xvfb", xvfb))
        if not wait_for_xvfb(DISPLAY):
            print("ERROR: Xvfb did not come up in 5 s", file=sys.stderr)
            return 3

        # 2. ffmpeg recording
        print(f"[orchestrator] starting ffmpeg x11grab -> {MP4} ({RECORD_SECONDS}s)")
        ffmpeg = subprocess.Popen(
            [
                "ffmpeg", "-y", "-framerate", "30", "-video_size",
                f"{XVFB_WIDTH}x{XVFB_HEIGHT}",
                "-f", "x11grab", "-i", DISPLAY,
                "-t", str(RECORD_SECONDS),
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                str(MP4),
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        procs.append(("ffmpeg", ffmpeg))

        # 3. rl_sim_mujoco under PTY (stdin = PTY so termios works)
        print(f"[orchestrator] spawning {BIN.name} g1 scene_29dof")
        proc = pexpect.spawn(
            str(BIN), ["g1", "scene_29dof"],
            env=env, timeout=None,
            encoding="utf-8", codec_errors="replace",
            logfile=log_fh,
        )

        # 4. Key sequence: wait → '0' → wait → '3' → watch
        print(f"[orchestrator] waiting {STARTUP_WAIT}s for FSM/viewer init")
        time.sleep(STARTUP_WAIT)
        print("[orchestrator] sending '0' (Num0 → GetUp)")
        proc.send("0")
        time.sleep(GETUP_WAIT)
        print("[orchestrator] sending '3' (Num3 → dance_102)")
        proc.send("3")
        print(f"[orchestrator] watching dance for {DANCE_WATCH}s")
        time.sleep(DANCE_WATCH)

        # 5. Clean shutdown of the binary
        print("[orchestrator] sending SIGINT to rl_sim_mujoco")
        proc.kill(signal.SIGINT)
        try:
            proc.expect(pexpect.EOF, timeout=10)
        except pexpect.TIMEOUT:
            print("[orchestrator] EOF timeout; SIGTERM-ing")
            proc.terminate(force=True)
        return 0

    finally:
        # Stop ffmpeg first (it may still be writing the trailer)
        for name, p in procs:
            if p.poll() is None:
                print(f"[orchestrator] stopping {name}")
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
        log_fh.close()


if __name__ == "__main__":
    sys.exit(main())
