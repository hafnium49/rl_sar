#!/usr/bin/env python3
"""Headless G1 dance_102 run for rl_sar on a DGX Spark (no real display).

Pipeline:
  Xvfb :99  →  rl_sim_mujoco g1 scene_29dof (via PTY)  →  ffmpeg x11grab → MP4

Keyboard input goes through stdin via termios non-canonical mode (rl_sdk.cpp:383).
The FSM path to dance_102 needs three keys with a wait between each:

  Passive --'0'--> GetUp(interp) --'1'--> RLRoboMimicLocomotion --'3'--> dance_102

We use pexpect.expect() with regex patterns to gate each next-key on the actual
log-line of the FSM transition. This is robust against the very low real-time-
factor under Xvfb llvmpipe on aarch64 (~1% RTF for this 29-DoF G1 model).
pexpect.expect() drives the PTY read loop, so output is drained continuously
and the binary doesn't block on a full PTY buffer.

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

W, H = 640, 480

# Per-transition timeout (sim under llvmpipe runs at ~1% RTF for the 29-DoF scene)
TRANSITION_TIMEOUT_S = 900   # 15 min upper bound each
# After dance_102 starts, how long to keep capturing
DANCE_CAPTURE_S = 180
# Hard ceiling on the ffmpeg recording
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


def expect_transition(proc: pexpect.spawn, regex: str, label: str,
                      timeout_s: float) -> bool:
    t0 = time.monotonic()
    print(f"[orchestrator] waiting: {label}", flush=True)
    try:
        proc.expect(regex, timeout=timeout_s)
    except pexpect.TIMEOUT:
        print(f"[orchestrator] TIMEOUT waiting: {label}", file=sys.stderr, flush=True)
        return False
    except pexpect.EOF:
        print(f"[orchestrator] EOF waiting: {label}", file=sys.stderr, flush=True)
        return False
    print(f"[orchestrator] matched: {label}  ({time.monotonic()-t0:.1f}s)", flush=True)
    return True


def send_until_transition(proc: pexpect.spawn, key: str, regex: str,
                          label: str, total_timeout_s: float,
                          inner_timeout_s: float = 5.0) -> bool:
    """Send `key` and wait up to inner_timeout_s for `regex`; if not seen, re-send.

    Needed because rl_sim_mujoco.cpp:222 calls ClearInput() at end of every
    RobotControl cycle — a single send may be wiped before the FSM's CheckChange
    notices it. Re-sending until the transition log appears wins the race.
    """
    t0 = time.monotonic()
    print(f"[orchestrator] send-loop '{key}' until: {label}  (≤{total_timeout_s:.0f}s)",
          flush=True)
    while time.monotonic() - t0 < total_timeout_s:
        proc.send(key)
        try:
            proc.expect(regex, timeout=inner_timeout_s)
            print(f"[orchestrator] matched: {label}  ({time.monotonic()-t0:.1f}s)",
                  flush=True)
            return True
        except pexpect.TIMEOUT:
            continue  # re-send
        except pexpect.EOF:
            print(f"[orchestrator] EOF in send-loop {label}", file=sys.stderr)
            return False
    print(f"[orchestrator] TIMEOUT in send-loop: {label}", file=sys.stderr, flush=True)
    return False


def main() -> int:
    if not BIN.exists():
        print(f"ERROR: binary missing: {BIN}", file=sys.stderr)
        return 2

    if LOG.exists():
        LOG.unlink()

    env = os.environ.copy()
    env["DISPLAY"] = DISPLAY
    env["LD_LIBRARY_PATH"] = f"{LIBTORCH}:{LIBMUJOCO}:" + env.get("LD_LIBRARY_PATH", "")

    procs: list[tuple[str, subprocess.Popen]] = []
    print(f"[orchestrator] log -> {LOG}", flush=True)
    print(f"[orchestrator] video -> {MP4}", flush=True)
    print(f"[orchestrator] DISPLAY={DISPLAY} screen={W}x{H}", flush=True)

    log_fh = None
    proc = None
    try:
        # 1. Xvfb
        print("[orchestrator] starting Xvfb", flush=True)
        xvfb = subprocess.Popen(
            ["Xvfb", DISPLAY, "-screen", "0", f"{W}x{H}x24"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        procs.append(("Xvfb", xvfb))
        if not wait_for_xvfb(DISPLAY):
            print("ERROR: Xvfb did not come up", file=sys.stderr)
            return 3

        # 2. ffmpeg recording (long ceiling; we stop it when done)
        print(f"[orchestrator] starting ffmpeg x11grab -> {MP4}", flush=True)
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

        # 3. Spawn rl_sim_mujoco under PTY; log everything pexpect reads
        log_fh = open(LOG, "w")
        print(f"[orchestrator] spawning {BIN.name} g1 scene_29dof", flush=True)
        proc = pexpect.spawn(
            str(BIN), ["g1", "scene_29dof"],
            env=env, timeout=None,
            encoding="utf-8", codec_errors="replace",
        )
        proc.logfile_read = log_fh  # log only output, not our sends

        # 4. Wait for the binary to print the initial FSMManager line, then
        #    drive transitions via expect()
        if not expect_transition(
            proc, r"FSMManager.*Registered type: g1",
            "binary up (FSMManager registered g1)", 60):
            return 4

        # Give the viewer & simulation_running flag a moment to settle
        time.sleep(3)

        # Send '0' once — Passive's CheckChange has no percent-gate, so a single
        # cycle with current_keyboard=Num0 triggers the transition.
        if not send_until_transition(
            proc, "0",
            r"Switch from RLFSMStatePassive to RLFSMStateGetUp",
            "Passive → GetUp", total_timeout_s=60):
            return 10

        # Wait for GetUp interpolation to reach 100%, THEN send Num1.
        # Interpolate prints "100.00% - Getting up" followed by "Getting up completed".
        if not expect_transition(
            proc, r"Getting up completed",
            "GetUp interpolation 100%", TRANSITION_TIMEOUT_S):
            return 13

        # Now send Num1 repeatedly until the transition fires (beats ClearInput race).
        if not send_until_transition(
            proc, "1",
            r"Switch from RLFSMStateGetUp to RLFSMStateRLRoboMimicLocomotion",
            "GetUp → RoboMimicLocomotion", total_timeout_s=120):
            return 11

        # RLRoboMimicLocomotion's Enter loads its model; Num3 must be seen by
        # CheckChange to transition. Send '3' repeatedly.
        if not send_until_transition(
            proc, "3",
            r"Switch from RLFSMStateRLRoboMimicLocomotion to RLFSMStateRLWholeBodyTrackingDance102",
            "Locomotion → WholeBodyTrackingDance102", total_timeout_s=300):
            return 12

        # 5. Capture dance frames. Use expect with TIMEOUT to keep the PTY drained
        #    while we wait DANCE_CAPTURE_S seconds.
        print(f"[orchestrator] dance_102 active — capturing {DANCE_CAPTURE_S}s", flush=True)
        try:
            proc.expect(pexpect.TIMEOUT, timeout=DANCE_CAPTURE_S)
        except pexpect.EOF:
            print("[orchestrator] binary exited during dance capture", file=sys.stderr)

        # 6. Clean shutdown
        print("[orchestrator] sending SIGINT to rl_sim_mujoco", flush=True)
        proc.kill(signal.SIGINT)
        try:
            proc.expect(pexpect.EOF, timeout=15)
        except pexpect.TIMEOUT:
            print("[orchestrator] EOF timeout; SIGTERM", flush=True)
            proc.terminate(force=True)
        return 0

    finally:
        # Stop ffmpeg + Xvfb
        for name, p in procs:
            if p.poll() is None:
                print(f"[orchestrator] stopping {name}", flush=True)
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
        if log_fh is not None:
            log_fh.close()


if __name__ == "__main__":
    sys.exit(main())
