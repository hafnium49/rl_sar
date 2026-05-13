# rl_sar G1 Dance Run Report

**Run date:** 2026-05-13
**Target policy:** `policy/g1/whole_body_tracking/dance_102`
**Outcome (initial):** ❌ Build failed at `setup_inference_runtime` step. Simulation not launched.
**Root cause:** LibTorch prebuilt is not available for **aarch64 Linux non-Jetson** (DGX Spark) in [scripts/download_inference_runtime.sh](scripts/download_inference_runtime.sh).
**Outcome (build resolved):** ✅ `cmake_build/bin/rl_sim_mujoco` builds (15 MB), `torch.jit.load` succeeds on `dance_102/policy.pt`. See §3b.
**Outcome (final, dance running headless):** ✅✅ Full FSM path executed under Xvfb: Passive → GetUp → RoboMimicLocomotion → **WholeBodyTrackingDance102**. The dance played from 0.00% to **99.94%** of its 34.98s reference motion in a 180s wall-clock capture (~20% RTF). Captured to [dance_102.mp4](dance_102.mp4). See §7b.

---

## 1. Environment

| | |
|---|---|
| Host | `gx10-ed0e` |
| Kernel | `Linux 6.17.0-1008-nvidia #8-Ubuntu SMP PREEMPT_DYNAMIC Wed Jan 21 17:56:56 UTC 2026 aarch64` |
| Arch | **aarch64** |
| GPU | **NVIDIA GB10**, driver `580.126.09` (DGX Spark / Grace-Blackwell class) |
| Python | `/usr/bin/python3` → Python 3.12.3 (no `torch` module installed) |
| ROS | `ROS_DISTRO` unset (CMake-only / `-mj` path was selected, so ROS is not required) |
| Display | `DISPLAY=` (empty), `WAYLAND_DISPLAY=` (empty) — **no GUI available on this host** |
| glxinfo | not installed |
| Sudo | password required (no passwordless sudo) |
| Conda/mamba | not installed |

### apt deps state

Required (from [README.md](README.md) §Dependency):
```
cmake g++ build-essential libyaml-cpp-dev libeigen3-dev libboost-all-dev
libspdlog-dev libfmt-dev libtbb-dev liblcm-dev
```

Installed on this host:
```
build-essential, cmake, g++
```

Missing (all 7):
```
libyaml-cpp-dev, libeigen3-dev, libboost-all-dev, libspdlog-dev,
libfmt-dev, libtbb-dev, liblcm-dev
```

`pkg-config --exists` confirms none of `yaml-cpp / eigen3 / spdlog / fmt / tbb` are findable. No Boost or LCM headers under `/usr/include` or `/usr/local/include`. These would have to be installed with `sudo apt-get install ...` before any build can succeed, regardless of the LibTorch blocker.

---

## 2. Repository state

| | |
|---|---|
| Working tree | `/home/h_fujiwara/projects/rl_sar` |
| HEAD | `cc7f9ffe4fd61097d129fc4160ad6cd24a338938` |
| `git status -s` | clean (only `CLAUDE.md`, `docs/g1_dance_mujoco_plan.md`, `build_mujoco.log`, this report file are local additions; no source modifications) |
| Submodules | none initialized; the standard fetch path is via `scripts/download_robot_descriptions.sh` which clones `fan-ziqi/rl_sar_zoo` into `src/rl_sar_zoo/` (not yet executed — build aborted before this step) |

---

## 3. Build result

### 3a. Initial run — failed

**Command:** `./build.sh -mj 2>&1 | tee build_mujoco.log`

**Outcome:** Failed at step 1 (`setup_inference_runtime`). The full captured log:

```
[Setting up Inference Runtime]
[INFO] Checking inference libraries...
[INFO] Checking inference runtimes...
[ERROR] ARM64 Linux detected but not identified as Jetson
[ERROR] LibTorch prebuilt binaries for generic ARM64 Linux are not available
[INFO] If this is a Jetson device, please check /etc/nv_tegra_release
[ERROR] Failed to setup inference libraries
```

(Note: my shell capture line was `... | tee build_mujoco.log; echo "BUILD_EXIT=$?"` — the `$?` reflected `tee`, not `build.sh`. The script itself `exit 1`s on this failure; see [scripts/download_inference_runtime.sh:105](scripts/download_inference_runtime.sh#L105).)

### Why it fails

[scripts/download_inference_runtime.sh:100-106](scripts/download_inference_runtime.sh#L100-L106):
```bash
Linux)
    if [ "${ARCH_TYPE}" = "aarch64" ]; then
        print_error "ARM64 Linux detected but not identified as Jetson"
        print_error "LibTorch prebuilt binaries for generic ARM64 Linux are not available"
        print_info  "If this is a Jetson device, please check /etc/nv_tegra_release"
        exit 1
    fi
    ...
```

The Jetson code path ([install_pytorch_jetson.sh:31-42](scripts/install_pytorch_jetson.sh#L31-L42)) is gated by two triggers:
- `/etc/nv_tegra_release` — **absent** on this host
- `/usr/local/cuda-*/targets/aarch64-linux` — **absent** on this host

DGX Spark (Grace-Blackwell aarch64, not Tegra) trips neither, so the script bails out before any download occurs. Nothing was downloaded; `library/`, `cmake_build/`, and `src/rl_sar_zoo/` remain absent.

### Why ONNX-only is not a viable substitute

The targeted policy at [policy/g1/whole_body_tracking/dance_102/](policy/g1/whole_body_tracking/dance_102) ships only:
```
policy.pt                          # LibTorch JIT, ~978 KB
G1_Take_102.bvh_60hz.csv           # reference motion, ~582 KB
config.yaml                        # model_name: policy.pt
```

There is no `.onnx` companion, so even though `download_inference_runtime.sh` has a working aarch64 ONNX Runtime URL ([line 201](scripts/download_inference_runtime.sh#L201)), it cannot be used to load this policy. The same is true for `gangnam_style` (only `policy.pt` + motion CSV + config).

### 3b. Resolved build

Final working recipe — bypass the script's aarch64 bail-out by pre-populating `library/` with assets sourced from a local `uv` venv (for MuJoCo) and an existing conda env (for LibTorch, since the pip-installed wheel had an ABI mismatch with Ubuntu 24.04):

| Step | What | Notes |
|---|---|---|
| 0 | `git submodule update --init --recursive` | The build needs `unitree_sdk2`, `Lite3_MotionSDK`, `agibot_D1_Edu-Ultra`, `joystick`, etc. — 6 submodules, all anonymous HTTPS |
| 1 | `uv venv /tmp/rl_sar_uv_env --python 3.10 && uv pip install "mujoco==3.2.7" "torch==2.3.0" numpy` | Per upstream MuJoCo install guidance |
| 2 | Symlink wheel's `include/` + `libmujoco.so.3.2.7` into `library/mujoco/`, write `VERSION_NUMBER=3.2.7` | Exact-version match; CMake's hardcoded `libmujoco.so.3.2.7` ([CMakeLists.txt:660](src/rl_sar/CMakeLists.txt#L660)) resolves directly |
| 3a | Initial: copy torch 2.3.0 (pip) + its `torch.libs/` sibling dir into `library/inference_runtime/libtorch/` | **Failed** — pip torch is `_GLIBCXX_USE_CXX11_ABI=0` (OLD ABI) while Ubuntu 24.04 system `yaml-cpp` is NEW ABI → undefined references on `YAML::LoadFile(std::string)` etc. |
| 3b | Switched to NVIDIA-built **torch 2.9.1+cu130** from conda env `sam3` (`CXX11_ABI=True`) | Eliminated the ABI mismatch. JIT-load of the 2.3-era policy still works (verified) |
| 3c | Copied 4 extra CUDA deps from `sam3/lib/.../nvidia/{cudnn,nccl,nvshmem,cusparselt}/lib/` into `library/inference_runtime/libtorch/lib/` | Needed by `libtorch_cuda.so` because find_package(Torch) links the full set even though we only do CPU inference |
| 4 | `sudo apt install` 9 dev packages: yaml-cpp, eigen3, boost-all, spdlog, fmt, tbb, lcm, glfw3-dev, python3-dev, python3-numpy (skipped transitional `libgl1-mesa-glx` and `libegl1-mesa`) | Ubuntu 24.04 noble naming |
| 5 | `CMAKE_PREFIX_PATH=.../libtorch/share/cmake CUDAToolkit_ROOT=/usr/local/cuda bash -o pipefail -c './build.sh -mj 2>&1 \| tee build_mujoco.log'` | CUDAToolkit_ROOT needed because `TorchConfig.cmake` pulls in `Caffe2Config.cmake` which `find_package(CUDAToolkit REQUIRED)` |

**Result:** all 7 binaries built (rl_real_a1/d1/g1/go2/l4w4/lite3 + **rl_sim_mujoco**). Build log: [build_mujoco.log](build_mujoco.log).

**Binary verification:**
- Size: 15 MB at `cmake_build/bin/rl_sim_mujoco`
- `ldd` resolves cleanly to our trees:
  ```
  libmujoco.so.3.2.7 -> library/mujoco/lib/libmujoco.so.3.2.7
  libtorch_cpu.so    -> library/inference_runtime/libtorch/lib/libtorch_cpu.so
  libc10.so          -> library/inference_runtime/libtorch/lib/libc10.so
  libcudnn.so.9      -> library/inference_runtime/libtorch/lib/libcudnn.so.9
  libarm_compute.so  -> library/inference_runtime/libtorch/lib/libarm_compute.so
  libnvpl_blas_lp64_gomp.so.0 -> library/inference_runtime/libtorch/lib/...
  ```
- Smoke-launch: `./cmake_build/bin/rl_sim_mujoco` (no args) registers `FSMManager` for all 11 robot types (a1, b2, b2w, d1, g1, go2, go2w, gr1t1, gr1t2, l4w4, lite3) — confirming static init and dynamic linking succeed before the binary exits with a usage error. Full simulation requires `<ROBOT> <SCENE>` args and a display.

**JIT-load smoke test (the canonical risk we wanted to surface early):**
```bash
$ /home/h_fujiwara/miniconda3/envs/sam3/bin/python -c "
  import torch
  m = torch.jit.load('policy/g1/whole_body_tracking/dance_102/policy.pt', map_location='cpu')
  print(type(m).__name__, next(m.parameters()).shape)
"
RecursiveScriptModule torch.Size([512, 154])
```
The `154` matches `num_observations: 154` in [dance_102/config.yaml](policy/g1/whole_body_tracking/dance_102/config.yaml). The 2.3-era policy loads cleanly on the 2.9.1 runtime — no schema or opcode drift in this particular network.

### 3c. What changed vs. the original plan

The plan called for pip torch 2.3.0. In execution we found:
1. The torch.libs sibling dir wasn't being copied by the reused `create_libtorch_from_pytorch` body — fixed by an extra copy step.
2. Pip torch 2.3.0 aarch64 wheel uses OLD ABI, incompatible with Ubuntu 24.04 system yaml-cpp/libstdc++ in NEW ABI. PyPI/PyTorch CPU index has no `cxx11.abi` variant for aarch64. So we replaced it with NVIDIA's NEW-ABI build from the local `sam3` conda env, which happened to ship CUDA support — needing the CUDAToolkit_ROOT environment variable.
3. Git submodules had to be init'd separately (the README mentions `--recursive` clone; the cloned tree here was non-recursive).

The plan's "use pre-installed mujoco" directive came through fully — `library/mujoco/lib/libmujoco.so.3.2.7` is the exact upstream-published wheel binary, no version fakery.

---

## 4. Policy discovery

| Item | Value |
|---|---|
| Robot dir | [policy/g1/](policy/g1) |
| Base config | [policy/g1/base.yaml](policy/g1/base.yaml) — 29 DoF, identity `joint_mapping` |
| Selected config dir | [policy/g1/whole_body_tracking/dance_102/](policy/g1/whole_body_tracking/dance_102) |
| Model file | `policy.pt` (LibTorch JIT, 978 KB) |
| Motion file | `G1_Take_102.bvh_60hz.csv` (582 KB, 60 Hz reference motion) |
| Model format | `.pt` only — **no `.onnx` available** |
| Observation dim | 154 |
| Observations | `whole_body_tracking/motion_command`, `whole_body_tracking/motion_anchor_ori_b`, `ang_vel`, `dof_pos`, `dof_vel`, `actions` |
| Action dim | 29 |
| Decimation × dt | 4 × 0.005 s = 50 Hz policy step (from base.yaml) |
| `joint_mapping` (WBT) | non-trivial permutation `[0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28]` that overrides base.yaml |

All required policy + motion assets are present and self-contained — no further policy download is needed.

---

## 5. Run command (not executed)

The intended command (per [agent.md §Step 6](agent.md) and [docker/README.md](docker/README.md)):

```bash
./cmake_build/bin/rl_sim_mujoco g1 scene_29dof
```

Resolved by [src/rl_sar/src/rl_sim_mujoco.cpp:64](src/rl_sar/src/rl_sim_mujoco.cpp#L64) to:
`<repo>/src/rl_sar_zoo/g1_description/mjcf/scene_29dof.xml`

This binary was never built (see §3) and `src/rl_sar_zoo/` was never fetched.

---

## 6. Skill activation (not exercised)

Confirmed in [src/rl_sar/fsm_robot/fsm_g1.hpp](src/rl_sar/fsm_robot/fsm_g1.hpp) (lines 280–370):

| FSM state | Key (Kbd) | Key (Gamepad) | `config_name` set |
|---|---|---|---|
| `RLFSMStatePassive` | start | start | – |
| `RLFSMStateGetUp` | Num0 | A | – |
| `RLFSMStateRLRoboMimicLocomotion` | Num1 | RB_DPadUp | `robomimic/locomotion` |
| `RLFSMStateRLRoboMimicCharleston` | Num2 | RB_DPadDown | `robomimic/charleston` |
| **`RLFSMStateRLWholeBodyTrackingDance102`** | **Num3** | **RB_DPadLeft** | **`whole_body_tracking/dance_102`** |
| `RLFSMStateRLWholeBodyTrackingGangnamStyle` | Num4 | RB_DPadRight | `whole_body_tracking/gangnam_style` |
| `RLFSMStateGetDown` | Num9 | B | – |

Intended sequence had the build succeeded: `Passive` → press Num0 → wait ~3 s for stand → press Num3 → observe dance_102.

---

## 7. Observed behavior

### 7a. Initial run (before unblocking the build)

None — the binary was never produced. No MuJoCo viewer, no FSM transitions, no policy load, no motion playback observed.

Secondary observation: this host has **no display server** (`$DISPLAY` empty, no Wayland, no glxinfo). The MuJoCo viewer (GLFW + GLX) cannot open without one of: an attached display, X11 forwarding (`ssh -X`), `MUJOCO_GL=egl` / `MUJOCO_GL=osmesa` headless rendering, or **Xvfb**. Resolved in §7b.

### 7b. Resolved run via Xvfb + scripted PTY input

After the build was unblocked (§3b), a headless orchestrator at [scripts/run_dance_headless.py](scripts/run_dance_headless.py) successfully drove a full dance sequence:

1. Started `Xvfb :99` at 640×480 software-rendered (`mesa-utils` shows `OpenGL renderer: llvmpipe`)
2. Started `ffmpeg -f x11grab` recording to MP4
3. Spawned `cmake_build/bin/rl_sim_mujoco g1 scene_29dof` under a PTY (via `pexpect`) so stdin is a real terminal — the binary's keyboard reader uses `tcsetattr` non-canonical mode ([src/rl_sar/library/core/rl_sdk/rl_sdk.cpp:350-374](src/rl_sar/library/core/rl_sdk/rl_sdk.cpp#L350-L374))
4. Sent the **three** keys required by the G1 FSM hierarchy (not just Num0→Num3 as initially assumed):

| When | Key | Transition (from run.log) | Wall-clock |
|---|---|---|---|
| t=11.6s | `0` | `Switch from RLFSMStatePassive to RLFSMStateGetUp` | 0.1 s after send |
| auto | — | `Getting up completed` (2-sec interpolation finished) | **2.2 s** wall (≈91% RTF) |
| t≈14s | `1` | `Switch from RLFSMStateGetUp to RLFSMStateRLRoboMimicLocomotion` | 0.1 s after send |
| t≈14s | `3` | `Switch from RLFSMStateRLRoboMimicLocomotion to RLFSMStateRLWholeBodyTrackingDance102` | 0.1 s after send |

After entering the dance state, the binary logged:

```
Successfully loaded Torch model: policy/g1/whole_body_tracking/dance_102/policy.pt
MotionLoader: Loaded 1749 frames, 29 joints, duration=34.98s
Motion reset with yaw alignment
Motion duration: 34.98s
[INFO] [...] 0.00% - whole_body_tracking/dance_102
[INFO] [...] 0.06% - whole_body_tracking/dance_102
…
[INFO] [...] 99.94% - whole_body_tracking/dance_102
```

The dance progressed from **0.00% → 99.94%** of the 34.98 s reference motion over a 180 s wall-clock capture window. That's a real-time-factor of about **20%** for the full RL-inference + MuJoCo step + software-rendered viewer pipeline. GetUp's pure-kinematics state ran at ~91% RTF; the dance state is slower because it adds `torch::jit::forward()` on a 154-dim observation each control tick at 50 Hz, all on CPU through Mesa llvmpipe.

**No protect warnings** (`TorqueProtect`, `AttitudeProtect`), **no JIT load failures**, **no motion-file errors**. The only error in the log was `Joystick [/dev/input/js0] open failed`, which is expected and harmless (no joystick attached).

### 7c. Two non-obvious behaviors discovered in execution

1. **G1's FSM does not allow direct GetUp → dance_102.** [fsm_g1.hpp:78-87](src/rl_sar/fsm_robot/fsm_g1.hpp#L78-L87) only accepts `Num1` (locomotion) or `Num9` (getdown) out of GetUp. You must transition GetUp → RoboMimicLocomotion → dance_102. The Num0→Num3 shorthand mentioned in `agent.md` §Step 8 is wrong for this build.
2. **`RobotControl()` clears keyboard state every cycle.** [rl_sim_mujoco.cpp:222](src/rl_sar/src/rl_sim_mujoco.cpp#L222) calls `this->control.ClearInput()` at the end of every control loop iteration, which resets `current_keyboard` to `last_keyboard`. A single keypress only stays "live" for one control cycle. The headless orchestrator therefore implements `send_until_transition()` — re-sending the key every 5 s until the corresponding FSM transition appears in the log. With this retry loop, all three transitions fire within 0.1 s of the matching send.

---

## 8. Artifacts

| Path | What |
|---|---|
| [build_mujoco.log](build_mujoco.log) | Full `./build.sh -mj` build output (final run; ~2000 lines) |
| [run.log](run.log) | Headless run output captured by `pexpect.logfile_read` (~3.3 MB; ANSI-laden) |
| [dance_102.mp4](dance_102.mp4) | Recorded simulation: 186 s, 640×480, 2788 frames @ 15 fps, 341 KB. First ~14 s is initial-pose + GetUp + locomotion transition; remainder is dance_102 motion playback at ~20% RTF (so ~35 s of sim motion shown). |
| [scripts/run_dance_headless.py](scripts/run_dance_headless.py) | The Xvfb + PTY + ffmpeg orchestrator |
| [docs/g1_dance_mujoco_plan.md](docs/g1_dance_mujoco_plan.md) | Original execution plan (build-unblock phase) |
| [CLAUDE.md](CLAUDE.md) | Project guidance |
| [rl_sar_g1_dance_run_report.md](rl_sar_g1_dance_run_report.md) | This report |

---

## 9. Next actions

Concrete steps, in dependency order. Each is a separate decision the user should approve before execution — none are autonomous.

### 9a. Provision aarch64 LibTorch (the actual blocker)

The script's bail-out is conservative — DGX Spark **can** run LibTorch, but the rl_sar repo doesn't ship a code path for it. Two practical options:

1. **Pip-install PyTorch for aarch64, then run the existing `create_libtorch_from_pytorch` function manually.**
   ```bash
   # Install NVIDIA's PyTorch wheel for Grace (aarch64+CUDA), or CPU-only torch from PyPI:
   pip3 install --user torch   # CPU wheel; for CUDA use NVIDIA's index
   # Then bypass the Jetson gate by sourcing only the create_libtorch_from_pytorch function:
   bash -c '
     LIBTORCH_DIR="/home/h_fujiwara/projects/rl_sar/library/inference_runtime/libtorch"
     mkdir -p "$(dirname "$LIBTORCH_DIR")"
     torch_path=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
     mkdir -p "$LIBTORCH_DIR"
     for d in bin include lib share; do
       [ -d "$torch_path/$d" ] && cp -r "$torch_path/$d" "$LIBTORCH_DIR/"
     done
   '
   ```
   This is essentially what [install_pytorch_jetson.sh:192-286](scripts/install_pytorch_jetson.sh#L192-L286) does after the platform check — we'd run that body without the gate.

2. **Patch the download script** to also accept `aarch64` non-Jetson by extending the Jetson detection to include "any aarch64 Linux with CUDA installed" or by adding a new "Grace Blackwell" code path. This is a code change the upstream maintainer might want as a PR.

Recommended: option 1 first, scoped to this host. Don't patch sources unless we're going to PR it.

### 9b. Install missing apt packages

```bash
sudo apt-get update
sudo apt-get install -y libyaml-cpp-dev libeigen3-dev libboost-all-dev \
                        libspdlog-dev libfmt-dev libtbb-dev liblcm-dev
```

Required regardless of how LibTorch is provisioned. Needs the user's sudo password.

### 9c. Solve the display problem before launching the viewer

Even after a successful build, `rl_sim_mujoco` needs interactive keyboard input through the MuJoCo GLFW window. On this headless host:

| Approach | Pros | Cons |
|---|---|---|
| SSH `-X` / `-Y` from a Linux client | Native key handling | Slow rendering over network; requires a desktop client |
| Run on a workstation with attached display | Simplest; MuJoCo at full FPS | Need a different machine |
| Xvfb + xdotool key injection | Fully headless and scriptable | The simulation is meant to be observed; you'd need video capture (`ffmpeg -f x11grab`) to verify the dance |
| Docker on a workstation (see [docker/README.md](docker/README.md)) | Encapsulates apt deps | Still needs a display on the host |

Recommendation: run this elsewhere (a workstation), or set up Xvfb + screencap on this DGX Spark for an unattended run.

### 9d. Re-run the plan

Once 9a–9c are resolved, the steps in [docs/g1_dance_mujoco_plan.md](docs/g1_dance_mujoco_plan.md) §3–§7 should execute as designed: `./build.sh -mj` will skip the now-present LibTorch (idempotent), fetch MuJoCo and `rl_sar_zoo`, compile `rl_sim_mujoco`, and the launch sequence Num0→wait→Num3 should bring up dance_102.

### 9e. Stretch — gangnam_style after dance_102

Per the original user direction, gangnam_style is deferred. After dance_102 is verified ≥60 s upright, switch via Num4 (or restart with `RLFSMStateRLWholeBodyTrackingGangnamStyle` as initial) without rebuilding.

---

## Out of scope (per agent.md, not attempted)

- Training any new policy
- Real-hardware deployment (`rl_real_g1` not built)
- Custom kung-fu / motion-retargeting pipeline
- Source patches to `src/` to bypass the LibTorch arch gate (would be a PR to upstream, not a local fix)
