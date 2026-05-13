# G1 dance in MuJoCo on DGX Spark — what was done

> **Status: complete.** Both `dance_102` and `gangnam_style` motions render cleanly via GPU offscreen rendering, with optional camera orbit.

This document is the as-built reference for everything done in this repo to take it from a fresh clone on an NVIDIA DGX Spark (aarch64, NVIDIA GB10, Ubuntu 24.04) all the way to GPU-rendered orbit videos of the G1 humanoid performing the pretrained whole-body-tracking motions. It supersedes the original "plan" document (which was the speculative pre-execution playbook).

The starting mission brief is at [agent.md](/home/h_fujiwara/projects/rl_sar/agent.md). The starting environment had **none** of `library/`, `cmake_build/`, `src/rl_sar_zoo/`, and the build script's aarch64 non-Jetson path was broken.

The final outcome is two working pipelines:

| Pipeline | Script | Output | What it shows |
|---|---|---|---|
| **C++ live sim** + Xvfb GLFW capture | [scripts/run_dance_headless.py](../scripts/run_dance_headless.py) | `dance_102.mp4` | rl_sim_mujoco binary running the real RL policy under physics — the robot falls during the FSM transitions because of the rapid GetUp → Locomotion → Dance cascade and the slow RTF of software-rendered llvmpipe. Useful for validating the deployment path; not pretty. |
| **Python offscreen** + kinematic playback (recommended) | [scripts/record_dance_offscreen.py](../scripts/record_dance_offscreen.py), [scripts/record_dance_orbit.py](../scripts/record_dance_orbit.py) | `dance_102_offscreen.mp4`, `dance_102_orbit.mp4`, `gangnam_style_orbit.mp4` | 1280×720 @ 60 fps, GPU-rendered via NVIDIA EGL on the GB10. Plays the BVH-derived reference motion (the choreography the policy was trained to track) with no physics, so the robot doesn't fall. ~150 fps render rate. |

---

## Phase 1 — Codebase orientation

Took a fresh clone and produced [`CLAUDE.md`](../CLAUDE.md), a concise project-instructions file covering build system, control flow, FSM-per-robot pattern, policy/config layout, library structure, and the lessons-learned about LibTorch JIT, ABI gotchas, CSV logger flag, and intentional `kp=0, kd=8` Passive damping.

That file is now committed and serves as the working brief for future agents.

---

## Phase 2 — Build unblock

### The actual blocker

`./build.sh -mj` exited 1 at step 1 (`setup_inference_runtime`). The root cause is [scripts/download_inference_runtime.sh:101-106](../scripts/download_inference_runtime.sh#L101-L106):

```bash
Linux)
    if [ "${ARCH_TYPE}" = "aarch64" ]; then
        print_error "ARM64 Linux detected but not identified as Jetson"
        print_error "LibTorch prebuilt binaries for generic ARM64 Linux are not available"
        exit 1
    fi
```

The script has no LibTorch path for aarch64 non-Jetson. DGX Spark trips neither Jetson marker (no `/etc/nv_tegra_release`, no `/usr/local/cuda-*/targets/aarch64-linux`).

### Several wrong-turns we backed out of

Before landing on the working recipe, we tried (and abandoned) several plans:

1. **Symlink a Python `mujoco` wheel's bundled `libmujoco.so.X.Y.Z` as `libmujoco.so.3.2.7`** — rejected by the multi-agent review because MuJoCo 3.3.3 removed the `mjvSceneState` API used throughout rl_sar's vendored [simulate.cc](../src/rl_sar/library/thirdparty/mujoco_simulate/simulate.cc); 3.3+ also requires C++20 while rl_sar pins C++17 ([src/rl_sar/CMakeLists.txt:5](../src/rl_sar/CMakeLists.txt#L5)).
2. **Build MuJoCo 3.2.7 from local source** at `/home/h_fujiwara/projects/mujoco` via a git worktree — rejected by user ("do not build manually").
3. **`uv pip install mujoco==3.2.7 torch==2.3.0`** — closer, but the pip-installed PyTorch aarch64 wheel is built with **`_GLIBCXX_USE_CXX11_ABI=0` (OLD ABI)** while Ubuntu 24.04 system `yaml-cpp`/`libstdc++` use NEW ABI. CMake link failed with `undefined reference to YAML::LoadFile(std::string const&)` and a cascade of similar mismatches. PyTorch's CPU index has **no `cxx11.abi` wheel for aarch64**.

### The working recipe (current state)

| Asset | Source | Where on disk |
|---|---|---|
| **LibTorch (NEW ABI)** | NVIDIA-built `torch==2.9.1+cu130` from existing `~/miniconda3/envs/sam3/lib/python3.10/site-packages/torch`, copied including `torch.libs/` and the four CUDA siblings (`libcudnn.so.9`, `libnccl.so.2`, `libnvshmem_host.so.3`, `libcusparseLt.so.0`) from `nvidia/{cudnn,nccl,nvshmem,cusparselt}/lib/` | `library/inference_runtime/libtorch/` (~970 MB) |
| **MuJoCo 3.2.7** | `uv pip install mujoco==3.2.7` into a throwaway `/tmp/rl_sar_uv_env` (Python 3.10) — the official PyPI wheel includes both `include/mujoco/mujoco.h` and `libmujoco.so.3.2.7` exactly. Symlinked into the expected layout. | `library/mujoco/` (symlinks + `VERSION_NUMBER`) |
| **ONNX Runtime 1.22.0** | Auto-downloaded by `build.sh -mj` (its aarch64 URL works fine) | `library/inference_runtime/onnxruntime/` |
| **`rl_sar_zoo` (G1 MJCF)** | `scripts/download_robot_descriptions.sh` (auto-fetched by build.sh -mj) | `src/rl_sar_zoo/` (~568 MB; gitignored) |
| **Git submodules** (unitree_sdk2 etc.) | `git submodule update --init --recursive --recommend-shallow` | `src/rl_sar/library/thirdparty/robot_sdk/*` |
| **apt dev deps** | One `sudo apt-get install -y` (~142 MB, ~200 packages) — see exact list below | system |

#### apt install line (Ubuntu 24.04 / noble)

```bash
sudo apt-get install -y \
    libyaml-cpp-dev libeigen3-dev libboost-all-dev \
    libspdlog-dev libfmt-dev libtbb-dev liblcm-dev \
    libglfw3 libglfw3-dev \
    libxrandr2 libxinerama1 libxcursor1 libxi6 \
    libgl1-mesa-dri libegl1 libglvnd-dev \
    python3-dev python3-numpy
```

`libgl1-mesa-glx` and `libegl1-mesa` from Docker's reference list are **dropped** — both are transitional packages removed in noble (`libegl1` + `libglvnd-dev` + `libgl1-mesa-dri` already cover the functionality).

#### Why we ended up with sam3's torch and not a fresh pin

The original plan was `uv pip install torch==2.3.0` to match the script's `LIBTORCH_VERSION="2.3.0"`. That works for download, but on aarch64 the wheel is OLD-ABI which mismatches Ubuntu 24.04's yaml-cpp. The pre-installed `sam3` env happens to contain NVIDIA's NEW-ABI torch 2.9.1+cu130 build (tagged for Grace-Blackwell), and crucially **JIT-loads `dance_102/policy.pt` successfully** despite the 2.3 → 2.9 version drift (verified by `python -c "import torch; torch.jit.load(...)"` returning `RecursiveScriptModule` with first-param shape `[512, 154]` matching `num_observations: 154`).

CMake needs `CUDAToolkit_ROOT=/usr/local/cuda` for `find_package(Torch)` because the 2.9 build's `Caffe2Config.cmake` includes `find_package(CUDAToolkit REQUIRED)`.

#### Two non-obvious copy steps

1. PyTorch's auditwheel-repaired aarch64 wheel keeps its bundled deps (OpenBLAS, OpenMP, arm_compute, ...) in a **sibling** dir `torch.libs/`, not in `torch/lib/`. The `create_libtorch_from_pytorch` function in [scripts/install_pytorch_jetson.sh:192-286](../scripts/install_pytorch_jetson.sh#L192-L286) only copies `bin/include/lib/share` — so we additionally copied `torch.libs/*` into `library/inference_runtime/libtorch/lib/` to satisfy the runtime linker.
2. For `find_package(Torch)` to actually find the tree, set `CMAKE_PREFIX_PATH="$REPO/library/inference_runtime/libtorch/share/cmake"` on the build command line.

#### The final build command

```bash
cd /home/h_fujiwara/projects/rl_sar
CMAKE_PREFIX_PATH="$PWD/library/inference_runtime/libtorch/share/cmake" \
CUDAToolkit_ROOT=/usr/local/cuda \
bash -o pipefail -c './build.sh -mj 2>&1 | tee build_mujoco.log'
```

(`pipefail` is necessary because `set -e` inside `build.sh` was being masked by `| tee` — the very first run reported `BUILD_EXIT=0` despite a `cmake Error at CMakeLists.txt:292 (add_subdirectory)`.)

Result: all 7 binaries built (`rl_real_a1`, `rl_real_d1`, `rl_real_g1`, `rl_real_go2`, `rl_real_l4w4`, `rl_real_lite3`, `rl_sim_mujoco`). The MuJoCo target at `cmake_build/bin/rl_sim_mujoco` is 15 MB; `ldd` resolves cleanly to our `library/inference_runtime/libtorch/lib/*` and `library/mujoco/lib/libmujoco.so.3.2.7`.

---

## Phase 3 — Running the C++ binary headlessly (Xvfb + scripted PTY)

The DGX Spark has no display server (`$DISPLAY` and `$WAYLAND_DISPLAY` both empty). The MuJoCo Simulate viewer (vendored at [src/rl_sar/library/thirdparty/mujoco_simulate/](../src/rl_sar/library/thirdparty/mujoco_simulate/)) needs a GLFW window, and the FSM keyboard reader at [src/rl_sar/library/core/rl_sdk/rl_sdk.cpp:350-410](../src/rl_sar/library/core/rl_sdk/rl_sdk.cpp#L350-L410) uses `tcsetattr` non-canonical stdin — it needs a real PTY.

Apt-installed `xvfb` (`mesa-utils` is optional for `glxinfo`). `pexpect` and `python-xlib` were `uv pip install`'d into `/tmp/rl_sar_uv_env`.

The orchestrator at [scripts/run_dance_headless.py](../scripts/run_dance_headless.py) does:

1. Start `Xvfb :99 -screen 0 1920x1080x24`.
2. Start `ffmpeg -f x11grab -i :99.0+0,0 -video_size 1280x720 ...` (only the GLFW window area — see "non-obvious finding #4" below).
3. Spawn `rl_sim_mujoco g1 scene_29dof` under `pexpect.spawn(...)` so its stdin is a PTY. Set `proc.logfile_read = run.log`.
4. `proc.expect("FSMManager.*Registered type: g1")` so we know the binary is alive.
5. Send `"0"`. Expect `Switch from RLFSMStatePassive to RLFSMStateGetUp`.
6. Use XTest (`Xlib.ext.xtest.fake_input(d, X.KeyPress, kc)`) to fake a real **Tab** keypress at the X server level, then **Shift+Tab**, to toggle both MuJoCo Simulate UI panels off ([simulate.cc:1680-1687](../src/rl_sar/library/thirdparty/mujoco_simulate/simulate.cc#L1680-L1687)). Plain `XSendEvent` is rejected by GLFW; XTest events are not.
7. Wait for `Getting up completed` line. **Then** start re-sending `"1"` every 5 s via `send_until_transition` until `Switch from RLFSMStateGetUp to RLFSMStateRLRoboMimicLocomotion` appears.
8. Same for `"3"` until `Switch from ... to RLFSMStateRLWholeBodyTrackingDance102`.
9. Sleep 180 s while the dance plays out (the dance state's `RLControl()` calls `torch::jit::forward()` at 50 Hz). Then SIGINT the binary, stop ffmpeg, stop Xvfb.

The orchestrator runs **inside the venv** (`/tmp/rl_sar_uv_env/bin/python scripts/run_dance_headless.py`) so it picks up `pexpect` and `python-xlib`.

### Four non-obvious findings discovered along the way

1. **G1's FSM does not allow direct `GetUp → dance_102`.** [fsm_g1.hpp:78-87](../src/rl_sar/fsm_robot/fsm_g1.hpp#L78-L87) only accepts `Num1` (→ locomotion) or `Num9` (→ getdown) out of GetUp. The dance is reached by `GetUp → RoboMimicLocomotion → dance_102`. The Num0→Num3 shorthand in [agent.md](../agent.md) §Step 8 is wrong for this build.
2. **`RobotControl()` calls `ClearInput()` at the end of every cycle** ([rl_sim_mujoco.cpp:222](../src/rl_sar/src/rl_sim_mujoco.cpp#L222)). `current_keyboard` reverts to `last_keyboard` (=None initially) within 1 control tick. A single keypress is wiped almost instantly. The orchestrator must re-send each FSM key until the corresponding transition log appears — implemented as `send_until_transition()`.
3. **MuJoCo Simulate viewer keyboard ≠ FSM keyboard.** Tab handling lives in the vendored simulate.cc's GLFW callback (X-level events). FSM keys (0, 1, 3, 9, …) come through `tcsetattr` non-canonical stdin reads in rl_sdk. The two paths are independent — xdotool/XTest is for UI panel toggling, the PTY is for FSM input.
4. **GLFW window is 2/3 of the Xvfb root.** [glfw_adapter.cc:60](../src/rl_sar/library/thirdparty/mujoco_simulate/glfw_adapter.cc#L60) creates the window at `(2 * vidmode.width) / 3 × (2 * vidmode.height) / 3` — so a 1920×1080 Xvfb gives a 1280×720 GLFW window pinned at (0,0). The rest of the Xvfb root is unused black. We crop ffmpeg's grab to `1280x720+0,0` to match.

### Result

[dance_102.mp4](../dance_102.mp4) — 22 MB, 1280×720 @ 15 fps, 184 s. All three FSM transitions fire successfully and verifiably (matched in the log). The dance state loads `policy.pt` and `G1_Take_102.bvh_60hz.csv` (1749 frames, 29 joints, 34.98s motion duration) and `torch::jit::forward()` runs at 50 Hz. The dance motion **progresses from 0.00% to 99.94%** of the reference motion across the capture window.

**The robot does fall** during the FSM cascade. The 1-second settle after binary spawn isn't long enough for the robot to settle in Passive (which uses `kp=0, kd=8` — damping only, no position holding). By the time GetUp's interpolation runs at high `fixed_kp`, the robot's base orientation is already off and the dance policy can't recover. This is a deployment-recipe issue, not a build issue — the C++ binary is correctly running everything end-to-end.

### Side-quest: VNC

For interactive UI access (debugging the camera view, for example), a separate launcher at [scripts/start_interactive_view.sh](../scripts/start_interactive_view.sh) boots Xvfb + `x11vnc` on :5900 + `websockify` + noVNC on :6080 + the binary. VS Code Remote-SSH auto-forwards :6080 so you open `http://localhost:6080/vnc.html` in any browser. Per user preference this was not used in the final flow but is kept for future debugging.

---

## Phase 4 — Direct offscreen rendering (the clean path)

Recommended for actually producing watchable videos. Discovered after the user pointed to a reference pattern at `/home/h_fujiwara/projects/so101-nmpc-control/docs/reference_record_script.py` (`mujoco.Renderer` + `cv2.VideoWriter`, no GLFW window).

`opencv-python-headless` was `uv pip install`'d. `MUJOCO_GL=egl` invokes NVIDIA's EGL ICD (`/usr/lib/aarch64-linux-gnu/libEGL_nvidia.so.0`) and renders directly on the **GB10 GPU**, ~150 fps wall-clock for the 29-DoF G1.

### `scripts/record_dance_offscreen.py`

Pure kinematic playback of the BVH-derived motion CSV — no physics, no policy, just `mj_kinematics` + render. Mirrors [motion_loader.cpp:LoadFromCSV](../src/rl_sar/library/core/motion_loader/motion_loader.cpp): 36 columns per row (`xyz` root pos + `xyzw` root quat permuted to `wxyz` for MuJoCo + 29 joint angles).

Two minor extensions needed beyond the so101 reference:
- The scene XML's default offscreen framebuffer is 640×480; the script bumps `model.vis.global_.offwidth/offheight` after load so `mujoco.Renderer(model, 720, 1280)` doesn't reject the request.
- Use `mj_kinematics(model, data)` (forward kinematics only) instead of `mj_step` — we're replaying recorded motion, not integrating physics.

Result: [dance_102_offscreen.mp4](../dance_102_offscreen.mp4), 11 MB, 1280×720 @ 60 fps, **29.15 s**, rendered in **11.5 s wall-clock** (~30× faster than the X11+llvmpipe capture). The robot dances smoothly without falling because there is no physics — this is the choreography the policy was trained to track.

### `scripts/record_dance_orbit.py`

Same pipeline, but each frame the camera's azimuth sweeps `args.az_start + progress * 360° × revolutions` and `cam.lookat` tracks `data.qpos[0:3] + [0, 0, args.lookat_z]` so the robot stays centered in frame even as its root translates during the dance. Tunables: `--distance`, `--elevation`, `--az-start`, `--revolutions`, `--lookat-z`.

Defaults render a single full revolution at distance=3.2 m, elevation=-12°, lookat_z=0.6 m. The camera circles the robot one full time across the duration of the motion.

Result for `dance_102`: [dance_102_orbit.mp4](../dance_102_orbit.mp4), 14 MB, 1280×720 @ 60 fps, 29.15 s.

### Gangnam Style

Same script, different `--motion`:

```bash
/tmp/rl_sar_uv_env/bin/python scripts/record_dance_orbit.py \
    --motion policy/g1/whole_body_tracking/gangnam_style/G1_gangnam_style_V01.bvh_60hz.csv \
    --output gangnam_style_orbit.mp4
```

Result: [gangnam_style_orbit.mp4](../gangnam_style_orbit.mp4), 15.8 MB, 1280×720 @ 60 fps, 32.37 s — visibly the classic horse-riding stance.

---

## Phase 5 — Git hygiene

Five MP4s ended up tracked at HEAD (~66 MB total). Added `*.mp4` to `.gitignore` and ran `git rm --cached` on them, then pushed (commit `0d75112` "chore: untrack mp4 recordings and ignore via *.mp4"). The MP4 blobs **remain in git history** in older commits — `origin/main` HEAD is clean, but a fresh clone still pulls them via the pack files. Purging history would require `git filter-repo` + force-push to `main`, not done.

The newer `dance_102_orbit.mp4` and `gangnam_style_orbit.mp4` are covered by the `*.mp4` rule and were never tracked.

---

## File inventory (this session's additions)

### Tracked in git

| Path | Purpose |
|---|---|
| [CLAUDE.md](../CLAUDE.md) | Project guidance for future Claude Code instances |
| [docs/g1_dance_mujoco_plan.md](g1_dance_mujoco_plan.md) | **This document** |
| [rl_sar_g1_dance_run_report.md](../rl_sar_g1_dance_run_report.md) | Run report with §3b "build resolved" and §7b "headless run via Xvfb" — pre-offscreen-rendering snapshot of progress |
| [scripts/run_dance_headless.py](../scripts/run_dance_headless.py) | Xvfb + pexpect + XTest orchestrator that runs the C++ rl_sim_mujoco binary headlessly and captures the GLFW window |
| [scripts/start_interactive_view.sh](../scripts/start_interactive_view.sh) | noVNC web-browser-accessible viewer (debugging aid) |
| [scripts/record_dance_offscreen.py](../scripts/record_dance_offscreen.py) | Offscreen kinematic playback via `mujoco.Renderer` + `cv2.VideoWriter` — fixed camera |
| [scripts/record_dance_orbit.py](../scripts/record_dance_orbit.py) | Same, with orbit camera tracking the robot's root |
| `.gitignore` | Added `*.mp4` rule |

### Generated / not tracked (gitignored)

| Path | Size | Notes |
|---|---|---|
| `library/inference_runtime/libtorch/` | ~970 MB | Copied from sam3 conda env + torch.libs + 4 CUDA dep libs |
| `library/inference_runtime/onnxruntime/` | small | Auto-fetched by build.sh -mj |
| `library/mujoco/` | symlinks only | Points to a `mujoco==3.2.7` wheel in `/tmp/rl_sar_uv_env` |
| `src/rl_sar_zoo/` | 568 MB | Cloned by `download_robot_descriptions.sh` |
| `cmake_build/` | 227 MB | CMake artifacts including the 7 binaries in `cmake_build/bin/` |
| `build_mujoco.log` | 12 KB | Final `./build.sh -mj` capture |
| `run.log` | 3.3 MB | Last `run_dance_headless.py` PTY capture |
| `dance_102.mp4` | 22 MB | C++ binary live, Xvfb-captured, robot falls |
| `dance_102_offscreen.mp4` | 11 MB | Python kinematic playback, fixed camera |
| `dance_102_orbit.mp4` | 14 MB | Python kinematic playback, orbit camera |
| `gangnam_style_orbit.mp4` | 15.8 MB | Same script, gangnam motion |
| `dance_102_v1_with_menus.mp4`, `_v2_left_hidden.mp4`, `_v3_full_frame.mp4` | 0.3 – 22 MB | Intermediate Xvfb-capture iterations kept for reference |
| `/tmp/rl_sar_uv_env/` | ~1.5 GB | Throwaway venv (Python 3.10) with mujoco, torch, pexpect, opencv, python-xlib |

### Outside the repo (system-level changes)

- apt packages: 18 new packages installed via `sudo apt-get install` (see "Phase 2 — apt install line" above).
- No `~/.gitconfig`, `~/.bashrc`, etc. were modified.
- No NVIDIA driver / CUDA changes.

---

## How to regenerate everything from scratch

```bash
# 1. Build the C++ binary (one-time)
cd /home/h_fujiwara/projects/rl_sar
git submodule update --init --recursive --recommend-shallow

# Stage LibTorch from the sam3 conda env (or any NEW-ABI torch ≥ 2.3)
mkdir -p library/inference_runtime
torch_root=/home/h_fujiwara/miniconda3/envs/sam3/lib/python3.10/site-packages/torch
cp -r "$torch_root" library/inference_runtime/libtorch
# For pip-style aarch64 torch wheels you also need:
# cp /path/to/site-packages/torch.libs/* library/inference_runtime/libtorch/lib/
# And the four CUDA libs if linking torch_cuda.so:
for d in cudnn nccl nvshmem cusparselt; do
  cp "$torch_root/../../nvidia/$d/lib/"*.so* library/inference_runtime/libtorch/lib/ 2>/dev/null
done

# Stage MuJoCo 3.2.7 from the official PyPI wheel
uv venv /tmp/rl_sar_uv_env --python 3.10
uv pip install --python /tmp/rl_sar_uv_env/bin/python "mujoco==3.2.7" torch==2.3.0 numpy
wheel=/tmp/rl_sar_uv_env/lib/python3.10/site-packages/mujoco
mkdir -p library/mujoco/lib
ln -sf "$wheel/include" library/mujoco/include
ln -sf "$wheel/libmujoco.so.3.2.7" library/mujoco/lib/libmujoco.so.3.2.7
ln -sf libmujoco.so.3.2.7 library/mujoco/lib/libmujoco.so
echo "3.2.7" > library/mujoco/VERSION_NUMBER

# Install the apt deps (one sudo prompt)
sudo apt-get install -y libyaml-cpp-dev libeigen3-dev libboost-all-dev \
    libspdlog-dev libfmt-dev libtbb-dev liblcm-dev libglfw3 libglfw3-dev \
    libxrandr2 libxinerama1 libxcursor1 libxi6 libgl1-mesa-dri libegl1 \
    libglvnd-dev python3-dev python3-numpy

# Build
CMAKE_PREFIX_PATH="$PWD/library/inference_runtime/libtorch/share/cmake" \
CUDAToolkit_ROOT=/usr/local/cuda \
bash -o pipefail -c './build.sh -mj 2>&1 | tee build_mujoco.log'

# 2. Render videos (any time, fast)
uv pip install --python /tmp/rl_sar_uv_env/bin/python opencv-python-headless

# dance_102 fixed-camera
/tmp/rl_sar_uv_env/bin/python scripts/record_dance_offscreen.py

# dance_102 orbit
/tmp/rl_sar_uv_env/bin/python scripts/record_dance_orbit.py

# gangnam style orbit
/tmp/rl_sar_uv_env/bin/python scripts/record_dance_orbit.py \
    --motion policy/g1/whole_body_tracking/gangnam_style/G1_gangnam_style_V01.bvh_60hz.csv \
    --output gangnam_style_orbit.mp4

# (Optional) C++ binary live with Xvfb + key automation
sudo apt-get install -y xvfb
uv pip install --python /tmp/rl_sar_uv_env/bin/python pexpect python-xlib
/tmp/rl_sar_uv_env/bin/python scripts/run_dance_headless.py
```

---

## Open questions / not done

- **Robot falls during the live C++ run.** The pipeline runs end-to-end but the rapid `GetUp → Locomotion → dance_102` FSM cascade plus slow software-rendered RTF destabilises the base. Possible fixes: longer settle in Passive before sending `0`; longer wait after `1` (locomotion) before sending `3`; or interactive view via [start_interactive_view.sh](../scripts/start_interactive_view.sh) so a human can time the transitions. Offscreen kinematic playback sidesteps the problem entirely.
- **No kung-fu / martial-arts policy** in rl_sar. The agent.md explicitly defers this to a separate **PBHC / KungfuBot** training pipeline (not in scope). `dance_102` does *look* martial-arts-flavored in mid-frame extensions, but the name "Take 102" is just a BVH session number, not a content label.
- **Old MP4 blobs still in git history.** `git filter-repo` + force-push to `origin/main` would purge them but rewrites public history; not done.
- **Upstream PR opportunity**: extend [scripts/download_inference_runtime.sh](../scripts/download_inference_runtime.sh) with an aarch64-non-Jetson branch that re-uses the existing `create_libtorch_from_pytorch` body from [scripts/install_pytorch_jetson.sh:192-286](../scripts/install_pytorch_jetson.sh#L192-L286). Would unblock other DGX Spark / Grace users.

---

## Key files to reference for future work

- [src/rl_sar/library/core/rl_sdk/rl_sdk.cpp:350-410](../src/rl_sar/library/core/rl_sdk/rl_sdk.cpp#L350-L410) — termios non-canonical stdin keyboard reader
- [src/rl_sar/library/core/motion_loader/motion_loader.cpp](../src/rl_sar/library/core/motion_loader/motion_loader.cpp) — BVH CSV format (36 cols, quat permute)
- [src/rl_sar/library/thirdparty/mujoco_simulate/simulate.cc:1680](../src/rl_sar/library/thirdparty/mujoco_simulate/simulate.cc#L1680) — Tab / Shift+Tab UI panel toggle
- [src/rl_sar/library/thirdparty/mujoco_simulate/glfw_adapter.cc:60](../src/rl_sar/library/thirdparty/mujoco_simulate/glfw_adapter.cc#L60) — `(2 * vidmode.width) / 3` window size rule
- [src/rl_sar/fsm_robot/fsm_g1.hpp:78-87](../src/rl_sar/fsm_robot/fsm_g1.hpp#L78-L87) — GetUp's outgoing transitions (Num1 / Num9 only)
- [src/rl_sar/fsm_robot/fsm_g1.hpp:290](../src/rl_sar/fsm_robot/fsm_g1.hpp#L290) — `rl.config_name = "whole_body_tracking/dance_102"`
- [src/rl_sar/src/rl_sim_mujoco.cpp:222](../src/rl_sar/src/rl_sim_mujoco.cpp#L222) — `ClearInput()` per cycle
- [scripts/download_inference_runtime.sh:101-106](../scripts/download_inference_runtime.sh#L101-L106) — the aarch64 non-Jetson bail-out we worked around
- `/home/h_fujiwara/projects/so101-nmpc-control/docs/reference_record_script.py` — the `mujoco.Renderer` + `cv2.VideoWriter` pattern we copied
- `/home/h_fujiwara/projects/mujoco/python/README.md` — DeepMind's "use the PyPI wheel" install guidance
