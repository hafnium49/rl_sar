# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`rl_sar` ("simulation and real") is a C++ deployment framework for reinforcement-learning controllers on quadruped, wheeled, and humanoid robots. Policies are *trained* externally (IsaacGym / IsaacLab / robot_lab) and *deployed* here in Gazebo, MuJoCo, or on real hardware. The Python training side is intentionally out of scope — see `agent.md` in the repo root for a worked simulation-only example (G1 dance via MuJoCo).

## Build system

There are three mutually exclusive build modes orchestrated by [build.sh](build.sh):

```bash
./build.sh                # ROS mode: catkin (noetic) or colcon (foxy/humble)
./build.sh package1 ...   # ROS mode, specific packages only
./build.sh -m             # CMake-only, hardware deployment (no simulator)
./build.sh -mj            # CMake + MuJoCo simulation
./build.sh -c             # Clean (removes symlinks AND build/cmake_build/devel/install/log)
```

Important behaviors:

- **ROS1 vs ROS2 is selected by `$ROS_DISTRO`**, not by a flag. Each ROS package ships `package.ros1.xml` and `package.ros2.xml`; [build.sh](build.sh) symlinks the right one to `package.xml` before building. If you find a stale `package.xml` symlink causing build failures across ROS versions, `./build.sh -c` is the fix.
- The ROS and CMake builds drop artifacts in **different directories**: `devel/`+`build/` (catkin), `install/`+`build/`+`log/` (colcon), or `cmake_build/{bin,lib}/` (CMake). The build script auto-detects an incompatible mix and offers to clean.
- `./build.sh -m` / `-mj` produces standalone binaries at `cmake_build/bin/rl_real_<robot>` and `cmake_build/bin/rl_sim_mujoco` — these have no ROS dependency and are the only build path that works on the onboard Jetson and on macOS (MuJoCo only).
- Build artifacts are placed in the **repo root**, not inside `src/rl_sar/`. The CMakeLists at [src/rl_sar/CMakeLists.txt](src/rl_sar/CMakeLists.txt) is run with `-B cmake_build` from the repo root.

Inference runtimes (LibTorch and/or ONNX Runtime) are downloaded to `library/inference_runtime/` by [scripts/download_inference_runtime.sh](scripts/download_inference_runtime.sh) before the build. The CMake detects which is present via `USE_TORCH` / `USE_ONNX` and compiles whatever it finds; Jetson auto-disables ONNX. MuJoCo and robot descriptions are similarly auto-fetched.

## Running

After `./build.sh`:

```bash
# Gazebo (two terminals — rl_sim is REQUIRED or the robot falls)
source devel/setup.bash      # or install/setup.bash for ROS2
roslaunch rl_sar gazebo.launch rname:=<ROBOT>          # ROS1
ros2 launch rl_sar gazebo.launch.py rname:=<ROBOT>     # ROS2
rosrun rl_sar rl_sim                                   # ROS1
ros2 run rl_sar rl_sim                                 # ROS2

# MuJoCo (CMake build, no ROS)
./cmake_build/bin/rl_sim_mujoco <ROBOT> <SCENE>
# e.g. ./cmake_build/bin/rl_sim_mujoco g1 scene_29dof

# Real hardware
./cmake_build/bin/rl_real_<robot> [<NETWORK_INTERFACE>] [extra-args]
```

Supported `<ROBOT>` keys live as policy/`<robot>`/ directories (a1, b2, b2w, d1, g1, go2, go2w, gr1t1, gr1t2, l4w4, lite3, tita). Not every robot supports every simulator — see the table in [README.md](README.md).

## Architecture

### Control flow (single robot)

```
main()  ──►  RL_Sim / RL_Sim_MuJoCo / RL_Real_<robot>     (concrete subclass per target)
              │                                            ↑ inherits RL (library/core/rl_sdk)
              ├─► GetState()      ← simulator/SDK callbacks fill robot_state
              ├─► RobotControl()  ← keyboard/joy/gamepad input → control.x/y/yaw
              ├─► RunModel() loop ← drives FSM.Run() at decimation rate
              │     └─► FSMState::Run() → if RL state, calls RLControl()
              │           └─► RL::Forward()  ← inference (LibTorch or ONNX)
              │           └─► ComputeOutput  ← scale, clip, kp/kd shape
              │           └─► output_dof_*_queue (TBB concurrent queue)
              └─► SetCommand() ← consumes queue, sends to simulator/SDK
```

The base [`RL` class](src/rl_sar/library/core/rl_sdk/rl_sdk.hpp) is abstract: `Forward()`, `GetState()`, `SetCommand()` are pure virtual. Subclasses bind it to a specific transport (ROS topic, Unitree SDK, Lite3 SDK, Agibot SDK, MuJoCo callback).

### FSM-per-robot pattern

Each robot has its own state machine under [src/rl_sar/fsm_robot/fsm_\<robot\>.hpp](src/rl_sar/fsm_robot/) plus an aggregator [fsm_all.hpp](src/rl_sar/fsm_robot/fsm_all.hpp). Robots register a factory via the `REGISTER_FSM_FACTORY` macro at the bottom of each header; the `FSMManager` singleton resolves `rl.robot_name` → the right factory at startup. Common states are `RLFSMStatePassive`, `RLFSMStateGetUp`, `RLFSMStateGetDown`, `RLFSMStateRLLocomotion`; skill-rich robots (g1, b2, go2w, d1) add per-skill states like `RLFSMStateRLDance102` / `RLFSMStateRLGangnamStyle` etc.

**Critical:** skill-to-key mapping is hard-coded in each robot's FSM header (not config). For example G1's Num1 → `robomimic/locomotion`, Num2 → `robomimic/charleston`, Num3 → `whole_body_tracking/dance_102`, Num4 → `whole_body_tracking/gangnam_style` — read [fsm_g1.hpp](src/rl_sar/fsm_robot/fsm_g1.hpp) to confirm before assuming. The FSM also sets `rl.config_name`, which becomes the subfolder under `policy/<robot>/` that gets loaded.

### Policy / config layout

```
policy/<ROBOT>/base.yaml              # PHYSICAL joint order & names — MUST match the real robot
policy/<ROBOT>/<CONFIG>/config.yaml   # Training-time obs/action layout, kp/kd, scales
policy/<ROBOT>/<CONFIG>/<name>.pt     # LibTorch JIT, OR
policy/<ROBOT>/<CONFIG>/<name>.onnx   # ONNX Runtime
```

`base.yaml` declares the robot's physical joint order. `config.yaml` declares the trained policy's joint order indirectly via `joint_mapping: [...]`, which **permutes the action vector** from training order into physical order. **A wrong `joint_mapping` will silently produce dangerous behavior on hardware** — README's Lite3 section flags this explicitly. If you train policies in `robot_lab`, its `joint_names` in the env cfg must match `base.yaml`'s `joint_names` here.

Other config-driven knobs that change behavior without code edits: `observations` (list of obs terms), `observations_history` (history buffer indices and ordering by "time" vs "term"), `clip_obs`, `clip_actions_lower/upper`, `action_scale` (per-joint vector), `rl_kp` / `rl_kd` (used while policy is active) vs `fixed_kp` / `fixed_kd` (used during get-up / passive). `dt * decimation` is the policy step.

### Core libraries (`src/rl_sar/library/core/`)

| Lib | What to know |
|---|---|
| `rl_sdk` | The `RL` base class, `RobotState` / `RobotCommand` templates, `YamlParams::Get<T>` (note its warning: don't iterate the returned vector directly — it's a temporary). |
| `inference_runtime` | Wraps LibTorch and ONNX behind one `Model` interface via `#ifdef USE_TORCH` / `USE_ONNX`. |
| `observation_buffer` | History stacking. Order controlled by `observations_history_priority`. |
| `motion_loader` | Reference motion playback (`.npz` / `.csv`) for whole-body-tracking and mimic skills. |
| `fsm` | Generic FSM + factory registry. Robot-specific states live in `src/rl_sar/fsm_robot/`. |
| `loop` | `LoopFunc` — periodic threaded callback for control / RL / plot loops. |
| `vector_math` | Quaternion / euler / projection helpers. |
| `logger`, `matplotlibcpp` | Colored stdout helpers; live plotting (compile-time `#define PLOT` in `rl_sim.hpp`). |

Thirdparty SDKs (Unitree legged_sdk / sdk2, Lite3 motionsdk, Agibot D1, joystick, MuJoCo viewer) live under `src/rl_sar/library/thirdparty/`. Robot descriptions (URDF/xacro/meshes) and the `rl_sar_zoo` package are fetched into `src/` by [scripts/download_robot_descriptions.sh](scripts/download_robot_descriptions.sh) — they are **not** committed to this repo.

### ROS packages

Three packages under `src/`: `rl_sar` (the controllers), `robot_joint_controller` (custom ros_control plugin used by Gazebo), `robot_msgs` (MotorCommand/MotorState). Whichever is being compiled, `build.sh` toggles its `package.xml` symlink so the same source tree builds against ROS1 or ROS2.

## Adding a robot

Per [README.md](README.md#add-your-robot), the **filenames must match exactly** (case-sensitive). Minimum touched files:

```
src/rl_sar_zoo/<ROBOT>_description/{CMakeLists.txt, package.ros1.xml, package.ros2.xml,
                                     xacro/robot.xacro, xacro/gazebo.xacro,
                                     config/robot_control{,_ros2}.yaml}
policy/<ROBOT>/base.yaml
policy/<ROBOT>/<CONFIG>/config.yaml
policy/<ROBOT>/<CONFIG>/<POLICY>.{pt,onnx}
src/rl_sar/fsm_robot/fsm_<ROBOT>.hpp        # plus include in fsm_all.hpp
src/rl_sar/src/rl_real_<ROBOT>.cpp          # plus matching include/rl_real_<ROBOT>.hpp
```

Then add the new executable to [src/rl_sar/CMakeLists.txt](src/rl_sar/CMakeLists.txt) following the existing `add_executable(rl_real_<robot> ...)` patterns (separate ROS1 / ROS2 / CMake-only blocks).

## Things to be aware of

- **LibTorch JIT export is required** for `.pt` files (not raw `state_dict`). Use [src/rl_sar/scripts/convert_policy.py](src/rl_sar/scripts/convert_policy.py) when converting from training checkpoints.
- **CSV logging for actuator-net training** is gated by `#define CSV_LOGGER` at the top of `rl_real_<robot>.hpp` / `rl_sim.hpp`. The trainer is [src/rl_sar/scripts/actuator_net.py](src/rl_sar/scripts/actuator_net.py).
- **Default joints in `RLFSMStatePassive` use `kp=0, kd=8`** — that's intentional damping, not a bug.
- Tests in [src/rl_sar/test/](src/rl_sar/test/) (`test_observation_buffer`, `test_vector_math`, `test_inference_runtime`) are commented out in CMakeLists; uncomment the relevant `add_executable` block to build them.
- `agent.md` in repo root is a **task brief** (mission spec for running pretrained G1 dance in MuJoCo), not authoritative project documentation. Treat it as one example workflow, not a constraint on other work.
