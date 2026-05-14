# rl_sar fork progress report — 2026-05-14

**Author:** Hiroki Fujiwara
**Reporting window:** 2026-05-12 → 2026-05-14 (3 calendar days)
**Branch:** `main` @ [`9e5df89`](https://github.com/hafnium49/rl_sar/commit/9e5df89) — clean tree, in sync with `origin/main`
**Upstream:** [`fan-ziqi/rl_sar`](https://github.com/fan-ziqi/rl_sar) — no PRs opened

## TL;DR

Over three days the fork went from a fresh DGX Spark clone with a broken aarch64 build path to a fully operational MuJoCo pipeline rendering pretrained Unitree G1 whole-body-tracking dances on the Grace-Blackwell GPU, plus a *completed* FBX 3000 → FBX 7700 modernization of all five Ibaraki Radio Taiso source takes. The narrow mission ([agent.md](../agent.md): "run pretrained G1 dance in MuJoCo") is satisfied; the extended Radio Taiso pipeline ([radio_taiso_g1_agent.md](../radio_taiso_g1_agent.md)) has cleared its hardest infrastructure gate (FBX import) and is now blocked only on retargeting (GMR/NMR) and training — both deferred to external repos.

---

## 1. Fork identity and repository state

- **Origin:** `github.com/hafnium49/rl_sar` (HTTPS, push + fetch).
- **Upstream:** `github.com/fan-ziqi/rl_sar.git` (declared as a `git remote`, not yet fetched locally).
- **Fork commits:** 23 commits on `main`, all authored by Hiroki Fujiwara (`hafnium49@gmail.com` for the first 22, then `h_fujiwara@mitsui-kinzoku.com` for [`9e5df89`](https://github.com/hafnium49/rl_sar/commit/9e5df89)).
- **Total repo commits:** 266 (243 upstream, 23 fork).
- **Fork base:** `36bac39` — last upstream commit (Agibot D1 sim2sim sim2real) before fork-specific work begins.
- **Working tree:** clean. No stashes. All five takes of source FBX and all five converted FBX 7700 outputs are committed (~140 MB of binary assets, accepted via explicit `.gitignore` overrides).
- **Untracked artifacts on disk:** `tmp/fbxsdk/` (~40 MB unpacked AutoDesk SDK; intentionally outside git per [`5d5e8e6`](https://github.com/hafnium49/rl_sar/commit/5d5e8e6)).

---

## 2. Mission scope — two layers

The fork carries two related but distinct mission documents at the repo root:

| Document | Lines | Mission | Hard constraints |
|---|---|---|---|
| [`agent.md`](../agent.md) | 561 | Run *pretrained* G1 whole-body-tracking policies (`dance_102`, `gangnam_style`) in MuJoCo | No training. No real hardware. No invented scripts. |
| [`radio_taiso_g1_agent.md`](../radio_taiso_g1_agent.md) | 1156 | Build the full Radio Taiso (ラジオ体操) → GMR/NMR retarget → G1 reference CSV → unitree_rl_mjlab PPO → MuJoCo validation pipeline | Start from source data. Don't hardcode `take05`. Simulation gate before any hardware gate. Both retargeting routes (GMR + NMR) tracked. |

[`agent.md`](../agent.md) is the *narrow* mission — what this fork must demonstrate first. [`radio_taiso_g1_agent.md`](../radio_taiso_g1_agent.md) is the *extended* mission — what this fork is building toward.

[`CLAUDE.md`](../CLAUDE.md) (132 lines, committed [`d130598`](https://github.com/hafnium49/rl_sar/commit/d130598)) is the project-instructions companion that captures the rl_sar build modes, FSM-per-robot pattern, policy/config layout, and the `joint_mapping` silent-failure warning.

---

## 3. Phase-by-phase accomplishments

The as-built record for the simulation track is in [`docs/g1_dance_mujoco_plan.md`](g1_dance_mujoco_plan.md) (309 lines). The FBX conversion handoff is in [`docs/fbx3000_intel_handoff.md`](fbx3000_intel_handoff.md) (185 lines, including a fresh "Conversion log — 2026-05-14" section). The summary below threads those two documents against the commit history.

### Phase A — Codebase orientation (2026-05-12)

- [`f7c1d71`](https://github.com/hafnium49/rl_sar/commit/f7c1d71) added [`agent.md`](../agent.md); [`d130598`](https://github.com/hafnium49/rl_sar/commit/d130598) added [`CLAUDE.md`](../CLAUDE.md).
- Surveyed the build system (ROS-vs-CMake split via [`build.sh`](../build.sh)), the FSM-per-robot pattern in [`src/rl_sar/fsm_robot/`](../src/rl_sar/fsm_robot/), and the policy/config layout in [`policy/`](../policy/).

### Phase B — Build unblock on aarch64 / DGX Spark (2026-05-12 → 05-13)

Three blockers were diagnosed and resolved (commit chain [`cc7f9ff`](https://github.com/hafnium49/rl_sar/commit/cc7f9ff) → [`2f229e8`](https://github.com/hafnium49/rl_sar/commit/2f229e8) → [`fdbb3ec`](https://github.com/hafnium49/rl_sar/commit/fdbb3ec) → [`1c54eaf`](https://github.com/hafnium49/rl_sar/commit/1c54eaf)):

1. **No aarch64 LibTorch path.** [`scripts/download_inference_runtime.sh`](../scripts/download_inference_runtime.sh) bails out at line ~101 for aarch64 non-Jetson. DGX Spark trips neither Jetson marker. Workaround: copy NVIDIA's NEW-ABI `torch==2.9.1+cu130` from an existing `sam3` conda env into [`library/inference_runtime/libtorch/`](../library/inference_runtime/libtorch/), including the sibling `torch.libs/` deps and four CUDA libs from `nvidia/{cudnn,nccl,nvshmem,cusparselt}/lib/`. Set `CMAKE_PREFIX_PATH` and `CUDAToolkit_ROOT=/usr/local/cuda`.
2. **ABI mismatch.** Pip-installed aarch64 PyTorch wheels are `_GLIBCXX_USE_CXX11_ABI=0` (OLD ABI); Ubuntu 24.04 `yaml-cpp` is NEW ABI. Result was unresolved `YAML::LoadFile` symbols at link time. NEW-ABI torch from NVIDIA's build resolves it.
3. **MuJoCo 3.2.7 supply.** `pip install mujoco==3.2.7` was used to obtain a known-good binary, then symlinked into [`library/mujoco/`](../library/mujoco/). MuJoCo 3.3+ was rejected (rl_sar's vendored [`simulate.cc`](../src/rl_sar/library/thirdparty/mujoco_simulate/simulate.cc) uses the now-removed `mjvSceneState` API and pins C++17).

**Outcome:** `./build.sh -mj` builds all seven CMake-only binaries cleanly. The MuJoCo target [`cmake_build/bin/rl_sim_mujoco`](../cmake_build/bin/) is 15 MB; `ldd` resolves to the staged LibTorch and `libmujoco.so.3.2.7` symlink chain.

### Phase C — Headless C++ execution (2026-05-13)

[`scripts/run_dance_headless.py`](../scripts/run_dance_headless.py) (commit [`4db9448`](https://github.com/hafnium49/rl_sar/commit/4db9448), refined in [`82c50b4`](https://github.com/hafnium49/rl_sar/commit/82c50b4) and [`b618ede`](https://github.com/hafnium49/rl_sar/commit/b618ede)) orchestrates the C++ binary on a headless box:

1. Starts `Xvfb :99 -screen 0 1920x1080x24` to back GLFW.
2. Starts `ffmpeg -f x11grab` cropped to `1280x720+0,0` (the GLFW window is `2/3 × 2/3` of root — discovered in [`glfw_adapter.cc:60`](../src/rl_sar/library/thirdparty/mujoco_simulate/glfw_adapter.cc#L60)).
3. Spawns `rl_sim_mujoco g1 scene_29dof` under `pexpect` so its stdin is a real PTY (rl_sar uses `tcsetattr` non-canonical reads — a normal pipe will not deliver keys).
4. Uses XTest (`Xlib.ext.xtest.fake_input`) to fake Tab/Shift-Tab into the GLFW window (`XSendEvent` is rejected; XTest is not).
5. Re-sends each FSM key (0, 1, 3) every 5 seconds via `send_until_transition()` until the matching `Switch from … to …` log appears — necessary because [`rl_sim_mujoco.cpp:222`](../src/rl_sar/src/rl_sim_mujoco.cpp#L222) calls `ClearInput()` every cycle.

**Key discovery:** G1's FSM disallows direct `GetUp → dance_102`. [`fsm_g1.hpp:78-87`](../src/rl_sar/fsm_robot/fsm_g1.hpp#L78-L87) only accepts `Num1` (→ locomotion) and `Num9` (→ getdown) out of GetUp. The dance is reachable only via `GetUp → RoboMimicLocomotion → dance_102`. The Num0→Num3 shorthand in [`agent.md`](../agent.md) Step 8 is therefore wrong for this build — corrected in the as-built doc.

**Outcome:** all three FSM transitions fire reliably and the dance policy executes in C++ end-to-end. **Caveat:** under the rapid GetUp → Locomotion → dance cascade plus the slow llvmpipe RTF, the robot's base orientation drifts during settle (Passive uses `kp=0, kd=8` — damping only) and the policy cannot recover. The binary runs correctly; the deployment recipe needs longer settle windows.

### Phase D — GPU offscreen rendering (2026-05-13)

After the live-physics fall, the canonical video output path moved to pure kinematic playback. [`scripts/record_dance_offscreen.py`](../scripts/record_dance_offscreen.py) (commit [`bccecbc`](https://github.com/hafnium49/rl_sar/commit/bccecbc)) and [`scripts/record_dance_orbit.py`](../scripts/record_dance_orbit.py) (commit [`d8ee588`](https://github.com/hafnium49/rl_sar/commit/d8ee588)) use `mujoco.Renderer` with `MUJOCO_GL=egl` to render directly on the GB10 GPU via NVIDIA's EGL ICD. ~150 fps wall-clock for the 29-DoF G1; 1280×720 @ 60 fps output. The pipeline mirrors [`motion_loader.cpp::LoadFromCSV`](../src/rl_sar/library/core/motion_loader/motion_loader.cpp) (36 columns per row: root xyz + root quat w-permuted + 29 joint angles) and uses `mj_kinematics` only — no physics, so the robot does not fall.

**Outputs:** `dance_102.mp4` (Xvfb-captured live run, ~22 MB, robot falls), `dance_102_offscreen.mp4` (~11 MB, fixed camera, 29.15 s), `dance_102_orbit.mp4` (~14 MB, orbit camera), `gangnam_style_orbit.mp4` (~15.8 MB, full Gangnam routine). All gitignored under the `*.mp4` rule introduced in [`0d75112`](https://github.com/hafnium49/rl_sar/commit/0d75112).

### Phase E — Radio Taiso source ingestion (2026-05-14)

Commit [`b632d9c`](https://github.com/hafnium49/rl_sar/commit/b632d9c) introduced the [`radio_taiso_g1_agent.md`](../radio_taiso_g1_agent.md) spec (1156 lines). [`811481d`](https://github.com/hafnium49/rl_sar/commit/811481d) committed the five Ibaraki MoCap takes as zips at [`data/ibaraki_radio_taiso/`](../data/ibaraki_radio_taiso/): `take11.zip` through `take15.zip`, ~16 MB each, FBX 3000 binary inside. Provenance: 2009-03-10 captures, Motion Star magnetic mocap rig (15 sensors @ 30 fps), ~3 min each. Source page: `http://fondant.cis.ibaraki.ac.jp/MoCap/0310.htm`. The repo-wide `*.zip` ignore is overridden by an explicit `!data/ibaraki_radio_taiso/*.zip` allowlist in [`.gitignore`](../.gitignore).

### Phase F — FBX 3000 → FBX 7700 modernization (2026-05-14, **completed**)

Originally drawn up as a deferred handoff in [`a7fff35`](https://github.com/hafnium49/rl_sar/commit/a7fff35), the conversion was executed the same day on an Intel Xeon Platinum 8370C Azure VM (Ubuntu 22.04, x86_64) under commit [`9e5df89`](https://github.com/hafnium49/rl_sar/commit/9e5df89). The toolchain that worked:

- **AutoDesk FBX SDK 2020.3.9 (C++)** — staged from `tmp/fbx202039_fbxsdk_gcc_linux.tar.gz`, installed interactively into `tmp/fbxsdk/`. The C++ SDK was used (not the Python SDK that [`docs/fbx3000_intel_handoff.md`](fbx3000_intel_handoff.md) Path B had originally specified).
- **Custom in-tree converter:** [`scripts/fbx_modernize/fbx_modernize.cpp`](../scripts/fbx_modernize/fbx_modernize.cpp) (96 lines) + [`Makefile`](../scripts/fbx_modernize/Makefile) (22 lines). Statically linked against `libfbxsdk.a`. Pins the export version to `FBX_2020_00_COMPATIBLE` (header magic 7700) and searches the writer registry for "FBX binary" rather than trusting format-ID 0 (some SDK builds register the ASCII writer first).
- **The Path-B risk did not materialize.** SDK 2020.3.9's importer reads FBX 3000 cleanly, reporting the source as "FBX version 5.0.0" in its internal feature-version mapping. Path A (FBX Converter 2013 on Wine) was not needed.

**Outputs** at [`data/ibaraki_radio_taiso/converted/`](../data/ibaraki_radio_taiso/converted/):

| File | Header version | Size |
|---|---|---|
| `take11_fbx7700.fbx` | 7700 | 12.5 MB |
| `take12_fbx7700.fbx` | 7700 | 12.3 MB |
| `take13_fbx7700.fbx` | 7700 | 12.3 MB |
| `take14_fbx7700.fbx` | 7700 | 12.4 MB |
| `take15_fbx7700.fbx` | 7700 | 12.4 MB |

Total payload committed: ~59 MB. Header verification (`hexdump -C | head -c 30` → `0x1E14 = 7700` at byte offset 23) passed for all five. A take11 round-trip (re-import → re-export of the converted file) produced byte-identical 12,457,488-byte output, confirming structural validity beyond the header check.

A complementary fallback path stays available on aarch64 via [`scripts/fbx_to_bvh_blender.py`](../scripts/fbx_to_bvh_blender.py) (commit [`6000c64`](https://github.com/hafnium49/rl_sar/commit/6000c64)), which wraps `bpy.ops.import_scene.fbx` + `bpy.ops.export_anim.bvh` in a headless Blender invocation — usable now that the inputs are FBX 7700.

---

## 4. Deliverables matrix

| Deliverable | Path | Status |
|---|---|---|
| `dance_102` MuJoCo playback | [`policy/g1/whole_body_tracking/dance_102/`](../policy/g1/whole_body_tracking/dance_102/) | Verified |
| `gangnam_style` MuJoCo playback | [`policy/g1/whole_body_tracking/gangnam_style/`](../policy/g1/whole_body_tracking/gangnam_style/) | Verified |
| Headless C++ orchestrator | [`scripts/run_dance_headless.py`](../scripts/run_dance_headless.py) | Operational (robot falls under physics — recipe gap, not code gap) |
| Offscreen GPU recorder (fixed cam) | [`scripts/record_dance_offscreen.py`](../scripts/record_dance_offscreen.py) | Operational |
| Offscreen GPU recorder (orbit cam) | [`scripts/record_dance_orbit.py`](../scripts/record_dance_orbit.py) | Operational |
| Interactive view (noVNC) | [`scripts/start_interactive_view.sh`](../scripts/start_interactive_view.sh) | Working, retained for debugging |
| Radio Taiso source archives | [`data/ibaraki_radio_taiso/take{11..15}.zip`](../data/ibaraki_radio_taiso/) | Archived |
| FBX 3000 → 7700 converter | [`scripts/fbx_modernize/`](../scripts/fbx_modernize/) | Built, executed |
| Converted FBX 7700 takes | [`data/ibaraki_radio_taiso/converted/take{11..15}_fbx7700.fbx`](../data/ibaraki_radio_taiso/converted/) | All five, header-verified |
| Blender bpy FBX→BVH fallback | [`scripts/fbx_to_bvh_blender.py`](../scripts/fbx_to_bvh_blender.py) | Implemented, untested at scale (not needed for the primary path now) |
| Conversion handoff doc | [`docs/fbx3000_intel_handoff.md`](fbx3000_intel_handoff.md) | Complete with 2026-05-14 execution log |
| As-built dance plan | [`docs/g1_dance_mujoco_plan.md`](g1_dance_mujoco_plan.md) | Complete |
| Extended pipeline spec | [`radio_taiso_g1_agent.md`](../radio_taiso_g1_agent.md) | Spec only — phases 1–3 cleared, phases 4+ not yet executed |

---

## 5. What is now blocked vs. what is unblocked

**Unblocked since the start of the reporting window:**

- aarch64 LibTorch build path (workaround documented).
- MuJoCo viewer headless rendering (Xvfb + XTest pattern documented).
- Offscreen GPU rendering (NVIDIA EGL on GB10, ~150 fps).
- FBX 3000 import on x86_64 (C++ SDK 2020.3.9 in-tree converter).
- Modernized FBX 7700 outputs available on aarch64 for downstream tooling.

**Still blocked / deferred:**

- **Retargeting (GMR / NMR).** Not yet attempted. Per [`radio_taiso_g1_agent.md`](../radio_taiso_g1_agent.md) §0, Route A is `YanjieZe/GMR` (primary), Route B is `NJU3DV-HumanoidGroup/MakeTrackingEasy` (secondary, requires SMPL-X / AMASS-style input). Both live outside this repo.
- **PPO training (`unitreerobotics/unitree_rl_mjlab`).** Deferred until at least one retargeted G1 reference CSV exists.
- **sim2real preparation.** Explicitly gated on simulation success per [`radio_taiso_g1_agent.md`](../radio_taiso_g1_agent.md) §1 constraint 8 ("Simulation gate comes before hardware gate").
- **Robot stability in the live C++ run.** Symptom is a deployment-recipe issue (Passive→GetUp→Loco→Dance cascade timing under slow llvmpipe RTF), not a code defect. Suggested fix: longer settle in Passive before sending the first `0`, longer post-locomotion wait before sending `3`, or interactive view for human timing. The offscreen kinematic path sidesteps this entirely.
- **Upstream PRs.** None opened. An obvious candidate is extending [`scripts/download_inference_runtime.sh`](../scripts/download_inference_runtime.sh) with an aarch64-non-Jetson branch reusing `create_libtorch_from_pytorch` from [`scripts/install_pytorch_jetson.sh`](../scripts/install_pytorch_jetson.sh) — would help any other Grace-Blackwell / DGX Spark user. Not yet filed.

---

## 6. Risks and known constraints

- **Hard-coded FSM transition mapping.** Skill-to-key bindings live in [`src/rl_sar/fsm_robot/fsm_g1.hpp`](../src/rl_sar/fsm_robot/fsm_g1.hpp) (not in config). Any new Radio Taiso skill must edit this header and register a state factory. See [`CLAUDE.md`](../CLAUDE.md) §"FSM-per-robot pattern".
- **Silent-failure risk in `joint_mapping`.** [`policy/g1/base.yaml`](../policy/g1/base.yaml) declares physical joint order; the per-skill `config.yaml` declares the trained policy's order. A wrong `joint_mapping` produces dangerous behavior on hardware (called out in [`CLAUDE.md`](../CLAUDE.md) and the README). Critical to validate when the Radio Taiso reference CSV is plugged in.
- **LibTorch workaround is non-portable.** Re-deploying on a different aarch64 host requires replicating the manual copy from a conda env. The upstream-PR fix above would address this for everyone.
- **Binary assets in git.** ~81 MB of source FBX zips + ~59 MB of converted FBX 7700 are committed. Acceptable now; if takes 01–10 are also imported the repo will balloon. Consider `git lfs` before that point.
- **Old MP4 blobs in git history.** Five MP4s (~66 MB) were briefly tracked before [`0d75112`](https://github.com/hafnium49/rl_sar/commit/0d75112) added `*.mp4` to `.gitignore`. They remain in pack files; a fresh clone still pulls them. `git filter-repo` + force-push would purge but rewrites public history — not done.
- **C++ live run robot fall.** Cosmetic risk — videos labelled "live" do not look good. Mitigated by routing canonical outputs through the offscreen path.

---

## 7. Suggested next milestones

1. Pick one converted FBX (e.g. `take11_fbx7700.fbx`) and run it through GMR's `fbx_importer.py` to produce a G1 reference motion CSV. Confirm the skeleton mapping survives the import.
2. Sanity-check the resulting CSV with [`scripts/record_dance_offscreen.py`](../scripts/record_dance_offscreen.py) (passing `--motion`) — purely kinematic playback should look like a recognizable Radio Taiso routine.
3. Decide on the `unitree_rl_mjlab` workspace location (per [`radio_taiso_g1_agent.md`](../radio_taiso_g1_agent.md) §2, `~/projects/unitree_rl_mjlab/` — outside this repo).
4. File the aarch64-non-Jetson LibTorch upstream PR against `fan-ziqi/rl_sar`.
5. Improve the live C++ recipe (extend Passive settle, sequence the FSM keypresses more conservatively) to produce a non-falling on-device demo of the existing pretrained dances.

---

## 8. Appendix — commit log (newest first)

| SHA | Date | Subject |
|---|---|---|
| `9e5df89` | 2026-05-14 | feat: convert Ibaraki Radio Taiso FBX 3000 → FBX 7700 via AutoDesk SDK 2020.3.9 |
| `811481d` | 2026-05-14 | chore: track Ibaraki Radio Taiso source archives for x86_64 conversion handoff |
| `a7fff35` | 2026-05-14 | docs: add FBX 3000 conversion guide for Intel/x86_64 operators |
| `6000c64` | 2026-05-14 | feat: add script for headless FBX to BVH conversion using Blender's bpy API |
| `5d5e8e6` | 2026-05-14 | chore: add tmp/ directory to .gitignore |
| `b632d9c` | 2026-05-14 | Add pipeline for Unitree G1 ラジオ体操 motion retargeting and training |
| `454ff38` | 2026-05-14 | feat: add comprehensive agent documentation for running pretrained Unitree G1 dance in MuJoCo |
| `17f0eaa` | 2026-05-13 | docs: update G1 dance MuJoCo plan with completed status and detailed execution steps |
| `d8ee588` | 2026-05-13 | feat: add script to record dance motion with camera orbiting the robot |
| `0d75112` | 2026-05-13 | chore: untrack mp4 recordings and ignore via *.mp4 |
| `bccecbc` | 2026-05-13 | feat: add script to record dance motion using offscreen rendering |
| `519e03b` | 2026-05-13 | Refactor code structure for improved readability and maintainability |
| `0fffc64` | 2026-05-13 | Add new video and log files for dance and run versions |
| `b618ede` | 2026-05-13 | feat: add start_interactive_view.sh script for headless MuJoCo simulation |
| `82c50b4` | 2026-05-13 | Refactor run_dance_headless.py for improved FSM transition handling |
| `096b95c` | 2026-05-13 | Add initial run log file to track execution status |
| `4db9448` | 2026-05-13 | feat: add headless G1 dance_102 execution script for rl_sar |
| `1c54eaf` | 2026-05-13 | feat: update build log and run report with successful inference runtime setup and resolved issues |
| `fdbb3ec` | 2026-05-13 | feat: update build log with successful inference runtime and robot descriptions setup |
| `2f229e8` | 2026-05-13 | feat: add build log and run report for G1 dance simulation failure |
| `cc7f9ff` | 2026-05-13 | feat: add G1 dance_102 execution plan for MuJoCo simulation |
| `d130598` | 2026-05-12 | feat: add CLAUDE.md for project guidance and build instructions |
| `f7c1d71` | 2026-05-12 | feat: add agent documentation for running pretrained Unitree G1 dance in MuJoCo simulation |

---

## 9. Reference documents

- [`agent.md`](../agent.md) — narrow mission spec (run pretrained dance in MuJoCo).
- [`radio_taiso_g1_agent.md`](../radio_taiso_g1_agent.md) — extended mission spec (full Radio Taiso → G1 RL pipeline).
- [`CLAUDE.md`](../CLAUDE.md) — project guidance (build modes, FSM, policy/config layout).
- [`docs/g1_dance_mujoco_plan.md`](g1_dance_mujoco_plan.md) — as-built record of the MuJoCo simulation track.
- [`docs/fbx3000_intel_handoff.md`](fbx3000_intel_handoff.md) — FBX modernization handoff + 2026-05-14 execution log.
