# FBX 3000 conversion — handoff to an Intel/x86_64 operator

## Why this document exists

The Radio Taiso source motion capture from Ibaraki University (http://fondant.cis.ibaraki.ac.jp/MoCap/0310.htm) ships as **FBX file-format version 3000**. This is the AutoDesk binary FBX layout from the 2003–2005 era, predating FBX 6.0 (5800), FBX 2011 (7100), FBX 2013 (7400), and FBX 2020 (7700).

On the aarch64 DGX Spark this is a hard blocker:

| Tool | Status on aarch64 | Reason |
|---|---|---|
| Blender 4.0.2 / 4.1 | ❌ refuses FBX 3000 | "FBX version (3000) is unsupported, must be 7100 or later" |
| Blender 2.79 | ❌ minimum supported is FBX 5800 | Still above 3000 |
| AutoDesk FBX Python SDK | ❌ no aarch64 build exists | AutoDesk ships only x86_64 binaries, closed-source — recompile impossible |
| pyassimp 5.2.5 | ❌ drops animation tracks for pre-7100 FBX | Assimp implementation gap |
| AutoDesk FBX Converter 2013 (the legacy tool that can read 3000) | ❌ Windows x86 / Wine only | Pre-dates Linux ports |

The realistic conversion paths all need x86 or x86_64 hardware. This document describes what to do, what to ship back, and how to verify the result.

## What you need to convert

Five Ibaraki MoCap takes of Radio Taiso Daiichi (ラジオ体操第一), one FBX file per take, ~16 MB compressed. The zips are committed in this repository so an Intel/x86_64 collaborator can `git pull` and convert them in place:

```
data/ibaraki_radio_taiso/take11.zip → take11.fbx (FBX 3000, ~36 MB)
data/ibaraki_radio_taiso/take12.zip → take12.fbx
data/ibaraki_radio_taiso/take13.zip → take13.fbx
data/ibaraki_radio_taiso/take14.zip → take14.fbx
data/ibaraki_radio_taiso/take15.zip → take15.fbx
```

These are tracked despite the repo-wide `*.zip` ignore via an explicit `!data/ibaraki_radio_taiso/*.zip` override in [.gitignore](../.gitignore).

Each `.fbx` is one performer doing the full ~3-minute routine, captured on a Motion Star magnetic mocap rig at 30 fps. Skeleton names and root joint are Japanese-labeled; the capture session notes (handwritten PDF) confirm 15 sensors mapped to feet, shins, thighs, pelvis, upper/lower arms, hands, chest, head.

## What to ship back

Either of these formats is accepted on the aarch64 side. Pick whichever your toolchain produces most reliably.

**Format A — modernized FBX (preferred)**
- FBX binary version **7400 or 7700** (FBX 2013+ / FBX 2020+)
- Animation curves preserved, skeleton hierarchy preserved, original joint names preserved
- One file per take, named `take{N}_fbx7400.fbx` (or `_fbx7700`)

**Format B — BVH (fallback)**
- Standard BVH text format
- HIERARCHY + MOTION sections both populated
- All frames included (do not truncate)
- Frame Time matches source (1/30 = 0.033333 s)
- One file per take, named `take{N}.bvh`

Ship whichever you produce back via the same git repository — `git add data/ibaraki_radio_taiso/converted/take{N}_fbx7400.fbx` (or `.bvh`), commit, push — or hand them off via any file transfer the operator prefers. On the aarch64 side both formats feed cleanly into GMR's `bvh_to_robot.py` (for BVH) or `fbx_importer.py` (for FBX ≥ 7100).

## Three conversion paths in order of expected reliability

### Path A — AutoDesk FBX Converter 2013 (Windows or Wine)

This is the legacy converter that AutoDesk shipped specifically to upgrade old FBX files. It's the most likely to handle FBX 3000 correctly because it's literally the tool AutoDesk wrote for that purpose.

1. Download "AutoDesk FBX Converter 2013.3" — Windows x86 installer, available from AutoDesk's archive page (search "FBX Converter 2013.3 download"). Free, no sign-in required for legacy versions.
2. Install on a Windows machine (or via Wine on Linux x86_64).
3. Launch GUI → "Add" each `take{N}.fbx` → "Destination format: FBX 2013 (binary)" or "FBX 2020 (binary)" → "Convert".
4. Output goes to whatever folder you point at; default is alongside the source. Files will be ~30–40 MB each.
5. **Critical**: keep "Embed media" off and "Animation only" unchecked — we want the full skeleton, not just the curves.

**Expected wall-clock**: 5–10 minutes for all five takes.

### Path B — AutoDesk FBX Python SDK on x86_64 Linux

If you have an Intel/AMD Linux box, this is scriptable and reproducible.

1. Download "FBX Python SDK 2020.3.7 for Linux" from AutoDesk Developer Network (x86_64 only). Install per their README — typically `pip install` the wheel into a Python 3.10 venv.
2. Convert with a short script along the lines of:
   ```python
   import fbx
   mgr = fbx.FbxManager.Create()
   ios = fbx.FbxIOSettings.Create(mgr, "IOSRoot")
   mgr.SetIOSettings(ios)
   importer = fbx.FbxImporter.Create(mgr, "")
   if not importer.Initialize("take11.fbx", -1, mgr.GetIOSettings()):
       raise RuntimeError(importer.GetStatus().GetErrorString())
   scene = fbx.FbxScene.Create(mgr, "scene")
   importer.Import(scene)
   importer.Destroy()
   exporter = fbx.FbxExporter.Create(mgr, "")
   # File-format ID 0 is the latest binary FBX (typically 7700 on SDK 2020.3.7)
   exporter.Initialize("take11_fbx7700.fbx", 0, mgr.GetIOSettings())
   exporter.Export(scene)
   exporter.Destroy()
   mgr.Destroy()
   ```
3. Loop over all five takes.

**Expected wall-clock**: 30 minutes to set up + 2 minutes to convert.

**Risk**: AutoDesk's Python SDK 2020 may still refuse to *read* FBX 3000 (the floor for the read-side is sometimes higher than what tooling docs claim). If `importer.Initialize` returns false with a version-error string, fall back to Path A.

### Path C — Blender 2.79 on x86_64

Last resort if Paths A and B are unavailable. Blender 2.79's FBX import floor is 5800, **not 3000** — so this only works if AutoDesk FBX Converter is unavailable AND you first round-trip the file through *some* FBX 5800+ intermediary. Unlikely to help in practice; included for completeness.

## Verification recipes (after conversion, before shipping back)

These are quick sanity checks the operator should run before declaring the conversion done. All can be done with command-line tools on Linux or `Get-Content` / Notepad++ on Windows.

**For a modernized FBX (Format A):**
```bash
# Inspect the first 30 bytes — should read "Kaydara FBX Binary  " followed by version
head -c 30 take11_fbx7400.fbx | hexdump -C
# Expect the uint32 at offset 23 to be 7400 (0x1CE8) or 7700 (0x1E14) in little-endian
```

A modernized FBX 7400 file's header looks like:
```
00000000  4b 61 79 64 61 72 61 20  46 42 58 20 42 69 6e 61  |Kaydara FBX Bina|
00000010  72 79 20 20 00 1a 00 e8  1c 00 00                 |ry  ...è...|
```
The `e8 1c 00 00` at offset 23 is 0x1CE8 = 7400. For FBX 7700 you'd see `14 1e 00 00` (= 0x1E14).

**For a BVH (Format B):**
```bash
# Should report two well-known sections and a sensible frame count
grep -c '^HIERARCHY\|^MOTION' take11.bvh   # → 2
grep '^Frames:' take11.bvh                  # → Frames: <a number near 5000–7000>
grep '^Frame Time:' take11.bvh              # → Frame Time: 0.033333  (or similar)

# Sanity-check the joint count
grep -c '^[[:space:]]*JOINT' take11.bvh     # → expect ~14 joints (15 sensors minus root)

# Spot-check that the motion section actually has float rows, not zeros
tail -5 take11.bvh
```

If the file passes these checks for at least one take, it'll feed GMR's pipeline on the aarch64 side. If frame count is 1 or Frame Time is 1.0, the exporter probably stripped the animation — try a different export option.

## Provenance of the source files

- Capture date: 2009-03-10 (recorded in Japanese on the accompanying handwritten PDF: `~/datasets/radio_taiso/reports/ibaraki_20090310_notes.pdf`)
- Performer / capture: Ibaraki University Motion Capture Database, take ratings recorded in the same PDF (take11 = OK, take15 = 96点 ○ "good")
- Source page: http://fondant.cis.ibaraki.ac.jp/MoCap/0310.htm
- Rig: Ascension Motion Star magnetic mocap, 15 sensors at 30 fps, total ~6000 frames per take
- License / usage: Ibaraki's archive is publicly downloadable; treat as research-use per their page (no commercial redistribution implied)

## Why these files are FBX 3000 in the first place

The Ibaraki capture session in 2009 used Ascension's Motion Star + a commercial mocap tool of that era to export FBX. The FBX 3000 layout was the SDK's default binary version at the time. AutoDesk acquired Kaydara (the FBX format's creator) in 2006 and rebased the SDK on a new format generation starting at version 6000–6100 (FBX 5800 header), so files exported by pre-AutoDesk tooling — exactly this dataset's vintage — sit one major-generation behind everything currently maintained.

## Alternative: a VPM source bypasses this blocker entirely

Ibaraki's archive page also lists a `.vpm` (Ascension Motion Star text dump) for at least take11 at `~/datasets/radio_taiso/source/ibaraki_vpm/take11_vpm.zip`. That format is ASCII per-sensor (15 sensors × 9 channels × ~6322 frames), trivially parseable on aarch64, and avoids the FBX SDK problem altogether — at the cost of needing a custom IK/retarget config to map raw sensor poses to a humanoid skeleton.

The Intel-handoff path in this document remains the right choice if you want **animated joint hierarchies straight from the original FBX**. If you'd prefer to skip the round-trip and work from VPM directly, no Intel hardware is needed; just write the parser.

## What happens after you ship the converted files back

1. Drop the `take{N}_fbx7400.fbx` or `take{N}.bvh` files into `~/datasets/radio_taiso/source/ibaraki_modernized/` on the aarch64 box.
2. GMR's `third_party/poselib/fbx_importer.py` (for FBX) or `scripts/bvh_to_robot.py --format lafan1` (for BVH) takes it from there.
3. Stage 2+ of `~/.claude/plans/read-agent-md-and-plan-sharded-platypus.md` continues: retarget → preview MP4 → CSV → NPZ → PPO training → MuJoCo play → deploy prep.

You don't need to touch any of that. Just ship the modernized files and we'll pick them up.

## Conversion log — 2026-05-14

Executed on an Intel Xeon Platinum 8370C Azure VM (Linux x86_64, Ubuntu 22.04, Python 3.10). Used an in-tree C++ tool against the AutoDesk FBX SDK 2020.3.9 — the closest practical equivalent of Path B since the user staged the C++ SDK (not the Python SDK) at `tmp/fbx202039_fbxsdk_gcc_linux.tar.gz`.

Reproducer:
1. `tar -xzf tmp/fbx202039_fbxsdk_gcc_linux.tar.gz -C tmp/` then `./tmp/fbx202039_fbxsdk_linux tmp/fbxsdk` (the installer is interactive — accept EULA).
2. `sudo apt-get install -y libxml2-dev` (the SDK's lone external build dep).
3. `make -C scripts/fbx_modernize` produces `scripts/fbx_modernize/fbx_modernize` statically linked against `libfbxsdk.a`.
4. For each take: unzip → run `fbx_modernize take{N}.fbx converted/take{N}_fbx7700.fbx`.

The SDK 2020.3.9 importer reads FBX 3000 cleanly — it reports the source as "FBX version 5.0.0" (its internal feature-version mapping) and exports as FBX 7700. **The Path-B risk flagged earlier (importer refusing FBX 3000) did not materialize.**

Output verification (FBX version field at byte offset 23, little-endian uint32):

| File | Header version | Size |
|---|---|---|
| `converted/take11_fbx7700.fbx` | 7700 | 12.5 MB |
| `converted/take12_fbx7700.fbx` | 7700 | 12.3 MB |
| `converted/take13_fbx7700.fbx` | 7700 | 12.3 MB |
| `converted/take14_fbx7700.fbx` | 7700 | 12.4 MB |
| `converted/take15_fbx7700.fbx` | 7700 | 12.4 MB |

A round-trip on take11 (re-import + re-export of the converted file) produced a byte-identical 12,457,488-byte output, confirming structural validity beyond the header check.

Total converted payload: ~59 MB, committed to the repo. Source see [scripts/fbx_modernize/](../scripts/fbx_modernize/).

---

## Downstream — what happens to the converted FBX files

Once converted, the FBX 7700 files are consumed by the retargeting pipeline documented in [`radio_taiso_retargeting.md`](radio_taiso_retargeting.md):

1. `scripts/fbx_motionstar_to_npz.py` — Blender headless extractor → per-frame 15-sensor NPZ
2. `scripts/motionstar_retarget.py` — NPZ → G1 qpos via GMR's custom `motionstar` source
3. `scripts/qpos_npz_to_csv.py` + `scripts/evaluate_retarget.py` — CSV emit + metric scoring

The canonical retargeted motion lives at `~/datasets/radio_taiso/gmr/radio_taiso_g1.npz`. See the pipeline doc for the full as-built description.
