# PHC Badminton Body-Imitation And Virtual Racket Baseline

This workspace tracks the badminton PHC body-imitation baseline plus the
separate no-physics virtual-racket head experiments. Historical notes below are
kept for provenance; the current status is summarized first.

## Current Status As Of 2026-05-30

Core representation:

- `racket_pose` is a source-side time-varying racket control signal, not a
  world-space racket tip.
- Fixed passive hand-mounted racket attachment was rejected.
- Dynamic reference replay passed.
- Live Goal V2 is the live controller target:
  - target and realized feedback are both in current simulated-root
    heading-local coordinates.
  - Goal V1 remains reference-root-local diagnostic/reference data only.
- The virtual racket is no-physics: no mass, inertia, collision, shuttle, or
  hitting/contact reward.

Frozen body and virtual head contracts:

- Confirmed frozen PHC body checkpoint:
  `humenv/data_preparation/PHC/output/HumanoidIm/phc_comp_3/Humanoid.pth`
- Confirmed body policy contract: original body observation `[934]` -> frozen
  MCP body action `[3]`; saved Hydra config uses `env.num_prim=3`.
- Stage 1A Model B racket head contract: Live Goal V2 `[9]` + realized feedback
  V2 `[6]` -> virtual racket action `[6]`.
- Stage 1B virtual env boundary: body `[3]` + racket `[6]` -> combined action
  `[9]`.
- The older `[4]+[6]=[10]` action shape came from smoke/default configs and is
  not the confirmed pretrained body checkpoint contract.

Validated stages:

- Stage 1A executed the first and only training so far: separate virtual racket
  head supervised oracle-action warm start. Original PHC body weights were not
  loaded for training and were not modified.
- Stage 1B corrected full held-out integration passed in no-physics scope:
  - held-out groups: `241217_2`, `241226_1`
  - clips / frames: `40 / 9833`
  - modes: `body_only`, `virtual_null`, `virtual_goal_only`,
    `virtual_goal_state`, `virtual_oracle`
  - body parity passed with max root-trace diff `0.0`
  - Model B no-physics virtual tracking: tip mean about `0.015254 m`, axis mean
    about `0.963681 deg`
  - oracle no-physics virtual tracking: tip mean about `0.004530 m`, axis mean
    about `0.169190 deg`

Current blocker:

- Stage 1C hand/wrist diagnostic is still provenance-limited.
- Existing `*.kintwin_trace.npz` files reproduce large hand/wrist diagnostics,
  but the initial saved traces lacked full body/ref/root/timing/body-name
  metadata needed for exact official MPJPE/root/heading/body-index validation.
- Do not treat the `~2.20 m` hand/wrist RMSE as confirmed frozen-body hand
  failure until the validity-trace audit resolves it.
- Do not design a hand/body coupling objective or reward based on that number
  yet.

Strict scope:

- No PPO/RL or reward-based tuning has been run for the racket task.
- No PHC body actor fine-tuning has been run.
- No physical racket, shuttle, collision, mass/inertia, or hitting reward has
  been added.
- Results here are not physical racket accuracy and not official PHC rollout
  racket accuracy.

## Current User-Run Commands

Stage 1B validity-trace regeneration plus Stage 1C CPU re-audit:

```bash
cd /train-data-1-hdd/guancheng/badminton_dataset
./phc_baseline/analyze/frozen_body_virtual_racket_stage1b/run_user_validity_trace_regen_and_audit.sh
```

Expected trace counts after a complete run:

- `*.validity_trace.npz`: `200`
- `*.kintwin_trace.npz`: `160`

If the run stops before completion, inspect:

```bash
tail -120 /tmp/stage1b_validity_trace_regen.log
find phc_baseline/reports/racket_calibration/frozen_body_head_integration/full_heldout_eval/children -name '*.validity_trace.npz' -type f | wc -l
find phc_baseline/reports/racket_calibration/frozen_body_head_integration/full_heldout_eval/children -name '*.json' -type f | wc -l
```

CPU-only Stage 1C audit from saved traces:

```bash
cd /train-data-1-hdd/guancheng/badminton_dataset
phc_baseline/envs/phc_isaac/bin/python \
  phc_baseline/analyze/frozen_body_virtual_racket_stage1b/audit_hand_wrist_metric_validity.py
```

Key current reports:

- `phc_baseline/reports/racket_calibration/phc_calibrated_racket_trajectory_report_v2.md`
- `phc_baseline/reports/racket_calibration/frozen_body_head_integration/stage1b_evaluation_report.md`
- `phc_baseline/reports/racket_calibration/frozen_body_head_integration/kintwin_style_tracking_metric_audit.md`
- `phc_baseline/reports/racket_calibration/frozen_body_head_integration/hand_wrist_metric_validity_conclusion.md`
- `phc_baseline/reports/racket_calibration/frozen_body_head_integration/same_run_body_metric_reproduction_report.md`

## Historical Body-Only Baseline Notes

The original body-imitation baseline goal was:

## Goal

Raw badminton SMPL motion -> PHC official motion format -> official PHC
pretrained SMPL humanoid inference -> videos and basic evaluation.

Historical status at that early gate: PHC/Isaac Gym Python-side installation
was complete in the Codex sandbox, but rollout was blocked there because no CUDA
GPU was visible. The user's actual execution environment can see NVIDIA GPUs and
has since run the Stage 1B smoke/full evaluation commands above.

## PHC Source

PHC repo:

`humenv/data_preparation/PHC`

Remote:

`https://github.com/ZhengyiLuo/PHC.git`

Commit SHA:

`34fa3a1c42c519895bc33ae47a10a1ef61a39520`

Pre-existing dirty files in PHC repo:

`download_data.sh`

`scripts/data_process/process_amass_db.py`

This baseline did not edit those files.

## Environment

See `phc_baseline/environment_audit.txt`.

Short version:

- OS: Ubuntu 22.04.1 LTS on Linux.
- Default Python: 3.13.12.
- Installed local PHC env: `phc_baseline/envs/phc_isaac`.
- Installed Python: 3.8.20.
- Installed torch: `1.13.1+cu116`.
- `torch.version.cuda`: `11.6`.
- `torch.cuda.is_available()`: `False`, because the NVIDIA driver/GPU is not usable.
- `nvidia-smi` exists but cannot communicate with the NVIDIA driver.
- `isaacgym` is installed from `IsaacGym_Preview_4_Package.tar.gz`.
- `smpl_sim` is installed from local editable `humenv/data_preparation/SMPLSim`.
- SMPL files are linked into PHC `data/smpl`.
- PHC sample motion and three checkpoints are downloaded.

## Raw Input Inspection

Raw source:

`humenv/data_preparation/AMASS/datasets/NewRacket`

Inspection output:

`phc_baseline/input/smpl_input_inspection.json`

Summary:

- Motion files: 4093 `.npz` sequences.
- Pose key: `poses`.
- Pose shape: `[T, 72]`.
- Translation key: `trans`.
- Translation shape: `[T, 3]`.
- Shape key: `betas`.
- Shape: `[10]`.
- Gender exists.
- FPS exists as `mocap_framerate`; observed value is `30.0`.
- No NaN/Inf detected.
- No SMPL-X / hand dimensions in pose; body is already SMPL 24-joint / 72D axis-angle candidate.

## Adapter Output

Script:

`phc_baseline/prepare_smpl_for_phc.py`

Command executed:

```bash
conda run -n kintwin python phc_baseline/prepare_smpl_for_phc.py --source humenv/data_preparation/AMASS/datasets/NewRacket --output phc_baseline/input/badminton_smpl_for_phc.pkl
```

Output:

`phc_baseline/input/badminton_smpl_for_phc.pkl`

Contains 4093 sequences in PHC converter input dict form:

```python
{
  sequence_name: {
    "pose_aa": np.ndarray[T, 72],
    "trans": np.ndarray[T, 3],
    "beta": np.ndarray[10],
    "gender": "male" | "female" | "neutral",
    "fps": 30.0,
  }
}
```

## PHC Conversion Wrapper

Script prepared:

`phc_baseline/convert_badminton_smpl_phc.py`

Intended command after official sample smoke test passes:

```bash
conda run -n phc_isaac python phc_baseline/convert_badminton_smpl_phc.py --phc-root humenv/data_preparation/PHC --input phc_baseline/input/badminton_smpl_for_phc.pkl --output phc_baseline/converted/badminton_phc_motion.pkl --smpl-data-dir data/smpl
```

Not executed in this run because PHC sample smoke test failed before rollout.

Expected output when unblocked:

`phc_baseline/converted/badminton_phc_motion.pkl`

## Official PHC Sample Smoke Test

Command attempted from `humenv/data_preparation/PHC`:

```bash
conda run -n kintwin python phc/run_hydra.py learning=im_mcp_big learning.params.network.ending_act=False exp_name=phc_comp_kp_2 env.obs_v=7 env=env_im_getup_mcp robot=smpl_humanoid robot.real_weight_porpotion_boxes=False env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl env.models=['output/HumanoidIm/phc_kp_2/Humanoid.pth'] env.num_prim=3 env.num_envs=1 headless=True epoch=-1 test=True
```

Result:

Previously failed immediately with:

```text
ModuleNotFoundError: No module named 'isaacgym'
```

No PHC rollout occurred. No sample video was produced.

After install request, the following were completed:

- Created `phc_baseline/envs/phc_isaac`.
- Installed torch/torchvision/torchaudio CUDA 11.6 wheels.
- Installed local SMPLSim.
- Installed PHC Python requirements using
  `phc_baseline/phc_requirements_no_smplsim.txt`.
- Linked SMPL files into `humenv/data_preparation/PHC/data/smpl`.
- Downloaded `sample_data/amass_isaac_standing_upright_slim.pkl`.
- Downloaded `output/HumanoidIm/phc_3/Humanoid.pth`.
- Downloaded `output/HumanoidIm/phc_comp_3/Humanoid.pth`.
- Downloaded `output/HumanoidIm/phc_kp_2/Humanoid.pth`.
- Installed Isaac Gym Preview 4 into the local env.
- Compiled the Isaac Gym `gymtorch` extension.
- Downgraded NumPy to `1.23.5` for Isaac Gym compatibility.

Latest smoke command:

```bash
cd humenv/data_preparation/PHC
PATH=/train-data-1-hdd/guancheng/badminton_dataset/phc_baseline/envs/phc_isaac/bin:$PATH \
LD_LIBRARY_PATH=/train-data-1-hdd/guancheng/badminton_dataset/phc_baseline/envs/phc_isaac/lib:$LD_LIBRARY_PATH \
TORCH_EXTENSIONS_DIR=/train-data-1-hdd/guancheng/badminton_dataset/phc_baseline/torch_extensions \
MPLCONFIGDIR=/tmp \
/train-data-1-hdd/guancheng/badminton_dataset/phc_baseline/envs/phc_isaac/bin/python phc/run_hydra.py learning=im_mcp_big exp_name=phc_comp_3 env=env_im_getup_mcp robot=smpl_humanoid env.zero_out_far=False robot.real_weight_porpotion_boxes=False env.num_prim=3 env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl env.models=['output/HumanoidIm/phc_3/Humanoid.pth'] env.num_envs=1 headless=True epoch=-1 test=True
```

Latest result:

```text
Found checkpoint
Started to play
RuntimeError: No CUDA GPUs are available
```

## Planned Commands Once Unblocked

Official-compatible sample smoke test:

```bash
cd humenv/data_preparation/PHC
conda run -n phc_isaac python phc/run_hydra.py learning=im_mcp_big learning.params.network.ending_act=False exp_name=phc_comp_kp_2 env.obs_v=7 env=env_im_getup_mcp robot=smpl_humanoid robot.real_weight_porpotion_boxes=False env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl env.models=['output/HumanoidIm/phc_kp_2/Humanoid.pth'] env.num_prim=3 env.num_envs=1 headless=False epoch=-1 test=True
```

Badminton PHC conversion:

```bash
conda run -n phc_isaac python phc_baseline/convert_badminton_smpl_phc.py --phc-root humenv/data_preparation/PHC --input phc_baseline/input/badminton_smpl_for_phc.pkl --output phc_baseline/converted/badminton_phc_motion.pkl --smpl-data-dir data/smpl
```

Badminton body imitation inference:

```bash
cd humenv/data_preparation/PHC
conda run -n phc_isaac python phc/run_hydra.py learning=im_mcp_big exp_name=phc_comp_3 env=env_im_getup_mcp robot=smpl_humanoid env.zero_out_far=False robot.real_weight_porpotion_boxes=False env.num_prim=3 env.motion_file=/train-data-1-hdd/guancheng/badminton_dataset/phc_baseline/converted/badminton_phc_motion.pkl env.models=['output/HumanoidIm/phc_3/Humanoid.pth'] env.num_envs=1 headless=False epoch=-1 test=True
```

Checkpoint path must be re-validated after `bash download_data.sh`; no
checkpoint exists locally right now.

## Outputs Created

Created:

- `phc_baseline/environment_audit.txt`
- `phc_baseline/input/smpl_input_inspection.json`
- `phc_baseline/input/badminton_smpl_for_phc.pkl`
- `phc_baseline/prepare_smpl_for_phc.py`
- `phc_baseline/convert_badminton_smpl_phc.py`
- `phc_baseline/reports/coordinate_validation.md`
- `phc_baseline/reports/phc_inference_command.txt`
- `phc_baseline/reports/phc_body_imitation_observation.md`
- `phc_baseline/reports/phc_body_imitation_metrics.json`

Not created because execution is blocked:

- `phc_baseline/converted/badminton_phc_motion.pkl`
- `phc_baseline/videos/phc_sample_smoke_test.mp4`
- `phc_baseline/videos/badminton_reference_phc_coordinate_check.mp4`
- `phc_baseline/videos/badminton_phc_imitation.mp4`

## Next Manual Inputs Needed

1. Fix NVIDIA driver so `nvidia-smi` works.
2. Rerun official sample smoke test before converting badminton motion.
