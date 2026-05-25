# PHC Badminton Body-Imitation Baseline

This workspace is an isolated PHC body-imitation baseline. It does not modify
KinTwin training code, HumEnv HDF5 conversion, V10-V12 profiles, reward logic,
curricula, or existing checkpoints.

## Goal

Raw badminton SMPL motion -> PHC official motion format -> official PHC
pretrained SMPL humanoid inference -> videos and basic evaluation.

Current status: PHC/Isaac Gym Python-side installation is complete, but PHC is
still blocked before rollout because no CUDA GPU is visible from this session.

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
