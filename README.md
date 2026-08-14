# R16-P18 Outcome-Boundary Adaptive Perception–Action Resolution

> **ManiSkill Step4 / Stage-2.5 final (2026-08-14): `REVISE_STOPPING_CONFOUND`.**
> The complete privileged-oracle protocol was executed despite downstream gate
> failures, as explicitly requested. StackCube passed its repaired baseline gate,
> but privileged first-success termination improved terminal success by 16.33pp
> (paired-bootstrap 95% CI [12.00pp, 21.00pp], 3/3 seeds), so the preregistered
> stopping-confound status takes precedence. The action gate and joint gate also
> failed: cross-seed joint coupling was 0/64, recall improvement was 0pp, and
> outcome-regret reduction was 3.12%. See
> [`STAGE25_FINAL_REPORT.md`](experiments/maniskill_stage25_repair_oracle_v1/docs/STAGE25_FINAL_REPORT.md)
> and the [sealed v26 artifacts](experiments/maniskill_stage25_repair_oracle_v1/artifacts/formal-run/r16p18-stage25-oracle-20260814-v26/README.md).
> This is neither validation nor refutation of the R16-P18 idea, and no learned
> predictor, OOD, Stage-3, Diffusion Policy, DINO-WM, or pi0.5 work was started.

> **ManiSkill RGB-ACT Stage-2 final (2026-08-14): `NO_GO_BASELINE_GATE`.**
> The audited 1,200-episode baseline passed only `StackCube-v1`; both
> `PullCubeTool-v1` and `PushT-v1` were below the positive-task floor, and the
> `PushCube-v1` negative control was below its 70% floor. The preregistered stop
> rule therefore prohibited the state-bank/oracle probe and Stage-3. See
> [`docs/MANISKILL_STAGE2_FINAL_REPORT.md`](docs/MANISKILL_STAGE2_FINAL_REPORT.md).
> This is not a validation or refutation of the adaptive mechanism.

This repository is the complete archived implementation and evidence package for the
**LIBERO Stage-1 Small-BC baseline gate** run on 2026-08-12. It contains the frozen
BoundaryBC-S source, PAI launch contracts and audit trail, all 36 checkpoints, raw
training/evaluation records, W&B run files, smoke-test evidence, and the experiment report.

> **Final decision: `NO_GO_BASELINE_GATE`.** This is a baseline health-gate result,
> not a result for the R16-P18 adaptive mechanism. The joint visual/action selector,
> effect predictor, state bank, and matched adaptive arms were intentionally not implemented
> because the preregistered gate failed.

## Baseline-gate result

| LIBERO task | Aggregate success | 95% paired bootstrap CI | Required range | Gate |
|---|---:|---:|---:|---|
| `push_the_plate_to_the_front_of_the_stove` | 80.0% | [52.0%, 98.0%] | 40%–90% | PASS |
| `put_the_wine_bottle_on_the_rack` | 98.0% | [94.0%, 100.0%] | 25%–80% | **FAIL (ceiling)** |
| `put_the_bowl_on_the_plate` | 88.7% | [78.0%, 97.3%] | 80%–100% | PASS |

The formal PAI job `dlcnouq6igkhfyub` completed successfully: 3 tasks × 3 model
seeds × 3000 optimizer steps and 50 fixed closed-loop episodes per task/seed,
for 9 models and 450 episodes. Bottle placement exceeded the preregistered 80%
upper bound, so all three tasks did not jointly pass.

Official LIBERO provides only 50 successful demonstrations for each exact task and no
original episode-seed field. This pilot used deterministic SHA256 identities and a 40/5/5
split. That preregistered protocol deviation independently prevents a Stage-1 GO.

## Repository map

- `boundarybc/`: BoundaryBC-S model, data, checkpointing, training, rollout, reporting,
  provenance, and smoke-test implementation.
- `configs/r16_p18_libero_stage1.yaml`: frozen protocol and thresholds.
- `scripts/pai_r16_p18_*.sh`: owner-safe PAI runtime launchers.
- `tests/test_stage1_contract.py`: static model/config/checkpoint/device-queue contract tests.
- `artifacts/dev-smoke/`: both development smoke records, including the final A800 smoke.
- `artifacts/formal-run/`: raw 3000-step training logs, all 450 episode records,
  summaries, W&B files, manifest, and gate reports.
- `artifacts/checkpoints/`: all 36 PyTorch checkpoint/final files plus completion markers
  and training summaries; `.pt` files are stored with Git LFS.
- `artifacts/pai-registry/`: exact final PAI registry files and complete 001–003 job evidence.
- `docs/EXPERIMENT_REPORT.md`: full experiment interpretation and results.
- `docs/ARTIFACTS.md`: inclusion/exclusion scope and artifact guide.
- `provenance/`: original source/registry commit patches and release checksums.

## Clone and verify

Git LFS is required for the model files:

```bash
git lfs install
git clone git@github.com:mikasaTu/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut.git
cd R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut
git lfs pull
```

With the pinned Python environment installed, run the read-only result recomputation and
unit tests:

```bash
python scripts/verify_archived_results.py
pytest -q tests/test_stage1_contract.py
sha256sum -c provenance/SHA256SUMS
```

The archived runtime was Python 3.11.11, PyTorch 2.5.1+cu124,
torchvision 0.20.1+cu124, CUDA 12.4, cuDNN 90100, MuJoCo 3.6.0,
robosuite 1.4.0, and NumPy 1.26.4. Full provenance is in
`artifacts/formal-run/r16-p18-libero-stage1-bc-gate-20260812-003/run_manifest.json`.

## Scope boundary

No claim is made that the R16-P18 idea is validated, accepted, or refuted. No Diffusion
Policy, DINO-WM, world model, or π0.5 experiment was started. A future attempt requires a
fresh preregistration with a harder constrained-placement task and a resolved 200-demo
episode-seed protocol; thresholds must not be widened retroactively.

---

## Vendored LIBERO base

The implementation is based on the Hugging Face LIBERO fork. Its original README follows.

<div align="center">
<img src="https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/images/libero_logo.png" width="360">


<p align="center">
<a href="https://github.com/Lifelong-Robot-Learning/LIBERO/actions">
<img alt="Tests Passing" src="https://github.com/anuraghazra/github-readme-stats/workflows/Test/badge.svg" />
</a>
<a href="https://github.com/Lifelong-Robot-Learning/LIBERO/graphs/contributors">
<img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/Lifelong-Robot-Learning/LIBERO" />
</a>
<a href="https://github.com/Lifelong-Robot-Learning/LIBERO/issues">
<img alt="Issues" src="https://img.shields.io/github/issues/Lifelong-Robot-Learning/LIBERO?color=0088ff" />

## **Benchmarking Knowledge Transfer for Lifelong Robot Learning**

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, Peter Stone

[[Website]](https://libero-project.github.io)
[[Paper]](https://arxiv.org/pdf/2306.03310.pdf)
[[Docs]](https://lifelong-robot-learning.github.io/LIBERO/)
______________________________________________________________________
![pull_figure](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/images//fig1.png)
</div>

**LIBERO** is designed for studying knowledge transfer in multitask and lifelong robot learning problems. Successfully resolving these problems require both declarative knowledge about objects/spatial relationships and procedural knowledge about motion/behaviors. 
This repository started as a fork of the official LIBERO benchmark
. It is now maintained by the Hugging Face team, with modifications for compatibility with LeRobot
, simplified installation, and large-scale robotics experiments.
**LIBERO** provides:
- a procedural generation pipeline that could in principle generate an infinite number of manipulation tasks.
- 130 tasks grouped into four task suites: **LIBERO-Spatial**, **LIBERO-Object**, **LIBERO-Goal**, and **LIBERO-100**. The first three task suites have controlled distribution shifts, meaning that they require the transfer of a specific type of knowledge. In contrast, **LIBERO-100** consists of 100 manipulation tasks that require the transfer of entangled knowledge. **LIBERO-100** is further splitted into **LIBERO-90** for pretraining a policy and **LIBERO-10** for testing the agent's downstream lifelong learning performance.
- five research topics.
- three visuomotor policy network architectures.
- three lifelong learning algorithms with the sequential finetuning and multitask learning baselines.
---


# Contents

- [Installation](#Installation)
- [Datasets](#Dataset)
- [Getting Started](#Getting-Started)
  - [Task](#Task)
  - [Training](#Training)
  - [Evaluation](#Evaluation)
- [Citation](#Citation)
- [License](#License)


# Installtion
Please run the following commands in the given order to install the dependency for **LIBERO**.
```
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

Then install the `libero` package:
```
pip install -e .
```

# Datasets
We provide high-quality human teleoperation demonstrations for the four task suites in **LIBERO**. To download the demonstration dataset, run:
```python
python benchmark_scripts/download_libero_datasets.py
```
By default, the dataset will be stored under the ```LIBERO``` folder and all four datasets will be downloaded. To download a specific dataset, use
```python
python benchmark_scripts/download_libero_datasets.py --datasets DATASET
```
where ```DATASET``` is chosen from `[libero_spatial, libero_object, libero_100, libero_goal`.

**NEW!!!**

Alternatively, you can download the dataset from HuggingFace by using:
```python
python benchmark_scripts/download_libero_datasets.py --use-huggingface
```

This option can also be combined with the specific dataset selection:
```python
python benchmark_scripts/download_libero_datasets.py --datasets DATASET --use-huggingface
```

The datasets hosted on HuggingFace are available at [here](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets).

## Assets

**IMPORTANT: Asset Loading from HuggingFace Hub**

The simulation assets (3D models, textures, scene files, etc.) are now automatically loaded from HuggingFace Hub instead of being bundled with the package. When you first run LIBERO, the assets will be automatically downloaded from the Hub repository [yifengzhu-hf/LIBERO-assets](https://huggingface.co/yifengzhu-hf/LIBERO-assets) and cached locally.

This change:
- Reduces the size of the installed package
- Ensures you always have the latest assets
- Allows for easy asset versioning and updates

The assets will be cached at `~/.cache/libero/assets/` and will only be downloaded once. If you have local assets installed from a previous version, those will be used instead.


# Getting Started

For a detailed walk-through, please either refer to the documentation or the notebook examples provided under the `notebooks` folder. In the following, we provide example scripts for retrieving a task, training and evaluation.

## Task

The following is a minimal example of retrieving a specific task from a specific task suite.
```python
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_10" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()

# retrieve a specific task
task_id = 0
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
      f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

# step over the environment
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
env.reset()
init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
init_state_id = 0
env.set_init_state(init_states[init_state_id])

dummy_action = [0.] * 7
for step in range(10):
    obs, reward, done, info = env.step(dummy_action)
env.close()
```
Currently, we only support sparse reward function (i.e., the agent receives `+1` when the task is finished). As sparse-reward RL is extremely hard to learn, currently we mainly focus on lifelong imitation learning.

## Training
To start a lifelong learning experiment, please choose:
- `BENCHMARK` from `[LIBERO_SPATIAL, LIBERO_OBJECT, LIBERO_GOAL, LIBERO_90, LIBERO_10]`
- `POLICY` from `[bc_rnn_policy, bc_transformer_policy, bc_vilt_policy]`
- `ALGO` from `[base, er, ewc, packnet, multitask]`

then run the following:

```shell
export CUDA_VISIBLE_DEVICES=GPU_ID && \
export MUJOCO_EGL_DEVICE_ID=GPU_ID && \
python libero/lifelong/main.py seed=SEED \
                               benchmark_name=BENCHMARK \
                               policy=POLICY \
                               lifelong=ALGO
```
Please see the documentation for the details of reproducing the study results.

## Evaluation

By default the policies will be evaluated on the fly during training. If you have limited computing resource of GPUs, we offer an evaluation script for you to evaluate models separately.

```shell
python libero/lifelong/evaluate.py --benchmark BENCHMARK_NAME \
                                   --task_id TASK_ID \ 
                                   --algo ALGO_NAME \
                                   --policy POLICY_NAME \
                                   --seed SEED \
                                   --ep EPOCH \
                                   --load_task LOAD_TASK \
                                   --device_id CUDA_ID
```

# Citation
If you find **LIBERO** to be useful in your own research, please consider citing our paper:

```bibtex
@article{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal={arXiv preprint arXiv:2306.03310},
  year={2023}
}
```

# License
| Component        | License                                                                                                                             |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Codebase         | [MIT License](LICENSE)                                                                                                                      |
| Datasets         | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode)                 |
