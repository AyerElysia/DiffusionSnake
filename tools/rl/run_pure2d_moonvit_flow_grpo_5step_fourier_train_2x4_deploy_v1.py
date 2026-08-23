#!/usr/bin/env python3
"""Fail-closed launcher for five-action RL with 2x4 deployment evaluation."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path('/home/medteam/Zhrch/DiffusionSnake-12-30')
PYTHON = Path('/home/medteam/miniconda3/envs/sam1_lgz/bin/python')
TRAINER = REPO / (
    'tools/rl/'
    'grpo_train_pure2d_moonvit_5step_fourier_train_2x4_deploy_v1.py'
)
CONFIG = REPO / (
    'configs/rl/'
    'pure2d_moonvit_flow_grpo_5step_fourier_train_2x4_deploy_v1.yaml'
)
SUPERVISED_CONFIG = REPO / (
    'configs/volmem/depth_sweep/'
    'pure2d_mainline_l6_f256_routeb_v410_moonvit_cached_'
    'flowtune60k_from40000.yaml'
)
SOURCE = REPO / (
    'data/outputs/depth_sweep/'
    'pure2d_mainline_l6_f256_routeb_v410_moonvit_cached_'
    'flowtune60k_from40000_v1/checkpoints/step_19000.pt'
)
TUNE_MANIFEST = REPO / (
    'configs/rl/manifests/volmem_fourier_validation37_20260820.csv'
)
PREFLIGHT = REPO / (
    'data/outputs/rl/'
    'pure2d_moonvit_flowonly_grpo_5step_fourier_train_2x4_deploy_'
    'from19000_preflight_v1'
)
FORMAL = REPO / (
    'data/outputs/rl/'
    'pure2d_moonvit_flowonly_grpo_5step_fourier_train_2x4_deploy_'
    'from19000_v1'
)

EXPECTED_TRAINER_SHA256 = (
    '8772fa80aa12b0e6e8757e762b558c63c310c36ac287e444abd232467b46122f'
)
EXPECTED_CONFIG_SHA256 = (
    'e358ca41f51834269b1b37faf5d9a6a5f066a8b7402156b8c9f3955990ff8b63'
)
EXPECTED_SUPERVISED_CONFIG_SHA256 = (
    'f8095f2754e2c3f0b94c4d7fdedc6b880bca5de40231f2603561ed557db3caaf'
)
EXPECTED_SOURCE_SHA256 = (
    'a337ba1566fe423c10a82dc4c08f8d6936ce8fc49ff1d61c8f735435854a337f'
)
EXPECTED_TUNE_MANIFEST_SHA256 = (
    'f2f11b5b430135f7a2e80bc40381ff3cd57b75e6c07f8da6e155476fbfe2707c'
)
EXPECTED_SOURCE_STEP = 19_000
EXPECTED_MODEL_PARAMETERS = 14_373_444
EXPECTED_FLOW_PARAMETERS = 11_127_108
EXPECTED_CONTEXT_PARAMETERS = 3_246_336
FLOW_PREFIXES = ('net.gcn.', 'gcn.')


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temporary, path)


def verify_inputs() -> dict:
    expected = {
        TRAINER: EXPECTED_TRAINER_SHA256,
        CONFIG: EXPECTED_CONFIG_SHA256,
        SUPERVISED_CONFIG: EXPECTED_SUPERVISED_CONFIG_SHA256,
        SOURCE: EXPECTED_SOURCE_SHA256,
        TUNE_MANIFEST: EXPECTED_TUNE_MANIFEST_SHA256,
    }
    observed = {}
    for path, expected_digest in expected.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_digest = sha256(path)
        observed[str(path)] = observed_digest
        if observed_digest != expected_digest:
            raise RuntimeError(
                f'SHA256 drift for {path}: {observed_digest} != {expected_digest}'
            )
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    observed[str(PYTHON)] = sha256(PYTHON)
    return observed


def gpu_sample(gpu: int) -> dict:
    raw = subprocess.check_output([
        'nvidia-smi', f'--id={gpu}',
        '--query-gpu=index,memory.used,utilization.gpu,uuid',
        '--format=csv,noheader,nounits',
    ], text=True).strip()
    parts = [item.strip() for item in raw.split(',')]
    if len(parts) != 4 or int(parts[0]) != gpu:
        raise RuntimeError(f'unexpected nvidia-smi row: {raw!r}')
    uuid = parts[3]
    apps_raw = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits',
    ], text=True).strip()
    compute_pids = []
    for line in apps_raw.splitlines():
        app_parts = [item.strip() for item in line.split(',')]
        if len(app_parts) == 2 and app_parts[0] == uuid:
            compute_pids.append(int(app_parts[1]))
    return {
        'gpu': gpu,
        'memory_mib': int(parts[1]),
        'util_percent': int(parts[2]),
        'uuid': uuid,
        'compute_pids': compute_pids,
        'time': now(),
    }


def require_idle_gpu(gpu: int) -> list[dict]:
    checks = [gpu_sample(gpu)]
    time.sleep(15)
    checks.append(gpu_sample(gpu))
    for sample in checks:
        if (
            sample['memory_mib'] > 20
            or sample['util_percent'] != 0
            or sample['compute_pids']
        ):
            raise RuntimeError(f'GPU{gpu} is not strictly idle: {checks}')
    return checks


def state_dict(payload):
    if not isinstance(payload, dict):
        return payload
    for key in ('state_dict', 'model', 'net', 'network'):
        if isinstance(payload.get(key), dict):
            return payload[key]
    return payload


def audit_preflight(output: Path, launch_log: Path) -> dict:
    import torch

    latest = output / 'checkpoints/latest.pt'
    log_path = output / 'posttrain_rl_fourier_outer_action/logs.jsonl'
    hparams_path = output / 'posttrain_rl_fourier_outer_action/v5_hparams.json'
    baseline_path = (
        output / 'posttrain_rl_fourier_outer_action/eval_baseline_step0.json'
    )
    required = (latest, log_path, hparams_path, baseline_path, launch_log)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f'preflight artifacts incomplete: {missing}')

    console = launch_log.read_text(encoding='utf-8', errors='replace')
    if (
        'five-stage AB2 training rollout alignment PASS max_abs=' not in console
        or 'production 2x4 AB2 deployment alignment PASS max_abs=' not in console
        or 'stage_s=[0.0, 0.6667]' not in console
    ):
        raise RuntimeError(
            'five-stage training and production 2x4 alignment evidence is missing'
        )

    rows = [
        json.loads(line) for line in log_path.read_text().splitlines()
        if line.strip()
    ]
    if [int(row.get('step', -1)) for row in rows] != [1, 2]:
        raise RuntimeError('preflight must contain exactly effective steps 1 and 2')
    finite_keys = (
        'reward_mean', 'reward_std_mean', 'policy_loss', 'grad_norm',
        'ratio_mean', 'approx_kl',
    )
    for row in rows:
        for key in finite_keys:
            if not math.isfinite(float(row[key])):
                raise RuntimeError(
                    f'non-finite metric step={row["step"]} {key}={row[key]}'
                )
        if row.get('action_policy') != 'geom':
            raise RuntimeError('wrong action policy')
        if int(row.get('geom_lowfreq_modes', -1)) != 8:
            raise RuntimeError('wrong Fourier mode count')
        if [round(float(x), 4) for x in row['geom_sigma_px']] != [
            0.8, 0.7, 0.6, 0.5, 0.4
        ]:
            raise RuntimeError('wrong Fourier sigma schedule')
        if abs(float(row.get('outer_log_count_mean', -1.0)) - 5.0) > 1e-6:
            raise RuntimeError('each rollout must log exactly five outer actions')

    hparams = json.loads(hparams_path.read_text())
    fourier = hparams.get('fourier_configuration', {})
    reward_weights = hparams.get('reward_weights', {})
    reward_mode = hparams.get('reward_mode', {})
    deployment = hparams.get('deployment_schedule', {})
    observed_weights = [
        round(float(reward_weights.get(key, -1.0)), 4)
        for key in ('region', 'dice', 'iou', 'dist')
    ]
    if (
        fourier.get('profile')
        != 'legacy5_m8_sigma080_070_060_050_040_overlap_composite'
        or fourier.get('eval_dataset') != 'VolMemFourierVal37'
        or int(fourier.get('eval_panel_size', -1)) != 80
        or fourier.get('dev8_used_for_selection') is not False
        or fourier.get('eval_latent_policy')
        != 'fixed_per_panel_row_across_all_steps'
        or observed_weights != [0.1, 0.4, 0.4, 0.1]
        or reward_mode.get('delta_nsd') is not False
        or [round(float(x), 4) for x in hparams.get('stage_s_values', [])]
        != [0.0, 0.2, 0.4, 0.6, 0.8]
        or int(deployment.get('outer_steps', -1)) != 2
        or [round(float(x), 4) for x in deployment.get('fractions', [])]
        != [0.6667, 1.0]
        or [round(float(x), 4) for x in deployment.get('stage_s_values', [])]
        != [0.0, 0.6667]
        or int(deployment.get('ode_steps_per_outer_stage', -1)) != 4
        or int(deployment.get('total_nfe', -1)) != 8
        or deployment.get('used_for_all_reported_evaluation') is not True
    ):
        raise RuntimeError('five-stage train / 2x4 deploy metadata drift')

    baseline = json.loads(baseline_path.read_text())
    if (
        int(baseline.get('eval_n', -1)) <= 0
        or float(baseline.get('eval_iou', -1.0)) < 0.45
        or float(baseline.get('eval_dice', -1.0)) < 0.60
    ):
        raise RuntimeError(f'supervised baseline sanity gate failed: {baseline}')

    source_payload = torch.load(str(SOURCE), map_location='cpu', weights_only=False)
    output_payload = torch.load(str(latest), map_location='cpu', weights_only=False)
    if int(source_payload.get('step', -1)) != EXPECTED_SOURCE_STEP:
        raise RuntimeError('source step drift')
    if int(output_payload.get('step', -1)) != 2:
        raise RuntimeError('preflight checkpoint step is not 2')
    source_state = state_dict(source_payload)
    output_state = state_dict(output_payload)
    if set(source_state) != set(output_state):
        raise RuntimeError('checkpoint key set differs from source')

    changed_flow = []
    changed_frozen = []
    nonfinite = []
    for key, source_tensor in source_state.items():
        output_tensor = output_state[key]
        if not torch.is_tensor(source_tensor) or not torch.is_tensor(output_tensor):
            continue
        if tuple(source_tensor.shape) != tuple(output_tensor.shape):
            raise RuntimeError(f'tensor shape drift: {key}')
        if not torch.isfinite(output_tensor).all().item():
            nonfinite.append(key)
        if key.startswith(FLOW_PREFIXES):
            if not torch.equal(source_tensor, output_tensor):
                changed_flow.append(key)
        elif not torch.equal(source_tensor, output_tensor):
            changed_frozen.append(key)
    if nonfinite or changed_frozen or not changed_flow:
        raise RuntimeError(
            'state audit failed: nonfinite={} changed_frozen={} changed_flow={}'.format(
                nonfinite[:8], changed_frozen[:8], len(changed_flow)
            )
        )
    return {
        'latest_checkpoint': str(latest),
        'latest_checkpoint_sha256': sha256(latest),
        'effective_steps': [1, 2],
        'changed_flow_tensors': len(changed_flow),
        'changed_frozen_tensors': len(changed_frozen),
        'nonfinite_tensors': len(nonfinite),
        'five_stage_training_alignment_pass': True,
        'deployment_2x4_alignment_pass': True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('preflight', 'train'), required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()

    if Path.cwd().resolve() != REPO.resolve():
        raise RuntimeError(f'launcher must run from {REPO}')
    hashes = verify_inputs()
    output = PREFLIGHT if args.mode == 'preflight' else FORMAL
    if output.exists():
        raise FileExistsError(f'refusing to reuse output: {output}')
    if args.mode == 'train':
        preflight_manifest_path = PREFLIGHT / 'manifest.json'
        if not preflight_manifest_path.is_file():
            raise RuntimeError('formal training requires the completed preflight')
        preflight_manifest = json.loads(preflight_manifest_path.read_text())
        if (
            preflight_manifest.get('status') != 'COMPLETED'
            or not preflight_manifest.get('preflight_audit')
            or preflight_manifest.get('hashes') != hashes
        ):
            raise RuntimeError('preflight is missing, failed, or input hashes changed')

    gpu_checks = require_idle_gpu(args.gpu)
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / 'manifest.json'
    launch_log = output.parent / f'{output.name}.launch.log'
    manifest = {
        'status': 'STARTED',
        'mode': args.mode,
        'pipeline': 'moonvit_flowonly_five_action_fourier_grpo_2x4_deploy_v1',
        'output': str(output),
        'gpu': args.gpu,
        'launcher_pid': os.getpid(),
        'started_at': now(),
        'hashes': hashes,
        'gpu_idle_checks': gpu_checks,
        'source_checkpoint': str(SOURCE),
        'source_step': EXPECTED_SOURCE_STEP,
        'source_sha256': EXPECTED_SOURCE_SHA256,
        'model_parameters': EXPECTED_MODEL_PARAMETERS,
        'flow_trainable_parameters': EXPECTED_FLOW_PARAMETERS,
        'frozen_context_parameters': EXPECTED_CONTEXT_PARAMETERS,
        'training_contract': {
            'outer_fractions': [0.2, 0.25, 0.3333, 0.5, 1.0],
            'inner_steps_per_outer_stage': 4,
            'solver': 'ab2',
            'total_training_nfe': 20,
            'rl_action_count': 5,
            'fourier_modes': 8,
            'fourier_sigma_px': [0.8, 0.7, 0.6, 0.5, 0.4],
            'full_extrap_credit_map': [0, 1, 2, 3, 4],
            'stage_progress_s': [0.0, 0.2, 0.4, 0.59998, 0.79999],
            'inner_ab2_evaluations_are_rl_actions': False,
        },
        'deployment_contract': {
            'outer_fractions': [0.6667, 1.0],
            'inner_steps_per_outer_stage': 4,
            'solver': 'ab2',
            'total_nfe': 8,
            'stage_progress_s': [0.0, 0.6667],
            'used_for_all_reported_evaluation': True,
            'five_stage_training_rollout_is_not_reported_as_deployment': True,
        },
        'reward_weights': {
            'boundary': 0.1,
            'dice': 0.4,
            'iou': 0.4,
            'distance': 0.1,
        },
    }
    atomic_json(manifest_path, manifest)

    environment = os.environ.copy()
    existing_pythonpath = environment.get('PYTHONPATH', '').strip()
    environment.update({
        'CUDA_VISIBLE_DEVICES': str(args.gpu),
        'PYTHONUNBUFFERED': '1',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
        'RL_V4_MODEL_DIR': str(output),
        'PYTHONPATH': (
            str(REPO) if not existing_pythonpath
            else f'{REPO}{os.pathsep}{existing_pythonpath}'
        ),
    })
    if args.mode == 'preflight':
        environment['RL_V4_STEPS'] = '2'
    command = [str(PYTHON), str(TRAINER), '--cfg_file', str(CONFIG)]
    try:
        with launch_log.open('x', encoding='utf-8') as log_handle:
            process = subprocess.Popen(
                command,
                cwd=str(REPO),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=False,
            )
            manifest['trainer_pid'] = process.pid
            atomic_json(manifest_path, manifest)
            return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f'trainer exited with code {return_code}')
        if args.mode == 'preflight':
            manifest['preflight_audit'] = audit_preflight(output, launch_log)
        else:
            latest = output / 'checkpoints/latest.pt'
            if not latest.is_file():
                raise RuntimeError('formal training ended without latest checkpoint')
            manifest['latest_checkpoint_sha256'] = sha256(latest)
        manifest['status'] = 'COMPLETED'
        manifest['completed_at'] = now()
        atomic_json(manifest_path, manifest)
        return 0
    except BaseException as error:
        manifest['status'] = 'FAILED'
        manifest['failed_at'] = now()
        manifest['error'] = f'{type(error).__name__}: {error}'
        atomic_json(manifest_path, manifest)
        raise


if __name__ == '__main__':
    sys.exit(main())
