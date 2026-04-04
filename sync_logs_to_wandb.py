import argparse
import json
import os


def _flatten_entry(entry: dict) -> dict:
    payload = dict(entry)
    loss_stats = payload.pop('loss_stats', None)
    if isinstance(loss_stats, dict):
        for k, v in loss_stats.items():
            payload[f"loss_stats/{k}"] = v
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default='data/outputs/one_sample/logs.jsonl')
    parser.add_argument('--project', type=str, default=os.environ.get('WANDB_PROJECT', 'DiffusionSnake'))
    parser.add_argument('--entity', type=str, default=os.environ.get('WANDB_ENTITY', None))
    parser.add_argument('--name', type=str, default=os.environ.get('WANDB_NAME', None))
    parser.add_argument('--run_id', type=str, default=os.environ.get('WANDB_RUN_ID', None))
    parser.add_argument('--wandb_dir', type=str, default=os.environ.get('WANDB_DIR', None))
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--end_step', type=int, default=-1)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    try:
        import wandb
    except Exception as e:
        raise RuntimeError('wandb is not available. Please install wandb first.') from e

    if not os.path.exists(args.log_path):
        raise FileNotFoundError(args.log_path)

    init_kwargs = {
        'project': args.project,
        'entity': args.entity,
        'name': args.name,
        'resume': 'allow',
    }
    if args.wandb_dir:
        init_kwargs['dir'] = args.wandb_dir
    if args.run_id:
        init_kwargs['id'] = args.run_id

    run = wandb.init(**init_kwargs)

    uploaded = 0
    skipped = 0
    with open(args.log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue

            try:
                step = int(entry.get('step', -1))
            except Exception:
                step = -1

            if step < 0:
                skipped += 1
                continue
            if args.start_step and step < int(args.start_step):
                skipped += 1
                continue
            if args.end_step is not None and int(args.end_step) >= 0 and step > int(args.end_step):
                skipped += 1
                continue

            payload = _flatten_entry(entry)

            if not args.dry_run:
                try:
                    wandb.log(payload, step=step)
                except Exception:
                    continue
            uploaded += 1

    if not args.dry_run:
        try:
            wandb.finish()
        except Exception:
            pass

    print(f"done. uploaded={uploaded} skipped={skipped} run={getattr(run, 'url', None)}")


if __name__ == '__main__':
    main()
