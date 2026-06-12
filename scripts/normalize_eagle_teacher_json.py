#!/usr/bin/env python3
import argparse
import json
import os


def _first(obj, keys, default=None):
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            return obj[key]
    return default


def _as_points(value):
    if value is None:
        return None
    pts = value
    if isinstance(pts, dict):
        pts = _first(pts, ['points', 'extreme_points', 'extremes'])
    if not isinstance(pts, list):
        return None
    flat = pts
    if flat and not isinstance(flat[0], list):
        if len(flat) < 8:
            return None
        flat = [[flat[i], flat[i + 1]] for i in range(0, min(len(flat), 8), 2)]
    if len(flat) < 4:
        return None
    return [[float(x), float(y)] for x, y in flat[:4]]


def _iter_records(obj):
    if isinstance(obj, dict) and isinstance(obj.get('samples'), list):
        yield from obj['samples']
    elif isinstance(obj, list):
        yield from obj
    elif isinstance(obj, dict):
        for path, value in obj.items():
            if isinstance(value, dict):
                rec = dict(value)
                rec.setdefault('img_path', path)
                yield rec
            elif isinstance(value, list):
                yield {'img_path': path, 'instances': value}


def normalize(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    samples = []
    for rec in _iter_records(raw):
        if not isinstance(rec, dict):
            continue
        img_path = _first(rec, ['img_path', 'image_path', 'path', 'file_name', 'image', 'id'])
        if img_path is None:
            continue
        instances = _first(rec, ['instances', 'objects', 'predictions', 'labels'], [rec])
        out_instances = []
        for inst in instances:
            if not isinstance(inst, dict):
                continue
            label = _first(inst, ['label_id', 'cls_id', 'class_id', 'category_id', 'label'])
            points = _as_points(_first(inst, ['extreme_points', 'extremes', 'points_4', 'extreme_4py', 'points']))
            if label is None or points is None:
                continue
            out_instances.append({
                'label_id': int(label),
                'extreme_points': points,
                'confidence': float(_first(inst, ['confidence', 'score'], 1.0)),
            })
        samples.append({'img_path': str(img_path), 'instances': out_instances})

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'samples': samples}, f, ensure_ascii=False, indent=2)
    print(f'wrote {len(samples)} samples -> {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Normalize Eagle/LocateAnything teacher outputs for V9 training.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    normalize(args.input, args.output)


if __name__ == '__main__':
    main()
