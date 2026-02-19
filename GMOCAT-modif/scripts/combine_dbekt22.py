#!/usr/bin/env python3
"""Combine DBEKT22 CSV (Transaction.csv) with mapping JSONs to produce
a train_task-like JSON file.

Usage:
  python combine_dbekt22.py --csv raw_data/DBEKT22/datasets/Transaction.csv \
      --map_dir data --out data/generated_train_task_dbekt22.json

Assumptions:
- `question_map_dbekt22.json` maps original question ids (as strings) to
  integer indices used by the model.
- `Transaction.csv` has columns including `student_id`, `question_id`, and
  `answer_state` (we convert `answer_state == '1'` to label 1, else 0).
"""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--map_dir", default="data")
    p.add_argument("--out", default="data/generated_train_task_dbekt22.json")
    p.add_argument("--time_col", default="start_time", help="timestamp column for ordering (optional)")
    return p.parse_args()


def load_question_map(map_dir):
    path = os.path.join(map_dir, "question_map_dbekt22.json")
    with open(path, "r", encoding="utf-8") as f:
        qm = json.load(f)
    # convert keys to int for lookup convenience
    return {int(k): v for k, v in qm.items()}


def read_transactions(csv_path, time_col="start_time"):
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # normalize numeric fields
            try:
                r['student_id'] = int(r.get('student_id') or r.get('user_id') or 0)
            except Exception:
                r['student_id'] = r.get('student_id')
            try:
                r['question_id'] = int(r.get('question_id') or r.get('item_id') or 0)
            except Exception:
                r['question_id'] = r.get('question_id')
            # keep answer_state raw
            # parse timestamp for ordering if present
            ts = r.get(time_col) or r.get('time') or ''
            try:
                # many timestamps include timezone, try fallback to raw string
                r['_ts'] = datetime.fromisoformat(ts) if ts else None
            except Exception:
                r['_ts'] = None
            rows.append(r)
    return rows


def build_student_sequences(rows, qmap):
    by_student = defaultdict(list)
    for r in rows:
        sid = r['student_id']
        qid = r['question_id']
        if qid is None:
            continue
        mapped = qmap.get(qid)
        if mapped is None:
            # skip questions not present in mapping
            continue
        # derive label: treat answer_state == '1' as correct
        ans = r.get('answer_state')
        label = 1 if str(ans).strip() == '1' else 0
        by_student[sid].append((r.get('_ts'), mapped, label))

    out = []
    for sid, seq in by_student.items():
        # sort by timestamp when available, otherwise keep original order
        seq_sorted = sorted(seq, key=lambda x: x[0] or datetime.min)
        q_ids = [m for _, m, _ in seq_sorted]
        labels = [l for _, _, l in seq_sorted]
        out.append({
            "student_id": sid,
            "q_ids": q_ids,
            "labels": labels,
            "log_num": len(q_ids),
        })
    return out


def main():
    args = parse_args()
    print("Loading question map from", args.map_dir)
    qmap = load_question_map(args.map_dir)
    print("Reading transactions from", args.csv)
    rows = read_transactions(args.csv, time_col=args.time_col)
    print(f"Loaded {len(rows)} transaction rows")
    combined = build_student_sequences(rows, qmap)
    print(f"Built sequences for {len(combined)} students")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2)
    print("Wrote combined JSON to", args.out)


if __name__ == '__main__':
    main()
