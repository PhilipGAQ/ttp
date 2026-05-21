#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Tuple

HISTORY_TYPES = ["search", "recommend", "review", "other"]
DEFAULT_INPUT_DIR = Path("path/to/origin")
DEFAULT_OUTPUT_DIR = Path("path/to/output")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [normalize_text(v) for v in value]
        return " ".join([p for p in parts if p]).strip()
    if isinstance(value, dict):
        parts = []
        for key in sorted(value.keys()):
            text = normalize_text(value[key])
            if text:
                parts.append(f"{key}: {text}")
        return " ".join(parts).strip()
    return str(value).strip()


def to_timestamp(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def normalize_type(value: Any) -> str:
    text = normalize_text(value).lower()
    if text in {"rec", "recommend"}:
        return "recommend"
    if text in {"search", "review"}:
        return text
    return "other"


def safe_copy_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def get_first_feature_text(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            text = normalize_text(item)
            if text:
                return text
        return ""
    return normalize_text(value)


def product_to_text(product: Dict[str, Any]) -> str:
    pairs = [
        ("title", normalize_text(product.get("title"))),
        ("main_category", normalize_text(product.get("main_category"))),
        ("features", get_first_feature_text(product.get("features"))),
    ]
    return " [SEP] ".join(
        [f"{key}: {value}" for key, value in pairs if value]
    ).strip()


def make_query_id(user_id: str, timestamp: int, idx: int, split: str) -> str:
    return f"{user_id}_{timestamp}_{split}_{idx}"


def load_inputs(input_dir: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    user_history = load_json(input_dir / "user_history.json")
    all_products = load_json(input_dir / "all_products.json")
    user_instructions = load_json(input_dir / "user_instructions.json")
    return user_history, all_products, user_instructions


def resolve_product(
    parent_asin: str,
    product_info: Dict[str, Any],
    doc_store: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if parent_asin and parent_asin in doc_store:
        return doc_store[parent_asin]
    if product_info:
        resolved = dict(product_info)
        if parent_asin:
            resolved.setdefault("parent_asin", parent_asin)
            doc_store.setdefault(parent_asin, resolved)
        return resolved
    return {}


def extract_product_text_from_history(
    item: Dict[str, Any],
    doc_store: Dict[str, Dict[str, Any]],
) -> str:
    product_info = safe_copy_dict(item.get("product_info"))
    parent_asin = normalize_text(product_info.get("parent_asin") or item.get("parent_asin"))
    resolved = resolve_product(parent_asin, product_info, doc_store)
    if not resolved:
        return ""
    return product_to_text(resolved)


def extract_target(
    instruction: Dict[str, Any],
    doc_store: Dict[str, Dict[str, Any]],
) -> Tuple[str, str]:
    target = safe_copy_dict(instruction.get("target"))
    product_info = safe_copy_dict(target.get("product_info"))
    parent_asin = normalize_text(product_info.get("parent_asin") or target.get("parent_asin"))
    if not parent_asin:
        return "", ""
    resolved = resolve_product(parent_asin, product_info, doc_store)
    if not resolved:
        return parent_asin, ""
    return parent_asin, product_to_text(resolved)


def snapshot_history(history_state: Dict[str, Deque[str]]) -> Dict[str, List[str]]:
    snapshot: Dict[str, List[str]] = {}
    for history_type in HISTORY_TYPES:
        snapshot[history_type] = list(history_state.get(history_type, deque()))
    return snapshot


def count_total_history_items(history_snapshot: Dict[str, List[str]]) -> int:
    return sum(len(history_snapshot.get(history_type, [])) for history_type in HISTORY_TYPES)


def build_dataset(
    input_dir: Path,
    output_dir: Path,
    max_hist_per_type: int,
    max_users: int,
) -> Dict[str, Any]:
    ensure_dir(output_dir)
    user_history_map, product_map, user_instructions = load_inputs(input_dir)
    doc_store: Dict[str, Dict[str, Any]] = {str(k): v for k, v in product_map.items()}

    train_by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in user_instructions.get("train", []):
        user_id = normalize_text(row.get("user_id"))
        if user_id:
            train_by_user[user_id].append(row)

    test_by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in user_instructions.get("test", []):
        user_id = normalize_text(row.get("user_id"))
        if user_id:
            test_by_user[user_id].append(row)

    train_rows: List[Dict[str, Any]] = []
    test_queries: List[Dict[str, Any]] = []
    test_qrels: List[Dict[str, Any]] = []

    train_type_counter: Counter[str] = Counter()
    test_type_counter: Counter[str] = Counter()
    all_users = sorted(set(train_by_user.keys()) | set(test_by_user.keys()))
    if max_users > 0:
        all_users = all_users[:max_users]

    stats: Dict[str, Any] = {
        "num_users": len(all_users),
        "num_products_input": len(product_map),
        "skipped_missing_query": 0,
        "skipped_missing_target_asin": 0,
        "skipped_missing_target_product": 0,
    }
    train_hist_len_sum = 0
    test_hist_len_sum = 0

    for user_id in all_users:
        history_items = [
            item
            for item in user_history_map.get(user_id, [])
            if normalize_text(item.get("split")).lower() == "history"
        ]
        history_items.sort(
            key=lambda item: to_timestamp(
                safe_copy_dict(item.get("review")).get("timestamp") or item.get("timestamp")
            )
        )

        history_state: Dict[str, Deque[str]] = defaultdict(
            lambda: deque(maxlen=max_hist_per_type)
        )
        for hist_item in history_items:
            hist_type = normalize_type(hist_item.get("type") or "review")
            hist_text = extract_product_text_from_history(hist_item, doc_store)
            if hist_text:
                history_state[hist_type].append(hist_text)

        user_train = sorted(
            train_by_user.get(user_id, []),
            key=lambda inst: to_timestamp(inst.get("timestamp")),
        )
        train_idx = 0
        for inst in user_train:
            query = normalize_text(inst.get("task"))
            timestamp = to_timestamp(inst.get("timestamp"))
            event_type = normalize_type(inst.get("type"))
            target_asin, pos_text = extract_target(inst, doc_store)

            if not query:
                stats["skipped_missing_query"] += 1
                continue
            if not target_asin:
                stats["skipped_missing_target_asin"] += 1
                continue
            if not pos_text:
                stats["skipped_missing_target_product"] += 1
                continue

            qid = make_query_id(user_id, timestamp, train_idx, "train")
            history_snapshot = snapshot_history(history_state)
            train_hist_len_sum += count_total_history_items(history_snapshot)
            train_rows.append(
                {
                    "id": qid,
                    "query": query,
                    "pos": [pos_text],
                    "neg": [],
                    "hist_list": history_snapshot,
                }
            )
            train_type_counter[event_type] += 1
            train_idx += 1
            history_state[event_type].append(pos_text)

        user_test = sorted(
            test_by_user.get(user_id, []),
            key=lambda inst: to_timestamp(inst.get("timestamp")),
        )
        test_idx = 0
        for inst in user_test:
            query = normalize_text(inst.get("task"))
            timestamp = to_timestamp(inst.get("timestamp"))
            event_type = normalize_type(inst.get("type"))
            target_asin, pos_text = extract_target(inst, doc_store)

            if not query:
                stats["skipped_missing_query"] += 1
                continue
            if not target_asin:
                stats["skipped_missing_target_asin"] += 1
                continue
            if not pos_text:
                stats["skipped_missing_target_product"] += 1
                continue

            qid = make_query_id(user_id, timestamp, test_idx, "test")
            history_snapshot = snapshot_history(history_state)
            test_hist_len_sum += count_total_history_items(history_snapshot)
            test_queries.append(
                {
                    "id": qid,
                    "text": query,
                    "user_hist": history_snapshot,
                }
            )
            test_qrels.append(
                {
                    "qid": qid,
                    "docid": target_asin,
                    "relevance": 1,
                }
            )
            test_type_counter[event_type] += 1
            test_idx += 1
            history_state[event_type].append(pos_text)

    corpus_rows = [
        {"id": parent_asin, "text": product_to_text(product)}
        for parent_asin, product in sorted(doc_store.items())
        if product_to_text(product)
    ]

    train_path = output_dir / "train.jsonl"
    corpus_path = output_dir / "corpus.jsonl"
    test_queries_path = output_dir / "test_queries.jsonl"
    test_qrels_path = output_dir / "test_qrels.jsonl"
    stats_path = output_dir / "stats.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(corpus_path, corpus_rows)
    write_jsonl(test_queries_path, test_queries)
    write_jsonl(test_qrels_path, test_qrels)

    total_example_count = len(train_rows) + len(test_queries)
    total_hist_len_sum = train_hist_len_sum + test_hist_len_sum

    stats.update(
        {
            "num_products_output": len(corpus_rows),
            "num_train": len(train_rows),
            "num_test_queries": len(test_queries),
            "num_test_qrels": len(test_qrels),
            "avg_history_length": (
                total_hist_len_sum / total_example_count if total_example_count else 0.0
            ),
            "avg_train_history_length": (
                train_hist_len_sum / len(train_rows) if train_rows else 0.0
            ),
            "avg_test_history_length": (
                test_hist_len_sum / len(test_queries) if test_queries else 0.0
            ),
            "train_type_count": dict(sorted(train_type_counter.items())),
            "test_type_count": dict(sorted(test_type_counter.items())),
            "output_files": {
                "train": str(train_path),
                "corpus": str(corpus_path),
                "test_queries": str(test_queries_path),
                "test_qrels": str(test_qrels_path),
                "stats": str(stats_path),
            },
        }
    )

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PersonalWAB train/test data with history kept by type"
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_hist_per_type", type=int, default=100)
    parser.add_argument("--max_users", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_hist_per_type=args.max_hist_per_type,
        max_users=args.max_users,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
