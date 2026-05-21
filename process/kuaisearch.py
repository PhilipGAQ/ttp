#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

DEFAULT_RECALL = Path("path/to/recall_lite")
DEFAULT_ITEMS = Path("path/to/items_lite")
DEFAULT_TRAIN_OUT = Path("path/to/train.jsonl")
DEFAULT_TEST_OUT_DIR = Path("path/to/test")


@dataclass
class Session:
    idx: int
    user_id: int
    time_index: int
    split: str
    query: str
    clicked: List[int]
    purchased: List[int]


@dataclass
class SplitInfo:
    use_split_test: bool
    max_time_index: int
    split_counter: Dict[str, int]


@dataclass
class TrainBuildStats:
    rows: int = 0
    pos_docs: int = 0
    hist_non_empty: int = 0
    mapped_pos_docs: int = 0
    mapped_clicked_hist_docs: int = 0
    mapped_purchase_hist_docs: int = 0
    missing_pos_docs: int = 0
    missing_clicked_hist_docs: int = 0
    missing_purchase_hist_docs: int = 0

    def to_dict(self) -> Dict[str, Any]:
        hist_non_empty_ratio = 0.0 if self.rows == 0 else self.hist_non_empty / self.rows
        avg_pos_docs = 0.0 if self.rows == 0 else self.pos_docs / self.rows
        return {
            "rows": self.rows,
            "pos_docs": self.pos_docs,
            "avg_pos_docs": avg_pos_docs,
            "hist_non_empty": self.hist_non_empty,
            "hist_non_empty_ratio": hist_non_empty_ratio,
            "mapped_pos_docs": self.mapped_pos_docs,
            "mapped_clicked_hist_docs": self.mapped_clicked_hist_docs,
            "mapped_purchase_hist_docs": self.mapped_purchase_hist_docs,
            "missing_pos_docs": self.missing_pos_docs,
            "missing_clicked_hist_docs": self.missing_clicked_hist_docs,
            "missing_purchase_hist_docs": self.missing_purchase_hist_docs,
        }


@dataclass
class TestBuildStats:
    queries: int = 0
    qrels: int = 0
    corpus: int = 0
    missing_hist_docs: int = 0
    missing_qrel_docs: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queries": self.queries,
            "qrels": self.qrels,
            "corpus": self.corpus,
            "missing_hist_docs": self.missing_hist_docs,
            "missing_qrel_docs": self.missing_qrel_docs,
        }

def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_item_text(item: Dict[str, Any]) -> str:
    brand_name = safe_text(item.get("brand_name"))
    item_title = safe_text(item.get("item_title"))
    seller_name = safe_text(item.get("seller_name"))

    cat_l1 = safe_text(item.get("category_level1_name"))
    cat_l2 = safe_text(item.get("category_level2_name"))
    cat_l3 = safe_text(item.get("category_level3_name"))

    parts: List[str] = []
    if brand_name and brand_name != "UNKNOWN":
        parts.append(f"Brand: {brand_name}")
    parts.append(f"Title: {item_title}")

    cats = [x for x in [cat_l1, cat_l2, cat_l3] if x and x != "UNKNOWN"]
    if cats:
        parts.append(f"Category: {' '.join(cats)}")

    parts.append(f"Seller: {seller_name}")
    return " ".join(parts)


def merge_unique(a: List[int], b: List[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in a:
        if x not in seen:
            seen.add(x)
            out.append(x)
    for x in b:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_sessions(recall_path: Path) -> List[Session]:
    sessions: List[Session] = []
    with recall_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            sessions.append(
                Session(
                    idx=idx,
                    user_id=int(row["user_id"]),
                    time_index=int(row["time_index"]),
                    split=str(row.get("split", "")),
                    query=safe_text(row.get("query")),
                    clicked=[int(x) for x in row.get("clicked_item_ids", [])],
                    purchased=[int(x) for x in row.get("purchased_item_ids", [])],
                )
            )
    return sessions


def analyze_split(sessions: List[Session]) -> SplitInfo:
    split_counter = Counter([s.split for s in sessions if s.split])
    use_split_test = split_counter.get("test", 0) > 0
    max_time_index = max(s.time_index for s in sessions)
    return SplitInfo(
        use_split_test=use_split_test,
        max_time_index=max_time_index,
        split_counter=dict(split_counter),
    )


def build_base_rows(
    sessions: List[Session],
    max_train_rows: int,
    max_test_rows: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], SplitInfo]:
    split_info = analyze_split(sessions)

    by_user: Dict[int, List[Session]] = defaultdict(list)
    for s in sessions:
        by_user[s.user_id].append(s)

    train_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    qrels_rows: List[Dict[str, Any]] = []

    next_train_id = 1
    next_test_qid = 0

    for user_sessions in by_user.values():
        user_sessions.sort(key=lambda x: (x.time_index, x.idx))
        clicked_cum: List[int] = []
        purchased_cum: List[int] = []

        i = 0
        n = len(user_sessions)
        while i < n:
            time_index = user_sessions[i].time_index
            j = i
            while j < n and user_sessions[j].time_index == time_index:
                j += 1

            for k in range(i, j):
                session = user_sessions[k]
                pos_ids = merge_unique(session.clicked, session.purchased)
                if not pos_ids:
                    continue

                if split_info.use_split_test:
                    is_test = session.split == "test"
                    is_train = session.split == "train"
                else:
                    is_test = session.time_index == split_info.max_time_index
                    is_train = not is_test

                base_row = {
                    "query": session.query,
                    "clicked_hist_ids": clicked_cum.copy(),
                    "purchase_hist_ids": purchased_cum.copy(),
                    "pos_ids": pos_ids,
                }

                if is_train and (max_train_rows < 0 or len(train_rows) < max_train_rows):
                    row = dict(base_row)
                    row["id"] = next_train_id
                    train_rows.append(row)
                    next_train_id += 1

                if is_test and (max_test_rows < 0 or len(test_rows) < max_test_rows):
                    qid = str(next_test_qid)
                    row = dict(base_row)
                    row["qid"] = qid
                    test_rows.append(row)
                    for item_id in pos_ids:
                        qrels_rows.append(
                            {
                                "qid": qid,
                                "docid": str(item_id),
                                "relevance": 1,
                            }
                        )
                    next_test_qid += 1

            for k in range(i, j):
                clicked_cum.extend(user_sessions[k].clicked)
                purchased_cum.extend(user_sessions[k].purchased)

            i = j

    return train_rows, test_rows, qrels_rows, split_info


def load_item_texts(items_path: Path) -> Dict[int, str]:
    item_text_map: Dict[int, str] = {}
    with items_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            item_id = int(item["item_id"])
            item_text_map[item_id] = build_item_text(item)
    return item_text_map


def map_ids_to_texts(ids: List[int], item_text_map: Dict[int, str]) -> Tuple[List[str], int]:
    texts: List[str] = []
    missing = 0
    for item_id in ids:
        text = item_text_map.get(item_id)
        if text is None:
            missing += 1
            continue
        texts.append(text)
    return texts, missing

def build_train_dataset(
    train_rows: List[Dict[str, Any]],
    item_text_map: Dict[int, str],
    output_path: Path,
) -> TrainBuildStats:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = TrainBuildStats()

    with output_path.open("w", encoding="utf-8") as fout:
        for row in train_rows:
            pos_texts, pos_missing = map_ids_to_texts(row["pos_ids"], item_text_map)
            clicked_hist, clicked_missing = map_ids_to_texts(
                row["clicked_hist_ids"], item_text_map
            )
            purchase_hist, purchase_missing = map_ids_to_texts(
                row["purchase_hist_ids"], item_text_map
            )
            hist_list = purchase_hist + clicked_hist

            stats.mapped_pos_docs += len(pos_texts)
            stats.mapped_clicked_hist_docs += len(clicked_hist)
            stats.mapped_purchase_hist_docs += len(purchase_hist)
            stats.missing_pos_docs += pos_missing
            stats.missing_clicked_hist_docs += clicked_missing
            stats.missing_purchase_hist_docs += purchase_missing

            out_row = {
                "id": row["id"],
                "query": row["query"],
                "pos": pos_texts,
                "neg": [],
                "hist_list": hist_list,
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            stats.rows += 1
            stats.pos_docs += len(pos_texts)
            if hist_list:
                stats.hist_non_empty += 1

    return stats


def build_test_dataset(
    test_rows: List[Dict[str, Any]],
    qrels_rows: List[Dict[str, Any]],
    item_text_map: Dict[int, str],
    out_dir: Path,
) -> TestBuildStats:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = TestBuildStats()

    query_rows: List[Dict[str, Any]] = []
    valid_qids = set()

    for row in test_rows:
        purchase_hist, purchase_missing = map_ids_to_texts(
            row["purchase_hist_ids"], item_text_map
        )
        clicked_hist, clicked_missing = map_ids_to_texts(
            row["clicked_hist_ids"], item_text_map
        )
        history_texts = purchase_hist + clicked_hist

        query_rows.append(
            {
                "id": row["qid"],
                "text": row["query"],
                "user_hist": history_texts,
            }
        )
        valid_qids.add(row["qid"])
        stats.missing_hist_docs += purchase_missing + clicked_missing

    valid_doc_ids = {str(item_id) for item_id in item_text_map.keys()}
    filtered_qrels: List[Dict[str, Any]] = []
    for row in qrels_rows:
        if row["qid"] not in valid_qids:
            continue
        if row["docid"] not in valid_doc_ids:
            stats.missing_qrel_docs += 1
            continue
        filtered_qrels.append(row)

    write_jsonl(out_dir / "test_queries.jsonl", query_rows)
    write_jsonl(
        out_dir / "corpus.jsonl",
        ({"id": str(item_id), "text": text} for item_id, text in item_text_map.items()),
    )
    write_jsonl(out_dir / "test_qrels.jsonl", filtered_qrels)

    stats.queries = len(query_rows)
    stats.qrels = len(filtered_qrels)
    stats.corpus = len(item_text_map)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build unified KuaiSearch train/test data from raw recall_lite and items_lite"
    )
    parser.add_argument("--recall", type=Path, default=DEFAULT_RECALL)
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--train_output", type=Path, default=DEFAULT_TRAIN_OUT)
    parser.add_argument("--test_output_dir", type=Path, default=DEFAULT_TEST_OUT_DIR)
    parser.add_argument(
        "--stats_output",
        type=Path,
        default=Path("output/kuaisearch/build_stats.json"),
    )
    parser.add_argument("--max_train_rows", type=int, default=-1)
    parser.add_argument("--max_test_rows", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sessions = load_sessions(args.recall)
    train_rows, test_rows, qrels_rows, split_info = build_base_rows(
        sessions=sessions,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
    )
    item_text_map = load_item_texts(args.items)

    train_stats = build_train_dataset(
        train_rows=train_rows,
        item_text_map=item_text_map,
        output_path=args.train_output,
    )
    test_stats = build_test_dataset(
        test_rows=test_rows,
        qrels_rows=qrels_rows,
        item_text_map=item_text_map,
        out_dir=args.test_output_dir,
    )

    summary = {
        "recall": str(args.recall),
        "items": str(args.items),
        "split_info": {
            "use_split_test": split_info.use_split_test,
            "max_time_index": split_info.max_time_index,
            "split_counter": split_info.split_counter,
        },
        "train_output": str(args.train_output),
        "test_output_dir": str(args.test_output_dir),
        "train": train_stats.to_dict(),
        "test": test_stats.to_dict(),
    }

    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    with args.stats_output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
