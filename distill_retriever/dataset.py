import math
import random
from dataclasses import dataclass

from transformers import DataCollatorWithPadding

from FlagEmbedding.abc.finetune.embedder import (
    AbsEmbedderTrainDataset,
    AbsEmbedderSameDatasetTrainDataset,
)


class EncoderOnlyEmbedderDistillTrainDataset(AbsEmbedderTrainDataset):
    """Train dataset with extra query_gen field for query-rewrite distillation."""

    def __getitem__(self, item):
        data = self.dataset[item]
        train_group_size = self.args.train_group_size

        query = data["query"]
        query_gen = data.get("query_gen", query)

        if self.args.query_instruction_for_retrieval is not None:
            prompt = data["prompt"] if "prompt" in data else self.args.query_instruction_for_retrieval
            query = self.args.query_instruction_format.format(prompt, query)
            query_gen = self.args.query_instruction_format.format(prompt, query_gen)

        passages = []
        teacher_scores = []

        assert isinstance(data["pos"], list) and isinstance(data["neg"], list)

        pos_idx = random.choice(list(range(len(data["pos"]))))
        passages.append(self._shuffle_text(data["pos"][pos_idx]))

        neg_idxs = []
        if train_group_size > 1 and len(data["neg"]) > 0:
            neg_all_idx = list(range(len(data["neg"])))
            if len(data["neg"]) < train_group_size - 1:
                num = math.ceil((train_group_size - 1) / len(data["neg"]))
                neg_idxs = random.sample(neg_all_idx * num, train_group_size - 1)
            else:
                neg_idxs = random.sample(neg_all_idx, self.args.train_group_size - 1)
            for neg_idx in neg_idxs:
                passages.append(data["neg"][neg_idx])

        if self.args.knowledge_distillation:
            assert isinstance(data["pos_scores"], list) and isinstance(data["neg_scores"], list)
            teacher_scores.append(data["pos_scores"][pos_idx])
            for neg_idx in neg_idxs:
                teacher_scores.append(data["neg_scores"][neg_idx])
            if not all(isinstance(score, (int, float)) for score in teacher_scores):
                raise ValueError("pos_score or neg_score must be digit")
        else:
            teacher_scores = None

        if self.args.passage_instruction_for_retrieval is not None:
            passages = [
                self.args.passage_instruction_format.format(
                    self.args.passage_instruction_for_retrieval, p
                )
                for p in passages
            ]

        return query, query_gen, passages, teacher_scores


@dataclass
class EncoderOnlyEmbedderDistillCollator(DataCollatorWithPadding):
    """Collator for query + query_gen + passages."""

    query_max_len: int = 32
    passage_max_len: int = 128
    sub_batch_size: int = -1

    def __call__(self, features):
        queries = [f[0] for f in features]
        query_gens = [f[1] for f in features]
        passages = [f[2] for f in features]
        teacher_scores = [f[3] for f in features]

        if teacher_scores[0] is None:
            teacher_scores = None
        elif isinstance(teacher_scores[0], list):
            teacher_scores = sum(teacher_scores, [])

        if isinstance(queries[0], list):
            queries = sum(queries, [])
        if isinstance(query_gens[0], list):
            query_gens = sum(query_gens, [])
        if isinstance(passages[0], list):
            passages = sum(passages, [])

        queries_inputs = self.tokenizer(
            queries,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors=None,
        )
        query_gens_inputs = self.tokenizer(
            query_gens,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors=None,
        )
        passages_inputs = self.tokenizer(
            passages,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors=None,
        )

        if self.sub_batch_size is None or self.sub_batch_size <= 0:
            q_collated = self.tokenizer.pad(
                queries_inputs,
                padding=self.padding,
                max_length=self.query_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            q_gen_collated = self.tokenizer.pad(
                query_gens_inputs,
                padding=self.padding,
                max_length=self.query_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            d_collated = self.tokenizer.pad(
                passages_inputs,
                padding=self.padding,
                max_length=self.passage_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
        else:
            batch_size = self.sub_batch_size

            q_collated = []
            for i in range(0, len(queries_inputs["attention_mask"]), batch_size):
                start = i
                end = min(len(queries_inputs["attention_mask"]), i + batch_size)
                sub_features = {k: v[start:end] for k, v in queries_inputs.items()}
                q_collated.append(
                    self.tokenizer.pad(
                        sub_features,
                        padding=self.padding,
                        max_length=self.query_max_len,
                        pad_to_multiple_of=self.pad_to_multiple_of,
                        return_tensors=self.return_tensors,
                    )
                )

            q_gen_collated = []
            for i in range(0, len(query_gens_inputs["attention_mask"]), batch_size):
                start = i
                end = min(len(query_gens_inputs["attention_mask"]), i + batch_size)
                sub_features = {k: v[start:end] for k, v in query_gens_inputs.items()}
                q_gen_collated.append(
                    self.tokenizer.pad(
                        sub_features,
                        padding=self.padding,
                        max_length=self.query_max_len,
                        pad_to_multiple_of=self.pad_to_multiple_of,
                        return_tensors=self.return_tensors,
                    )
                )

            d_collated = []
            for i in range(0, len(passages_inputs["attention_mask"]), batch_size):
                start = i
                end = min(len(passages_inputs["attention_mask"]), i + batch_size)
                sub_features = {k: v[start:end] for k, v in passages_inputs.items()}
                d_collated.append(
                    self.tokenizer.pad(
                        sub_features,
                        padding=self.padding,
                        max_length=self.passage_max_len,
                        pad_to_multiple_of=self.pad_to_multiple_of,
                        return_tensors=self.return_tensors,
                    )
                )

        return {
            "queries": q_collated,
            "query_gens": q_gen_collated,
            "passages": d_collated,
            "teacher_scores": teacher_scores,
            "no_in_batch_neg_flag": False,
        }


class EncoderOnlyEmbedderDistillSameDatasetTrainDataset(AbsEmbedderSameDatasetTrainDataset):
    """Same-dataset sampler with query_gen support."""

    def _create_batch_data(self, batch_raw_data):
        queries, query_gens, passages, teacher_scores = [], [], [], []

        train_group_size, data_type = self._get_train_group_size(batch_raw_data)

        for i in range(len(batch_raw_data["query"])):
            if data_type is not None:
                assert batch_raw_data["type"][i] == data_type, "Data type is not consistent in the same batch"

            prompt = batch_raw_data["prompt"][i] if "prompt" in batch_raw_data else self.args.query_instruction_for_retrieval
            query = batch_raw_data["query"][i]
            query_gen = (
                batch_raw_data["query_gen"][i]
                if "query_gen" in batch_raw_data and batch_raw_data["query_gen"][i] is not None
                else query
            )

            query = self.args.query_instruction_format.format(prompt, query)
            query_gen = self.args.query_instruction_format.format(prompt, query_gen)

            queries.append(query)
            query_gens.append(query_gen)

            tmp_passages = []
            pos_idx = random.choice(list(range(len(batch_raw_data["pos"][i]))))
            pos = self._shuffle_text(batch_raw_data["pos"][i][pos_idx])
            tmp_passages.append(pos)

            neg_idxs = []
            if train_group_size > 1 and len(batch_raw_data["neg"][i]) > 0:
                neg_all_idx = list(range(len(batch_raw_data["neg"][i])))
                if len(batch_raw_data["neg"][i]) < train_group_size - 1:
                    num = math.ceil((train_group_size - 1) / len(batch_raw_data["neg"][i]))
                    neg_idxs = random.sample(neg_all_idx * num, train_group_size - 1)
                else:
                    neg_idxs = random.sample(neg_all_idx, train_group_size - 1)

                for neg_idx in neg_idxs:
                    tmp_passages.append(batch_raw_data["neg"][i][neg_idx])

            if self.args.knowledge_distillation:
                if "pos_scores" in batch_raw_data and batch_raw_data["pos_scores"][i] is not None:
                    teacher_scores.append(batch_raw_data["pos_scores"][i][pos_idx])
                for neg_idx in neg_idxs:
                    if "neg_scores" in batch_raw_data and batch_raw_data["neg_scores"][i] is not None:
                        teacher_scores.append(batch_raw_data["neg_scores"][i][neg_idx])
            else:
                teacher_scores = None

            if data_type is not None and data_type in ["symmetric_sts", "symmetric_clustering"]:
                tmp_passages = [
                    self.args.query_instruction_format.format(prompt, p)
                    for p in tmp_passages
                ]
            else:
                if self.args.passage_instruction_for_retrieval is not None:
                    tmp_passages = [
                        self.args.passage_instruction_format.format(
                            self.args.passage_instruction_for_retrieval, p
                        )
                        for p in tmp_passages
                    ]

            passages.extend(tmp_passages)

            if teacher_scores is not None and len(teacher_scores) > 0 and len(passages) > 0:
                assert len(teacher_scores) == len(passages)

        return queries, query_gens, passages, teacher_scores

    def __getitem__(self, _):
        batch_indices, no_in_batch_neg_flag = self.batch_datas[self.step]
        cur_batch_size = int(len(batch_indices) / self.num_processes)
        batch_indices = batch_indices[self.process_index * cur_batch_size: (self.process_index + 1) * cur_batch_size]
        batch_data = self.dataset[batch_indices]
        self.step += 1
        queries, query_gens, passages, teacher_scores = self._create_batch_data(batch_raw_data=batch_data)
        return queries, query_gens, passages, teacher_scores, no_in_batch_neg_flag


@dataclass
class EncoderOnlyEmbedderDistillSameDatasetCollator(DataCollatorWithPadding):
    """Same-dataset collator with query_gen support."""

    query_max_len: int = 32
    passage_max_len: int = 128
    sub_batch_size: int = -1

    def __call__(self, features):
        queries = features[0][0]
        query_gens = features[0][1]
        passages = features[0][2]
        teacher_scores = features[0][3]
        no_in_batch_neg_flag = features[0][4]

        queries_inputs = self.tokenizer(
            queries,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors=None,
        )
        query_gens_inputs = self.tokenizer(
            query_gens,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors=None,
        )
        passages_inputs = self.tokenizer(
            passages,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors=None,
        )

        if self.sub_batch_size is None or self.sub_batch_size <= 0:
            q_collated = self.tokenizer.pad(
                queries_inputs,
                padding=self.padding,
                max_length=self.query_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            q_gen_collated = self.tokenizer.pad(
                query_gens_inputs,
                padding=self.padding,
                max_length=self.query_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            d_collated = self.tokenizer.pad(
                passages_inputs,
                padding=self.padding,
                max_length=self.passage_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
        else:
            batch_size = self.sub_batch_size

            q_collated = []
            for i in range(0, len(queries_inputs["attention_mask"]), batch_size):
                start = i
                end = min(len(queries_inputs["attention_mask"]), i + batch_size)
                sub_features = {k: v[start:end] for k, v in queries_inputs.items()}
                q_collated.append(
                    self.tokenizer.pad(
                        sub_features,
                        padding=self.padding,
                        max_length=self.query_max_len,
                        pad_to_multiple_of=self.pad_to_multiple_of,
                        return_tensors=self.return_tensors,
                    )
                )

            q_gen_collated = []
            for i in range(0, len(query_gens_inputs["attention_mask"]), batch_size):
                start = i
                end = min(len(query_gens_inputs["attention_mask"]), i + batch_size)
                sub_features = {k: v[start:end] for k, v in query_gens_inputs.items()}
                q_gen_collated.append(
                    self.tokenizer.pad(
                        sub_features,
                        padding=self.padding,
                        max_length=self.query_max_len,
                        pad_to_multiple_of=self.pad_to_multiple_of,
                        return_tensors=self.return_tensors,
                    )
                )

            d_collated = []
            for i in range(0, len(passages_inputs["attention_mask"]), batch_size):
                start = i
                end = min(len(passages_inputs["attention_mask"]), i + batch_size)
                sub_features = {k: v[start:end] for k, v in passages_inputs.items()}
                d_collated.append(
                    self.tokenizer.pad(
                        sub_features,
                        padding=self.padding,
                        max_length=self.passage_max_len,
                        pad_to_multiple_of=self.pad_to_multiple_of,
                        return_tensors=self.return_tensors,
                    )
                )

        if isinstance(teacher_scores, list) and len(teacher_scores) == 0:
            teacher_scores = None

        return {
            "queries": q_collated,
            "query_gens": q_gen_collated,
            "passages": d_collated,
            "teacher_scores": teacher_scores,
            "no_in_batch_neg_flag": no_in_batch_neg_flag,
        }
