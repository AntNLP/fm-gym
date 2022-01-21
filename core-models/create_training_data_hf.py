import argparse
import json
import logging as logger
import os
import pickle
import random
import time
from typing import List

import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import DataLoader, Dataset
from transformers import (BertTokenizer, DataCollatorForLanguageModeling,
                          PreTrainedTokenizer)


# has added truncate_seq_pair for keeping example length not great than block_size
class TextDatasetForNextSentencePrediction(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        short_seq_probability=0.1,
        nsp_probability=0.5,
    ):
        if not os.path.isfile(file_path):
            raise ValueError(f"Input file path {file_path} not found")

        self.short_seq_probability = short_seq_probability
        self.nsp_probability = nsp_probability

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            f"cached_nsp_{tokenizer.__class__.__name__}_{block_size}_{filename}",
        )

        self.tokenizer = tokenizer

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"

        # Input file format:
        # (1) One sentence per line. These should ideally be actual sentences, not
        # entire paragraphs or arbitrary spans of text. (Because we use the
        # sentence boundaries for the "next sentence prediction" task).
        # (2) Blank lines between documents. Document boundaries are needed so
        # that the "next sentence prediction" task doesn't span between documents.
        #
        # Example:
        # I am very happy.
        # Here is the second sentence.
        #
        # A new document.

        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]",
                    time.time() - start,
                )
            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.documents = [[]]
                with open(file_path, encoding="utf-8") as f:
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        line = line.strip()

                        # Empty lines are used as document delimiters
                        if not line and len(self.documents[-1]) != 0:
                            self.documents.append([])
                        tokens = tokenizer.tokenize(line)
                        tokens = tokenizer.convert_tokens_to_ids(tokens)
                        if tokens:
                            self.documents[-1].append(tokens)

                # Remove empty documents
                self.documents = [x for x in self.documents if x]

                logger.info(f"Creating examples from {len(self.documents)} documents.")
                self.examples = []
                for doc_index, document in enumerate(self.documents):
                    self.create_examples_from_document(document, doc_index, block_size)

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )

    def create_examples_from_document(
        self, document: List[List[int]], doc_index: int, block_size: int
    ):
        """Creates examples for a single document."""

        max_num_tokens = block_size - self.tokenizer.num_special_tokens_to_add(
            pair=True
        )

        # We *usually* want to fill up the entire sequence since we are padding
        # to `block_size` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pretraining and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `block_size` is a hard limit.
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_probability:
            target_seq_length = random.randint(2, max_num_tokens)

        current_chunk = []  # a buffer stored current working segments
        current_length = 0
        i = 0

        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []

                    if (
                        len(current_chunk) == 1
                        or random.random() < self.nsp_probability
                    ):
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)

                        # This should rarely go for more than one iteration for large
                        # corpora. However, just to be careful, we try to make sure that
                        # the random document is not the same as the document
                        # we're processing.
                        for _ in range(10):
                            random_document_index = random.randint(
                                0, len(self.documents) - 1
                            )
                            if random_document_index != doc_index:
                                break

                        random_document = self.documents[random_document_index]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break
                        # We didn't actually use these segments so we "put them back" so
                        # they don't go to waste.
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    # Actual next
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
                        """Truncates a pair of sequences to a maximum sequence length."""
                        while True:
                            total_length = len(tokens_a) + len(tokens_b)
                            if total_length <= max_num_tokens:
                                break
                            trunc_tokens = (
                                tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
                            )
                            if not (len(trunc_tokens) >= 1):
                                raise ValueError(
                                    "Sequence length to be truncated must be no less than one"
                                )
                            # We want to sometimes truncate from the front and sometimes from the
                            # back to add more randomness and avoid biases.
                            if random.random() < 0.5:
                                del trunc_tokens[0]
                            else:
                                trunc_tokens.pop()

                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                    if not (len(tokens_a) >= 1):
                        raise ValueError(
                            f"Length of sequence a is {len(tokens_a)} which must be no less than 1"
                        )
                    if not (len(tokens_b) >= 1):
                        raise ValueError(
                            f"Length of sequence b is {len(tokens_b)} which must be no less than 1"
                        )

                    # add special tokens
                    input_ids = self.tokenizer.build_inputs_with_special_tokens(
                        tokens_a, tokens_b
                    )

                    assert len(input_ids) <= block_size
                    # add token type ids, 0 for sentence a, 1 for sentence b
                    token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(
                        tokens_a, tokens_b
                    )

                    example = {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "token_type_ids": torch.tensor(
                            token_type_ids, dtype=torch.long
                        ),
                        "next_sentence_label": torch.tensor(
                            1 if is_random_next else 0, dtype=torch.long
                        ),
                    }

                    self.examples.append(example)

                current_chunk = []
                current_length = 0

            i += 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def write_examples_to_jsonl(
    segmented_file_path, output_file_path, model_name, block_size, batch_size
):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    examples = create_examples_from_documents(
        segmented_file_path, tokenizer, block_size=block_size, batch_size=batch_size
    )
    with open(output_file_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    logger.info(f"{len(examples)} have been writed!")


def create_examples_from_documents(
    segmented_file_path, tokenizer, block_size, batch_size
):

    dataset = TextDatasetForNextSentencePrediction(
        tokenizer=tokenizer,
        file_path=segmented_file_path,
        block_size=block_size,
        overwrite_cache=True,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

    examples = []
    for step, batch in enumerate(data_loader):

        for idx in range(len(batch)):
            input_ids = batch["input_ids"][idx]
            next_sentence_label = batch["next_sentence_label"][idx]
            labels = batch["labels"][idx]

            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            try:
                first_pad = tokens.index("[PAD]")
                tokens = tokens[:first_pad]
            except:
                # have no [PAD] in tokens
                ...
            labels = tokenizer.convert_ids_to_tokens(labels)
            labels_dict = dict(
                filter(lambda x: x[1] != "[UNK]", list(zip(range(len(labels)), labels)))
            )

            mask_positions = list(labels_dict.keys())
            mask_tokens = list(labels_dict.values())
            next_sentence_label = int(next_sentence_label)

            example = {
                "tokens": tokens,
                "mask_positions": mask_positions,
                "mask_tokens": mask_tokens,
                "next_sentence_label": next_sentence_label,
                # "sentences_length": len(tokens),
                # "masked_num": len(mask_positions),
            }
            examples.append(example)

    return examples


def main(args):
    set_seed(42)

    write_examples_to_jsonl(
        args.segmented_file_path,
        args.output_file_path,
        args.model_name,
        args.block_size,
        args.batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmented_file_path", type=str, required=True)
    parser.add_argument("--output_file_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)
