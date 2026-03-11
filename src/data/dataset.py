import torch
from torch.utils.data import Dataset
import pandas as pd
import ast
import logging

log = logging.getLogger(__name__)

class BoundaryDataset(Dataset):
    def __init__(self, path, tokenizer, split=None, ratio=1.0, max_length=512):
        self.data_path = path
        self.tokenizer = tokenizer
        self.max_length = max_length

        log.info(f"📂 Caricamento dataset da: {self.data_path}")

        self.data = self.prepare_data(path, split, ratio)
        log.info(f"✅ Esempi caricati: {len(self.data)}")
        
        

    def __len__(self):
        return len(self.data)

    def _safe_literal_eval(self, value, default):
        if isinstance(value, (list, dict, tuple)):
            return value
        if pd.isna(value):
            return default
        try:
            return ast.literal_eval(str(value))
        except Exception as e:
            log.warning(f"Parsing fallito per valore={value!r}: {e}")
            return default

    def _normalize_text(self, text):
        return str(text).strip()

    def _build_char_spans_from_chunks(self, text, chunks):
        spans = []
        cursor = 0
        text_len = len(text)

        for chunk in chunks:
            chunk_text = self._normalize_text(chunk.get("text", ""))
            chunk_label = int(chunk.get("label", 0))

            if not chunk_text:
                continue

            while cursor < text_len and text[cursor].isspace():
                cursor += 1

            start = cursor
            end = start + len(chunk_text)

            if text[start:end] != chunk_text:
                found = text.find(chunk_text, cursor)
                if found == -1:
                    log.warning(
                        "Chunk non trovato nel testo finale. "
                        f"Chunk ignorato: {chunk_text[:80]!r}"
                    )
                    continue
                start = found
                end = start + len(chunk_text)

            spans.append((start, end, chunk_label))
            cursor = end

        return spans

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        text = str(row["text"]) if "text" in row else str(row["hybrid_text"])
        chunks = self._safe_literal_eval(row.get("chunks", "[]"), default=[])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offset_mapping = encoding["offset_mapping"].squeeze(0).tolist()

        labels = torch.zeros(self.max_length, dtype=torch.long)

        chunk_spans = self._build_char_spans_from_chunks(text, chunks)

        for token_idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == 0 and tok_end == 0:
                continue

            assigned = False
            for span_start, span_end, span_label in chunk_spans:
                if tok_start < span_end and tok_end > span_start:
                    labels[token_idx] = span_label
                    assigned = True
                    break

            if not assigned:
                labels[token_idx] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def prepare_data(self, data_path, split, ratio):
        df = pd.read_csv(data_path)
        if split is None:
            return df
        else:
            assert split in ['train', 'val', 'test']
            return df[df['train_ix'] == split].sample(frac=ratio, random_state=42, ignore_index=True)