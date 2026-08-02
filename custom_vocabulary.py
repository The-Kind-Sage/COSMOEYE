import json
import sys
import torch


class CustomVocabulary:
    """Minimal, dependency-free tokenizer/vocabulary for text sequences.

    Built with only PyTorch + the Python standard library. Handles both
    Latin (English) and Devanagari (Nepali) scripts natively, since Python
    strings are Unicode and whitespace tokenization works for both.
    """

    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"

    def __init__(self):
        # Special tokens occupy the fixed head of the vocabulary:
        # 0 = <PAD>, 1 = <SOS>, 2 = <EOS>, 3 = <UNK>
        self.token2id = {
            self.PAD: 0,
            self.SOS: 1,
            self.EOS: 2,
            self.UNK: 3,
        }
        self.id2token = {idx: tok for tok, idx in self.token2id.items()}

    def __len__(self):
        return len(self.token2id)

    @staticmethod
    def tokenize(sentence):
        """Whitespace tokenization (works for English and Devanagari)."""
        return sentence.strip().split()

    def build_vocab(self, sentences):
        """Add every unique token from a list of sentences to the vocabulary.

        New tokens are appended after the special tokens, so indices 0-3
        stay reserved. Both English and Devanagari tokens are supported
        without any extra configuration.
        """
        for sentence in sentences:
            for token in self.tokenize(sentence):
                if token not in self.token2id:
                    new_id = len(self.token2id)
                    self.token2id[token] = new_id
                    self.id2token[new_id] = token
        return self

    def numericalize(self, sentence, with_special=True):
        """Map a sentence to a list of integer IDs.

        with_special=True wraps the sequence as <SOS> ... <EOS>.
        Out-of-vocabulary tokens become the <UNK> id (3).
        """
        ids = [self.token2id.get(tok, self.token2id[self.UNK])
               for tok in self.tokenize(sentence)]
        if with_special:
            ids = [self.token2id[self.SOS]] + ids + [self.token2id[self.EOS]]
        return ids

    def to_tensor(self, sentences, max_len=None, device="cpu"):
        """Batch sentences into a padded (B, T) LongTensor of token IDs.

        Sequences are right-padded with <PAD> to max_len (or the longest
        sequence in the batch if max_len is None).
        """
        ids_list = [self.numericalize(s) for s in sentences]
        if max_len is None:
            max_len = max(len(ids) for ids in ids_list)
        batch = torch.full((len(ids_list), max_len),
                           self.token2id[self.PAD], dtype=torch.long)
        for row, ids in enumerate(ids_list):
            batch[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        return batch.to(device)

    def detokenize(self, ids):
        """Map integer IDs back to tokens (excludes <PAD>/<EOS>/<SOS>)."""
        return [self.id2token.get(i, self.UNK)
                for i in ids if i not in (self.token2id[self.PAD],
                                          self.token2id[self.EOS],
                                          self.token2id[self.SOS])]

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"token2id": self.token2id, "id2token": self.id2token}, f,
                      ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls()
        vocab.token2id = {str(k): int(v) for k, v in data["token2id"].items()}
        vocab.id2token = {int(k): v for k, v in data["id2token"].items()}
        return vocab


# --- Text database: bilingual (English + Nepali) landslide corpus ------------

TEXT_DATABASE = [
    # English
    "landslide blocked the highway near the village",
    "satellite imagery detected a new landslide in the mountains",
    "rescue teams reached the affected district today",
    "heavy rainfall triggered multiple landslides overnight",
    "the road is closed due to a major landslide",
    # Devanagari (Nepali)
    "पहिरोले गाउँ नजिकको राजमार्ग बन्द गर्‍यो",
    "उपग्रह तस्बिरले पहाडमा नयाँ पहिरो पत्ता लगायो",
    "उद्धार टोली आज प्रभावित जिल्लामा पुगेको छ",
    "भारी वर्षाले रातभर धेरै पहिरो निम्त्यायो",
    "ठूलो पहिरोका कारण सडक बन्द छ",
]

if __name__ == "__main__":
    # Windows consoles default to cp1252 and cannot print Devanagari
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    vocab = CustomVocabulary().build_vocab(TEXT_DATABASE)

    print(f"Vocabulary size: {len(vocab)} tokens (4 special + {len(vocab) - 4} unique)")
    print(f"Special tokens: {vocab.token2id}\n")

    samples = ["landslide blocked the road", "पहिरोले सडक बन्द गर्‍यो", "unknownwordxyz पहिरो"]
    for sample in samples:
        ids = vocab.numericalize(sample)
        print(f"IN : {sample}")
        print(f"IDS: {ids}")
        print(f"OUT: {' '.join(vocab.detokenize(ids))}\n")

    batch = vocab.to_tensor(samples, max_len=12)
    print("Padded batch tensor:")
    print(batch)

    vocab.save("vocabulary.json")
    reloaded = CustomVocabulary.load("vocabulary.json")
    print(f"\nSaved/loaded vocabulary size: {len(reloaded)} (round-trip OK)")
