import json
import os
import re

import numpy as np
import onnxruntime
from sacremoses import MosesDetokenizer, MosesTokenizer

from .syllable_splitter import SyllableSplitter

PHONETIC_MAPPING = {
    "sy": "ʃ",
    "ny": "ɲ",
    "ng": "ŋ",
    "dj": "dʒ",
    "'": "ʔ",
    "c": "tʃ",
    "è": "ɛ",
    "ê": "ə",
    "j": "dʒ",
    "ô": "ɔ",
    "q": "k",
    "v": "f",
    "x": "ks",
    "y": "j",
}

dirname = os.path.dirname(__file__)

# Predict pronounciation with BERT Masking
# Read more: https://w11wo.github.io/posts/2022/04/predicting-phonemes-with-bert/
class Predictor:
    def __init__(self, model_path):
        # fmt: off
        self.vocab = ['', '[UNK]', 'a', 'è', 'i', 'ê', 'n', 'r', 's', 't', 'o', 'k', 'l', 'm', 'g', 'p', 'u', 'd', 'b', 'f', 'h', 'c', 'j', 'v', 'w', 'y', '-', 'z', '.', 'x', 'q', '[mask]']
        self.mask_token_id = self.vocab.index("[mask]")
        # fmt: on
        self.session = onnxruntime.InferenceSession(model_path)

    def predict(self, word):
        text = [self.vocab.index(c) if c != "e" else self.mask_token_id for c in word]
        text.extend([0] * (32 - len(text)))  # Pad to 32 tokens
        input_1 = np.array([text])
        (predictions,) = self.session.run(None, {"input_1": input_1})

        # find masked idx token
        _, masked_index = np.where(input_1 == self.mask_token_id)

        # get prediction at those masked index only
        mask_prediction = predictions[0][masked_index]
        predicted_ids = np.argmax(mask_prediction, axis=1)

        # replace mask with predicted token
        for i, idx in enumerate(masked_index):
            text[idx] = predicted_ids[i]

        return "".join([self.vocab[i] for i in text if i != 0])


class G2P:
    def __init__(self):
        self.tokenizer = MosesTokenizer(lang="id")
        self.detokenizer = MosesDetokenizer(lang="id")
        self.protected_patterns = [r"\'"]

        dict_path = os.path.join(dirname, "data/dict.json")
        with open(dict_path) as f:
            self.dict = json.load(f)

        model_path = os.path.join(dirname, "bert_pron.onnx")
        self.predictor = Predictor(model_path)

        self.syllable_splitter = SyllableSplitter()

    def __call__(self, text: str):
        text = text.upper()
        text = re.sub(r"[^ A-Z0-9'\.,?!-]", "", text)
        text = text.replace("-", " ")

        prons = []
        words = self.tokenizer.tokenize(
            text, protected_patterns=self.protected_patterns, escape=False
        )
        for word in words:
            word = word.lower()
            # PUEBI pronunciation
            if word in self.dict:
                pron = self.dict[word]
            elif "e" not in word or not word.isalpha():
                pron = word
            elif "e" in word:
                pron = self.predictor.predict(word)

            # [ALOFON] o or ô
            # [HOMOFON] nk => ng
            if "o" in word or "nk" in word:
                sylls = self.syllable_splitter.split_syllables(pron)
                for i, syll in enumerate(sylls):
                    # [ALOFON] o or ô
                    if syll[-1] != "o":  # Posisi tertutup
                        sylls[i] = syll.replace("o", "ô")
                    # [HOMOFON] nk => ng
                    if syll.endswith("nk"):
                        sylls[i] = syll[:-2] + "ng"
                pron = "".join(sylls)

            # "IPA" pronunciation
            if pron.startswith("x"):
                pron = re.sub(r"^x", "s", pron)
            if pron.endswith("k"):
                pron = re.sub(r"k$", "'", pron)

            # Apply phonetic mapping
            for g, p in PHONETIC_MAPPING.items():
                pron = pron.replace(g, p)
            pron = pron.replace("kh", "x")

            prons.append(pron)
            prons.append(" ")

        return self.detokenizer.detokenize(prons)
