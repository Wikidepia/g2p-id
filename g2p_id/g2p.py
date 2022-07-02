import json
import os
import re

import numpy as np
import onnxruntime
from nltk.tokenize import TweetTokenizer
from sacremoses import MosesDetokenizer

from .syllable_splitter import SyllableSplitter

ABJAD_MAPPING = {
    "a": "a",
    "b": "bé",
    "c": "cé",
    "d": "dé",
    "e": "é",
    "f": "èf",
    "g": "gé",
    "h": "ha",
    "i": "i",
    "j": "jé",
    "k": "ka",
    "l": "èl",
    "m": "èm",
    "n": "èn",
    "o": "o",
    "p": "pé",
    "q": "ki",
    "r": "èr",
    "s": "ès",
    "t": "té",
    "u": "u",
    "v": "vé",
    "w": "wé",
    "x": "èks",
    "y": "yé",
    "z": "zèt",
}

PHONETIC_MAPPING = {
    "sy": "ʃ",
    "ny": "ɲ",
    "ng": "ŋ",
    "dj": "dʒ",
    "'": "ʔ",
    "c": "tʃ",
    "é": "e",
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
        self.vocab = ['', '[UNK]', 'a', 'i', 'ê', 'e', 'n', 'r', 's', 't', 'o', 'k', 'l', 'm', 'g', 'p', 'u', 'd', 'b', 'f', 'h', 'c', 'j', 'v', 'w', 'y', 'z', 'x', 'q', '[mask]']
        self.mask_token_id = self.vocab.index("[mask]")
        # fmt: on
        self.session = onnxruntime.InferenceSession(model_path)

    def predict(self, word: str) -> str:
        """
        Predict the phonetic representation of a word.

        Args:
            word (str): The word to predict.

        Returns:
            str: The predicted phonetic representation of the word.
        """
        text = [self.vocab.index(c) if c != "e" else self.mask_token_id for c in word]
        text.extend([0] * (32 - len(text)))  # Pad to 32 tokens
        inputs = np.array([text])
        (predictions,) = self.session.run(None, {"input_4": inputs})

        # find masked idx token
        _, masked_index = np.where(inputs == self.mask_token_id)

        # get prediction at those masked index only
        mask_prediction = predictions[0][masked_index]
        predicted_ids = np.argmax(mask_prediction, axis=1)

        # replace mask with predicted token
        for i, idx in enumerate(masked_index):
            text[idx] = predicted_ids[i]

        return "".join([self.vocab[i] for i in text if i != 0])


class G2P:
    def __init__(self):
        self.tokenizer = TweetTokenizer()
        self.detokenizer = MosesDetokenizer(lang="id")

        dict_path = os.path.join(dirname, "data/dict.json")
        with open(dict_path) as f:
            self.dict = json.load(f)

        model_path = os.path.join(dirname, "model/bert_pron.onnx")
        self.predictor = Predictor(model_path)

        self.syllable_splitter = SyllableSplitter()

    def __call__(self, text: str) -> str:
        """
        Convert text to phonetic representation.

        Args:
            text (str): The text to convert.

        Returns:
            str: The phonetic representation of the text.
        """
        text = text.lower()
        text = re.sub(r"[^ a-z0-9'\.,?!-]", "", text)
        text = text.replace("-", " ")

        prons = []
        words = self.tokenizer.tokenize(text)
        for word in words:
            # PUEBI pronunciation
            if word in self.dict:
                pron = self.dict[word]
            elif len(word) == 1 and word in ABJAD_MAPPING:
                pron = ABJAD_MAPPING[word]
            elif "e" not in word or not word.isalpha():
                pron = word
            elif "e" in word:
                pron = self.predictor.predict(word)

            # [ALOFON] o or ô (vokal /o/)
            # [ALOFON] è or é (vokal /e/)
            # [HOMOFON] nk => ng
            if any(c in word for c in ["o", "e", "nk"]):
                sylls = self.syllable_splitter.split_syllables(pron)
                alofon_o, alofon_e = "o", "é"
                for i, syll in enumerate(sylls):
                    if "o" in syll and not syll.endswith("o"):
                        alofon_o = "ô"  # Tertutup
                    if "e" in syll and not syll.endswith("e"):
                        alofon_e = "è"  # Tertutup
                    if syll.endswith("nk"):
                        sylls[i] = syll[:-2] + "ng"
                    # Syllable stress
                    # Before last syllable
                    if i == len(sylls) - 2:
                        sylls[i] = "ˈ" + sylls[i]
                pron = "".join(sylls)
                # Apply alofon changes
                pron = pron.replace("o", alofon_o)
                pron = pron.replace("e", alofon_e)

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
