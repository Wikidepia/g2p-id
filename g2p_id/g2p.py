import json
import os
import re

import numpy as np
import onnxruntime
from nltk.tokenize import TweetTokenizer

from .syllable_splitter import SyllableSplitter

word_tokenize = TweetTokenizer().tokenize
dirname = os.path.dirname(__file__)


class Predictor:
    def __init__(self, model_path):
        # fmt: off
        self.text_vocab = ["[PAD]", "[START]", "[END]", 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        self.phoneme_vocab = ["[PAD]", "[START]", "[END]", 'a', 'b', 'c', 'd', 'ê', 'è', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        # fmt: on
        self.session = onnxruntime.InferenceSession(model_path)

    def predict(self, word):
        text = [1] + [self.text_vocab.index(c) for c in word] + [2]
        text.extend([0] * (28 - len(text)))  # Pad to 28 tokens
        indices, _ = self.session.run(None, {"text": [text]})

        output = indices[0]
        prediction = output[1 : len(word)].tolist()
        return "".join([self.phoneme_vocab[i] for i in prediction])


class G2P:
    def __init__(self):
        dict_path = os.path.join(dirname, "data/dict.json")
        with open(dict_path) as f:
            self.dict = json.load(f)

        map_path = os.path.join(dirname, "data/map.json")
        with open(map_path) as f:
            self.map = json.load(f)

        model_path = os.path.join(dirname, "tiny-pron.onnx")
        self.predictor = Predictor(model_path)

        self.syllable_splitter = SyllableSplitter()

    def __call__(self, text):
        text = text.lower()
        text = re.sub(r"[^ a-z'\.,?!-]", "", text)

        prons = []
        words = word_tokenize(text)
        for word in words:
            # PUEBI pronunciation
            if word in self.dict:
                pron = self.dict[word]
            elif "e" not in word or not word.isalpha():
                pron = word
            elif "e" in word:
                pron = self.predictor.predict(word)

            # [ALOFON] o or ô
            if "o" in word:
                sylls = self.syllable_splitter.split_syllables(pron)
                for i, syll in enumerate(sylls):
                    if "o" in syll:
                        if syll[-1] != "o":  # Posisi tertutup
                            sylls[i] = syll.replace("o", "ô")
                pron = "".join(sylls)

            # "IPA" pronunciation
            if pron.startswith("x"):
                pron = re.sub(r"^x", "s", pron)
            elif pron.endswith("k"):
                pron = re.sub(r"k$", "'", pron)

            if word.isalpha():
                for char in self.map:
                    pron = pron.replace(char, self.map[char])

            prons.append(pron)
            prons.append(" ")

        return "".join(prons)
