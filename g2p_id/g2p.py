import json
import os
import re

import torch
from dp.phonemizer import Phonemizer
from nltk.tokenize import TweetTokenizer

from .syllable_splitter import SyllableSplitter

word_tokenize = TweetTokenizer().tokenize
dirname = os.path.dirname(__file__)


class G2P:
    def __init__(self):
        dict_path = os.path.join(dirname, "data/dict.json")
        with open(dict_path) as f:
            self.dict = json.load(f)

        map_path = os.path.join(dirname, "data/map.json")
        with open(map_path) as f:
            self.map = json.load(f)

        model_path = os.path.join(dirname, "id_kbbi_autoreg.pt")
        self.phonemizer = Phonemizer.from_checkpoint(model_path)
        model = self.phonemizer.predictor.model
        self.phonemizer.predictor.model = torch.jit.script(model)

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
                pron = self.phonemizer(word, lang="id")

            # [ALOFON] o or ô
            if "o" in word:
                sylls = self.syllable_splitter.split_syllables("onde")
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
