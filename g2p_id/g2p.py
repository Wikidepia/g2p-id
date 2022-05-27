import json
import os
import re

from nltk.tokenize import TweetTokenizer

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

    def __call__(self, text):
        text = text.lower()
        text = re.sub("[^ a-z'.,?!\-]", "", text)

        prons = []
        words = word_tokenize(text)
        for word in words:
            # Get PUEBI pronunciation
            if word in self.dict:
                pron = self.dict[word]
            elif "e" not in word or not word.isalpha():
                pron = word
            elif "e" in word: # TODO: handle oov word
                pron = word.replace("e", "ê")

            # Get "IPA" pronunciation
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
