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
            elif "e" not in word or re.search("[a-z]", word) is None:
                pron = word

            # Get "IPA" pronunciation
            if pron[0] == "x":
                pron[0] = "s"
            elif pron[-1] == "k":
                pron[0] = "'"

            if re.search("[a-z]", word) is not None:
                for char in self.map:
                    pron = pron.replace(char, self.map[char])

            prons.append(pron)
            prons.append(" ")

        return "".join(prons)
