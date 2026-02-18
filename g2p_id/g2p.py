import csv
import re
import os

import pycrfsuite

dirname = os.path.dirname(__file__)

class G2P:
    def __init__(self):
        self.tagger = pycrfsuite.Tagger() # type: ignore
        self.tagger.open(os.path.join(dirname, "model/syllabifier.crfsuite"))
        self.schwa_dict = self.init_schwa_dict()

        self.vokal = set(["a", "i", "u", "e", "o"])
        self.abjad_mapping = {
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
        self.kv_pattern = [
            "VK",
            "KV",
            "KVK",
            "VKK",
            "KKV",
            "KKVK",
            "KVKK",
            "KKKV",
            "KKKVK",
            "KKVKK",
            "KVKKK",
        ]
        # glotal matching similar to bookbot's g2p
        # [a: at this point, it seems that most of this project code is the same lmao]
        self.glotal_regex = re.compile(r"[aiueəo]k[bcdfghjklmnpqrstvwxyz]")

    def init_schwa_dict(self):
        schwa = {}
        with open(os.path.join(dirname, "data/schwa_dict.csv")) as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                schwa[row[0]] = row[1]
        return schwa

    def kv_from_word(self, word):
        return "".join("V" if c in self.vokal else "K" for c in word)

    def replace_ranges(self, s, replacements):
        replacements.sort(key=lambda x: x[0])
        result = []
        last = 0

        for start, end, new in replacements:
            result.append(s[last:start])
            result.append(new)
            last = end

        result.append(s[last:])
        return "".join(result)

    def generate_crfsyll_feat(self, word, i):
        char = word[i]
        features = [
            "bias",
            f"c={char}",
        ]

        if i > 0:
            features.append(f"c[-1:0]={word[i-1]}{word[i]}")
        if i > 1:
            features.append(f"c[-2:0]={word[i-2]}{word[i-1]}{word[i]}")

        if i < len(word) - 1:
            features.append(f"c[0:+1]={word[i]}{word[i+1]}")
        if i < len(word) - 2:
            features.append(f"c[0:+2]={word[i]}{word[i+1]}{word[i+2]}")

        for n in range(1, min(6, i + 1)):
            features.append(f"c[-{n}]={word[i-n][0]}")

        for n in range(1, min(6, len(word) - i)):
            features.append(f"c[+{n}]={word[i+n][0]}")

        if i == 0:
            features.append("BOS")
        if i == len(word) - 1:
            features.append("EOS")
        return features

    def to_syllables(self, word) -> list:
        syllables_ret = [""]
        upper_word = word.replace("é", "e").upper()
        word_feat = [
            self.generate_crfsyll_feat(upper_word, i) for i in range(len(upper_word))
        ]
        for char, tag in zip(word, self.tagger.tag(word_feat)):
            if tag == "O":
                syllables_ret[-1] += char
            elif tag == "S":
                syllables_ret[-1] += char
                syllables_ret.append("")
            else:
                raise ValueError("new char?")
        return syllables_ret

    def to_phoneme(self, text, split_abbr=False):
        text_syllable = []
        text = text.lower()

        # simple phoneme mapping for double consonant
        text = text.replace("x", "ks").replace("c", "tʃ").replace("j", "dʒ")
        text = text.replace("ng", "ŋ").replace("ny", "ɲ").replace("sy", "ʃ")
        text = text.replace("kh", "x").replace("v", "f")

        repls = []
        for match_re in re.finditer(r"[a-zŋɲʃʒ]+", text):
            word = match_re.group()
            # `split_abbr` will expand abbr
            is_abbr = False
            if split_abbr:
                word_kv = self.kv_from_word(word)
                is_abbr = not any(p in word_kv for p in self.kv_pattern)
                if is_abbr:
                    word = "".join(self.abjad_mapping[c] for c in word)

            # replace word from schwa dictionary
            if "e" in word and word in self.schwa_dict:
                word = self.schwa_dict[word]
            word = word.replace("ê", "ə")

            # check for glotal stop /k/
            if word.endswith("k"):
                word = word[:-1] + "ʔ"
            for glot in list(self.glotal_regex.finditer(word)):
                word = word[:glot.start()+1] + "ʔ" + word[glot.end()-1:]

            new_word = word
            syllables = self.to_syllables(word)
            new_syllables = syllables

            # check for diftong
            if any(k in word for k in ["ai", "au", "oi"]) and not is_abbr:
                new_word, new_syllables = "", []
                for s in syllables:
                    # check for diftong
                    for d in ["ai", "au", "oi"]:
                        s = s.replace("ai", "aɪ")
                        s = s.replace("au", "aʊ")
                        s = s.replace("oi", "ɔɪ")

                    new_word += s
                    new_syllables.append(s)

            repls.append((match_re.start(), match_re.end(), new_word))
            text_syllable.extend(new_syllables)
            text_syllable.append(" ")  # add space after new word
        return self.replace_ranges(text, repls), text_syllable

    def to_grapheme(self, text):
        text = text.replace("tʃ", "c").replace("dʒ", "j").replace("ŋ", "ng")
        text = text.replace("ny", "ɲ").replace("ʃ", "sy").replace("x", "kh")
        return text
