# Indonesian Grapheme-to-Phoneme

This module is designed to convert Indonesian grapheme (spelling) to phonemes (pronunciation). Luckily, most Indonesian word pronunciations can be inferred from their spelling. However, there are some exceptions. For example, there are two(?) different ways to pronounce "e": "è" as in "lelet" and "ê" as in "enam".

We follow [Google's TTS Phonemes](https://cloud.google.com/text-to-speech/docs/phonemes#indonesian_indonesia_id-id) for mapping grapheme to phoneme.

Big thanks to Wilson Wongso for sharing about [Predicting Phonemes with BERT](https://w11wo.github.io/posts/2022/04/predicting-phonemes-with-bert/). His post helps me to get a better model for predicting "e" pronunciation.

## Installation

`pip3 install -U git+https://github.com/Wikidepia/g2p-id`

## TODO

- [ ] Add test cases
- [ ] Better model for predicting "e"
- [ ] Handle homograph
