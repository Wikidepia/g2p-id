# Indonesian Grapheme-to-Phoneme

This module is designed to convert Indonesian grapheme (spelling) to phonemes (pronunciation). Luckily, most Indonesian word pronunciations can be inferred from their spelling. However, there are some exceptions. For example, there are two different ways to pronounce "e":

1. "è" as in "lelet",
2. "ê" as in "enam".

This module follow [Google's TTS Phonemes](https://cloud.google.com/text-to-speech/docs/phonemes#indonesian_indonesia_id-id) for mapping grapheme to phoneme.

Big thanks to Wilson Wongso for sharing about [Predicting Phonemes with BERT](https://w11wo.github.io/posts/2022/04/predicting-phonemes-with-bert/). His post helps me to get a better model for predicting "e" pronunciation.

## Installation

`pip3 install -U git+https://github.com/Wikidepia/g2p-id`

## Example usage

```python
from g2p_id import G2P

g2p = G2P()
g2p("Rumah Agus terbakar.") # rumah agus tərbakar.
```

## References

- [Variasi Bunyi Vokal - Narabahasa](https://narabahasa.id/linguistik-umum/fonologi/variasi-bunyi-vokal)
- [Predicting Phonemes with BERT - Wilson Wongso](https://w11wo.github.io/posts/2022/04/predicting-phonemes-with-bert/)

## TODO

- [ ] Add test cases
- [ ] Better model for predicting "e"
- [ ] Handle homograph
