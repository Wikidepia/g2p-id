# Indonesian Grapheme-to-Phoneme

This module is designed to convert Indonesian graphemes (spelling) into phonemes (pronunciation). Fortunately, most Indonesian word pronunciations can be inferred from their spelling. Most of the work needed to convert grapheme to phoneme are: finding glottal stop `/ʔ/` (baʔso) and determining which 'e' to use, there are two: `/e/` (pensil) and `/ə/` (têman). (there might be more... *shrug*)

<!-- Big thanks to Wilson Wongso for sharing about [Predicting Phonemes with BERT](https://w11wo.github.io/posts/2022/04/predicting-phonemes-with-bert/). I used his code to implement the predictor used in this module. -->

## Installation

```bash
pip install git+https://github.com/Wikidepia/g2p-id
```

## Example usage

```python
from g2p_id import G2P

g2p = G2P()
phonemes, syllables = g2p.to_phoneme("Tak seorang pun boleh ditangkap, ditahan atau dibuang dengan sewenang-wenang.")

print(phonemes) # taʔ seoraŋ pun boleh ditaŋkap, ditahan ataʊ dibuaŋ deŋan sewenaŋ-wenaŋ.
print(syllables) # ['taʔ', ' ', 'se', 'o', 'raŋ', ' ', 'pun', ..., 'we', 'naŋ', ' ']
```

## References

- [Variasi Bunyi Vokal - Narabahasa](https://narabahasa.id/linguistik-umum/fonologi/variasi-bunyi-vokal)
<!-- - [Predicting Phonemes with BERT - Wilson Wongso](https://w11wo.github.io/posts/2022/04/predicting-phonemes-with-bert/) -->
- Moeliono, Anton M., dkk. 2017. Tata Bahasa Baku Bahasa Indonesia Edisi Keempat. Jakarta: Badan Pengembangan dan Pembinaan Bahasa.
- [Google Cloud TTS Supported phonemes and levels of stress](https://docs.cloud.google.com/text-to-speech/docs/phonemes#phonemes_12)

## TODO

- [ ] Add test cases
- [ ] Add BERT based homograph (resolver?)
- [ ] Proper versioning
