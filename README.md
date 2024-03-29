# Indonesian Grapheme-to-Phoneme

This module is designed to convert Indonesian graphemes (spelling) into phonemes (pronunciation). Fortunately, most Indonesian word pronunciations can be inferred from their spelling.

Big thanks to Wilson Wongso for sharing about [Predicting Phonemes with BERT](https://w11wo.github.io/posts/2022/04/predicting-phonemes-with-bert/). I used his code to implement the predictor used in this module.

## Installation

```bash
pip install git+https://github.com/Wikidepia/g2p-id
```

## Example usage

```python
from g2p_id import G2P

g2p = G2P()
g2p("Rumah Agus terbakar.") # ˈrumah ˈaɡʊs tərˈbakar.
```

## References

- [Variasi Bunyi Vokal - Narabahasa](https://narabahasa.id/linguistik-umum/fonologi/variasi-bunyi-vokal)
- [Predicting Phonemes with BERT - Wilson Wongso](https://w11wo.github.io/posts/2022/04/predicting-phonemes-with-bert/)
- Moeliono, Anton M., dkk. 2017. Tata Bahasa Baku Bahasa Indonesia Edisi Keempat. Jakarta: Badan Pengembangan dan Pembinaan Bahasa.

## TODO

- [x] Add test cases
- [ ] Better model for predicting "e"
- [ ] Handle heteronym
