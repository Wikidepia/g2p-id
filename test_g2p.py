from g2p_id import G2P

g2p = G2P()


def test_g2p_vocal():
    # Alofon [e]
    assert g2p.to_phoneme("serong")[0] == "seroŋ"
    assert g2p.to_phoneme("sore")[0] == "sore"
    assert g2p.to_phoneme("kare")[0] == "kare"

    # Alofon [ɛ]
    assert g2p.to_phoneme("teh")[0] == "teh"
    assert g2p.to_phoneme("pek")[0] == "peʔ"
    assert g2p.to_phoneme("bebek")[0] == "bəbəʔ" # this is not bebek animal..

    # Alofon [ə]
    assert g2p.to_phoneme("tante")[0] == "tantə"
    assert g2p.to_phoneme("enam")[0] == "ənam"
    assert g2p.to_phoneme("emas")[0] == "əmas"

    # Alofon [o]
    assert g2p.to_phoneme("toko")[0] == "toko"
    assert g2p.to_phoneme("roda")[0] == "roda"
    assert g2p.to_phoneme("sekolah")[0] == "səkolah"

    # Alofon [o]
    assert g2p.to_phoneme("rokok")[0] == "rokoʔ"
    assert g2p.to_phoneme("pojok")[0] == "podʒoʔ"
    assert g2p.to_phoneme("momok")[0] == "momoʔ"
    assert g2p.to_phoneme("pohon")[0] == "pohon"
    # assert g2p.to_phoneme("positif")[0] == "positif"

    # Alofon [i]
    assert g2p.to_phoneme("gigi")[0] == "gigi"
    assert g2p.to_phoneme("tali")[0] == "tali"
    assert g2p.to_phoneme("ini")[0] == "ini"
    assert g2p.to_phoneme("bila")[0] == "bila"
    assert g2p.to_phoneme("simpang")[0] == "simpaŋ"
    assert g2p.to_phoneme("periksa")[0] == "pəriʔsa"

    # Alofon [i]
    assert g2p.to_phoneme("banting")[0] == "bantiŋ"
    assert g2p.to_phoneme("salin")[0] == "salin"
    assert g2p.to_phoneme("parit")[0] == "parit"
    assert g2p.to_phoneme("pilih")[0] == "pilih"
    # assert g2p.to_phoneme("yakin")[0] == "jakin" # TODO: fix, currently returns 'yakin'
    assert g2p.to_phoneme("kirim")[0] == "kirim"

    # Alofon [u]
    assert g2p.to_phoneme("upah")[0] == "upah"
    assert g2p.to_phoneme("tukang")[0] == "tukaŋ"
    assert g2p.to_phoneme("bantu")[0] == "bantu"

    assert g2p.to_phoneme("kumbang")[0] == "kumbaŋ"
    assert g2p.to_phoneme("tunggu")[0] == "tuŋgu"
    assert g2p.to_phoneme("bundel")[0] == "bundəl"

    # Alofon [ʊ]
    assert g2p.to_phoneme("warung")[0] == "waruŋ"


if __name__ == "__main__":
    test_g2p_vocal()
