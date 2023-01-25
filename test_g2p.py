from g2p_id import G2P

g2p = G2P()


def test_g2p_vocal():
    # Alofon [e]
    assert g2p("serong") == "ˈseroŋ"
    assert g2p("sore") == "ˈsore"
    assert g2p("kare") == "ˈkare"

    # Alofon [ɛ]
    assert g2p("teh") == "tɛh"
    assert g2p("pek") == "pɛʔ"
    assert g2p("bebek") == "ˈbɛbɛʔ"

    # Alofon [ə]
    assert g2p("tante") == "ˈtantə"
    assert g2p("enam") == "əˈnam"
    assert g2p("emas") == "əˈmas"

    # Alofon [o]
    assert g2p("toko") == "ˈtoko"
    assert g2p("roda") == "ˈroda"
    assert g2p("sekolah") == "səˈkolah"

    # Alofon [ɔ]
    assert g2p("rokok") == "ˈrɔkɔʔ"
    assert g2p("pojok") == "ˈpɔdʒɔʔ"
    assert g2p("momok") == "ˈmɔmɔʔ"
    assert g2p("pohon") == "ˈpɔhɔn"
    # assert g2p("positif") == "pɔˈsitɪf"

    # Alofon [i]
    assert g2p("gigi") == "ˈɡiɡi"
    assert g2p("tali") == "ˈtali"
    assert g2p("ini") == "ˈini"
    assert g2p("bila") == "ˈbila"
    assert g2p("simpang") == "ˈsimpaŋ"
    assert g2p("periksa") == "pəˈriʔsa"

    # Alofon [ɪ]
    assert g2p("banting") == "ˈbantɪŋ"
    assert g2p("salin") == "ˈsalɪn"
    assert g2p("parit") == "ˈparɪt"
    assert g2p("pilih") == "ˈpilɪh"
    assert g2p("yakin") == "ˈjakɪn"
    assert g2p("kirim") == "ˈkirɪm"

    # Alofon [u]
    assert g2p("upah") == "ˈupah"
    assert g2p("tukang") == "ˈtukaŋ"
    assert g2p("bantu") == "ˈbantu"

    assert g2p("kumbang") == "ˈkumbaŋ"
    assert g2p("tunggu") == "ˈtuŋɡu"
    assert g2p("bundel") == "ˈbundəl"

    # Alofon [ʊ]
    assert g2p("warung") == "ˈwarʊŋ"
    assert g2p("dusta") == "ˈdʊsta"
    assert g2p("pulsa") == "ˈpʊlsa"


if __name__ == "__main__":
    test_g2p_vocal()
