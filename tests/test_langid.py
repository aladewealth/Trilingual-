from trilingua.langid import heuristic_langid


def test_heuristic_langid():
    assert heuristic_langid("The cat is sleeping on the sofa and it was very happy with the warm sun.")[0] == "en"
    assert heuristic_langid("El gato duerme en el sofá y está muy contento con el sol de la mañana.")[0] == "es"
    assert heuristic_langid("Le chat dort sur le canapé et il est très content avec le soleil du matin.")[0] == "fr"
    assert heuristic_langid("12345 !!!")[0] == "unk"
