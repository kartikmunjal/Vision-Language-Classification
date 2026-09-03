from vision_language_classification.schema import stable_id


def test_stable_id_is_repeatable():
    assert stable_id("139", "000000000139.jpg") == stable_id("139", "000000000139.jpg")
    assert stable_id("139", "000000000139.jpg") != stable_id("285", "000000000285.jpg")
