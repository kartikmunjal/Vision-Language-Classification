from vision_language_classification.labelers import parse_structured_labels


def test_parse_compact_integer_llm_output():
    payload = {
        "multiple_subjects": 0,
        "outdoor": 0,
        "human_present": 1,
        "animal_present": 0,
        "dynamic_scene": 0,
        "night": 0,
    }
    parsed = parse_structured_labels(payload)
    assert parsed["human_present"] == {"label": 1, "confidence": 1.0}


def test_parse_count_like_output_normalizes_to_binary_presence():
    payload = {
        "multiple_subjects": 2,
        "outdoor": 0,
        "human_present": 3,
        "animal_present": 0,
        "dynamic_scene": 1,
        "night": 0,
    }
    parsed = parse_structured_labels(payload)
    assert parsed["multiple_subjects"]["label"] == 1
    assert parsed["human_present"]["label"] == 1
