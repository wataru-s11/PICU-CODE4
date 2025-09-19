from vital_reader import normalize_ie_text


def test_normalize_ie_ratio_trims_trailing_decimal_zero():
    assert normalize_ie_text("1:2.0") == "1:2"


def test_normalize_ie_ratio_keeps_fractional_part():
    assert normalize_ie_text("1:2.5") == "1:2.5"


def test_normalize_ie_ratio_handles_full_width_characters():
    assert normalize_ie_text("1：3．0") == "1:3"


def test_normalize_ie_ratio_handles_alternate_separators_and_leading_zeros():
    assert normalize_ie_text("01/02.0") == "1:2"
