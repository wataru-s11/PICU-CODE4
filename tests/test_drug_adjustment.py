import drug_adjustment as da


def test_noradrenaline_reduction_and_hamp_increase():
    actions = da.adjust_medication(0.05, 0.02, 0.15, 0)
    assert "ノルアドレナリンの減量を検討" in actions
    assert "ハンプ増量を検討" in actions


def test_pitresin_reduction():
    actions = da.adjust_medication(0, 0.05, 0.0, 0)
    assert actions == ["ピトレシンの減量を検討"]


def test_kontomin_start():
    actions = da.adjust_medication(0, 0.02, 0.25, 0)
    assert "コントミンを0.1で開始を検討" in actions


def test_kontomin_warning():
    actions = da.adjust_medication(0, 0.02, 0.25, 0.2)
    assert "昇圧/降圧薬調整のみでの降圧は難しい可能性あり" in actions


def test_kontomin_no_special_instruction():
    actions = da.adjust_medication(0, 0.01, 0.25, 0.4)
    assert "コントミン量に関する特別な指示なし" in actions
