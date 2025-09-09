import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import bleeding_panel


def test_manage_bleeding_calls_surgeon_for_arterial(capsys):
    responses = iter(["1"])  # arterial color leads to immediate call
    bleeding_panel.manage_bleeding(input_func=lambda _: next(responses))
    out = capsys.readouterr().out
    assert "外科医をコールしてください。" in out
    assert "RBC:FFP:PC" not in out


def test_manage_bleeding_moderate_not_sticky_high_cvp_mediastinum(capsys):
    responses = iter(["2", "2", "2", "8", "3"])  # venous, moderate, not sticky, CVP=8, mediastinum
    bleeding_panel.manage_bleeding(input_func=lambda _: next(responses))
    out = capsys.readouterr().out
    assert "[フロセミド5mg IV]を提案します。" in out
    assert "RBC:FFP:PC = 2:1:1" in out
    assert "縦隔出血です。タンポナーデにならないように定期的なミルキングを行ってください。" in out


def test_manage_bleeding_site_none(capsys):
    responses = iter(["2", "3", "4"])  # venous, small amount, site none
    bleeding_panel.manage_bleeding(input_func=lambda _: next(responses))
    out = capsys.readouterr().out
    assert "経過観察を行います。" in out
    assert "特になし。経過観察を行い、ヘパリン持続投与を開始するか検討してください。" in out
