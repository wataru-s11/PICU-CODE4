import sys
import types

# Minimal pandas stub
pd_stub = types.SimpleNamespace(isna=lambda x: x != x)
sys.modules.setdefault("pandas", pd_stub)

from vitals.bpup_logic import evaluate_bpup

class DummyDF:
    def __init__(self, rows):
        self._rows = rows
        self.empty = len(rows) == 0
    def iterrows(self):
        for i, r in enumerate(self._rows):
            yield i, r

row = {
    'id': 'BPUP_A',
    'phase(acute=a, reevaluate=r)': 'a',
    'condition': 'True',
    '介入': 'placeholder',
    'ポーズ(min)': 1,
    '再評価用NextID': None,
    '備考': ''
}

def _run(vitals, elapsed=61):
    df = DummyDF([row])
    return evaluate_bpup(vitals, df, {}, 'a', elapsed_minutes=elapsed)[0][
        'instruction'
    ]


def test_bpup_noradrenaline():
    assert _run({'noradrenaline': 0.05}) == 'ノルアドレナリンを減量してもよいです'


def test_bpup_pitressin():
    vitals = {'noradrenaline': 0, 'pitressin': 0.04}
    assert _run(vitals) == 'ピトレシンを減量してもよいです'


def test_bpup_hanp():
    vitals = {'noradrenaline': 0, 'pitressin': 0.02, 'hanp': 0.1}
    assert _run(vitals) == 'ハンプを増量してもよいです'


def test_bpup_contomin_start():
    vitals = {'noradrenaline': 0, 'pitressin': 0.01, 'hanp': 0.25, 'contomin': 0}
    assert _run(vitals) == 'コントミンを0.1で開始してもよいです'


def test_bpup_contomin_difficult():
    vitals = {'noradrenaline': 0, 'pitressin': 0.01, 'hanp': 0.25, 'contomin': 0.2}
    assert _run(vitals) == '昇圧と降圧薬調整での降圧は難しい可能性があります'


def test_bpup_pause_only_after_pitressin_reduction():
    df = DummyDF([row])
    vitals = {'noradrenaline': 0, 'pitressin': 0.04}
    inst = evaluate_bpup(
        vitals, df, {}, 'a', {'pitressin': 0.05}, elapsed_minutes=61
    )[0]
    assert inst['pause_min'] == 1
    inst = evaluate_bpup(
        vitals, df, {}, 'a', {'pitressin': 0.04}, elapsed_minutes=61
    )[0]
    assert inst['pause_min'] == 0


def test_bpup_a_suppressed_within_one_hour():
    df = DummyDF([row])
    vitals = {'noradrenaline': 0.05}
    assert evaluate_bpup(vitals, df, {}, 'a', elapsed_minutes=30) == []
