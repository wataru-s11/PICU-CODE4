from types import SimpleNamespace
from vitals.bpup_logic import evaluate_bpup
from vitals.bpdown_logic import evaluate_bpdown
from vitals.bleed_logic import evaluate_bleed
from vitals.transfusion_logic import evaluate_transfusion


def _df(rows):
    def iterrows():
        for i, r in enumerate(rows):
            yield i, r
    return SimpleNamespace(iterrows=iterrows)


def test_evaluate_bpup_yaml():
    df = _df([
        {
            'id': 'BPUP_RULE',
            'phase(acute=a, reevaluate=r)': 'a',
            'condition': 'True',
            '介入': 'up',
            'ポーズ(min)': 1,
            '再評価用NextID': None,
            '備考': ''
        },
        {
            'id': 'BPDOWN_RULE',
            'phase(acute=a, reevaluate=r)': 'a',
            'condition': 'True',
            '介入': 'down',
            'ポーズ(min)': 1,
            '再評価用NextID': None,
            '備考': ''
        },
    ])
    result = evaluate_bpup({}, df, {}, 'a')
    assert result == [{
        'id': 'BPUP_RULE',
        'instruction': 'up',
        'pause_min': 1,
        'next_id': None,
        'comment': ''
    }]


def test_evaluate_bpdown_yaml():
    df = _df([
        {
            'id': 'BPDOWN_RULE',
            'phase(acute=a, reevaluate=r)': 'a',
            'condition': 'True',
            '介入': 'down',
            'ポーズ(min)': 1,
            '再評価用NextID': None,
            '備考': ''
        },
        {
            'id': 'BPUP_RULE',
            'phase(acute=a, reevaluate=r)': 'a',
            'condition': 'True',
            '介入': 'up',
            'ポーズ(min)': 1,
            '再評価用NextID': None,
            '備考': ''
        },
    ])
    result = evaluate_bpdown({}, df, {}, 'a')
    assert result == [{
        'id': 'BPDOWN_RULE',
        'instruction': 'down',
        'pause_min': 1,
        'next_id': None,
        'comment': ''
    }]


def test_evaluate_bleed_yaml():
    df = _df([
        {
            'id': 'BLEED_RULE',
            'phase(acute=a, reevaluate=r)': 'a',
            'condition': 'True',
            '介入': 'bleed',
            'ポーズ(min)': 1,
            '再評価用NextID': None,
            '備考': ''
        },
        {
            'id': 'TRANSFUSION_RULE',
            'phase(acute=a, reevaluate=r)': 'a',
            'condition': 'True',
            '介入': 'trans',
            'ポーズ(min)': 1,
            '再評価用NextID': None,
            '備考': ''
        },
    ])
    result = evaluate_bleed({}, df, {}, 'a')
    assert result == [{
        'id': 'BLEED_RULE',
        'instruction': 'bleed',
        'pause_min': 1,
        'next_id': None,
        'comment': ''
    }]


def test_evaluate_transfusion_yaml():
    df = _df([
        {
            'id': 'TRANSFUSION_RULE',
            'phase(acute=a, reevaluate=r)': 'a',
            'condition': 'True',
            '介入': 'trans',
            'ポーズ(min)': 1,
            '再評価用NextID': None,
            '備考': ''
        },
        {
            'id': 'BLEED_RULE',
            'phase(acute=a, reevaluate=r)': 'a',
            'condition': 'True',
            '介入': 'bleed',
            'ポーズ(min)': 1,
            '再評価用NextID': None,
            '備考': ''
        },
    ])
    result = evaluate_transfusion({}, df, {}, 'a')
    assert result == [{
        'id': 'TRANSFUSION_RULE',
        'instruction': 'trans',
        'pause_min': 1,
        'next_id': None,
        'comment': ''
    }]

