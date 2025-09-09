from common.rule_engine import evaluate_rules

# アドレナリン（AD）に関するtreeルールを評価

def evaluate_adrenaline(vitals, tree_df, thresholds=None, phase='a'):
    """Evaluate adrenaline related rules from tree data."""
    return evaluate_rules(vitals, tree_df, ["AD"], thresholds, phase)
