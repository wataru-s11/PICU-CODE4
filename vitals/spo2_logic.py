from common.rule_engine import evaluate_rules

# SpO2に関するtreeルールを評価

def evaluate_spo2(vitals, tree_df, thresholds=None, phase='a'):
    """Evaluate SpO2 related rules from tree data."""
    return evaluate_rules(vitals, tree_df, ["SPO2"], thresholds, phase)
