from common.rule_engine import evaluate_rules

# DOB（ドブタミン）に関するtreeルールを評価

def evaluate_dobutamine(vitals, tree_df, thresholds=None, phase='a'):
    """Evaluate dobutamine related rules from tree data."""
    return evaluate_rules(vitals, tree_df, ["DOB"], thresholds, phase)
