from common.rule_engine import evaluate_rules

# TRANSFUSION（輸血介入）に関するルールを評価

def evaluate_transfusion(vitals, tree_df, thresholds=None, phase='a'):
    """Evaluate transfusion-related rules from the tree data."""
    prefixes = ["TRANSFUSION"]
    return evaluate_rules(vitals, tree_df, prefixes, thresholds, phase)
