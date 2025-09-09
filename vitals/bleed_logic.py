from common.rule_engine import evaluate_rules

# 出血チェック（BLEED, PROPERTY, COLOR, MOUNT など）に関するルールを評価

def evaluate_bleed(vitals, tree_df, thresholds=None, phase='a'):
    """Evaluate bleeding-related rules from the tree data."""
    prefixes = ["BLEED", "PROPERTY", "COLOR", "MOUNT"]
    return evaluate_rules(vitals, tree_df, prefixes, thresholds, phase)
