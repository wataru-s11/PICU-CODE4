from common.rule_engine import evaluate_rules

# Critical SpO2に関するtreeルールを評価

def evaluate_critical_spo2(vitals, tree_df, thresholds=None, phase='a'):
    """Evaluate Critical SpO2 related rules from tree data."""
    instructions = evaluate_rules(vitals, tree_df, ["CRIT"], thresholds, phase)
    return [inst for inst in instructions if "SpO2" in inst["id"]]
