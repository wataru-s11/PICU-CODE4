from common.rule_engine import evaluate_rules

# BPDOWN（SBP低下時）に関連する複数薬剤介入（VASO/HANP/CONT）含む

def evaluate_bpdown(vitals, tree_df, thresholds=None, phase='a'):
    """Evaluate BPDOWN-related rules from the tree data."""
    prefixes = ["BPDOWN", "VASO_LOW", "VASO_HIGH"]
    return evaluate_rules(vitals, tree_df, prefixes, thresholds, phase)
