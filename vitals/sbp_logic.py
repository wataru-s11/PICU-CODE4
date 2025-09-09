from common.rule_engine import evaluate_rules

# SBPに関するtreeルールを評価

def evaluate_sbp(vitals, tree_df, thresholds=None, phase='a'):
    """Evaluate SBP related rules from tree data.

    Parameters
    ----------
    vitals : dict
        Latest vital values.
    tree_df : DataFrame-like
        Rules loaded via ``load_tree`` (YAML or Excel).
    thresholds : dict, optional
        Threshold values used inside conditions.
    phase : str, default 'a'
        'a' for acute phase or 'r' for reevaluation.
    """
    return evaluate_rules(vitals, tree_df, ["SBP"], thresholds, phase)
