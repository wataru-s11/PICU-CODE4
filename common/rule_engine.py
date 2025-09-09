import pandas as pd
from .tree_parser import row_matches

def evaluate_rules(vitals, tree_df, prefixes, thresholds=None, phase='a'):
    """Generic evaluator for tree-based rules.

    Parameters
    ----------
    vitals : dict
        Latest vital values.
    tree_df : DataFrame-like
        Rules loaded via ``load_tree``.  Supports both YAML ("condition") and
        Excel-style rows ("項目", "比較", ...).
    prefixes : list[str]
        List of id prefixes to filter relevant rules (e.g., ["SBP"]).
    thresholds : dict, optional
        Additional threshold values available within conditions.
    phase : str, default 'a'
        Evaluate rules only for this phase ('a' acute, 'r' reevaluate).
    """
    def _maybe_float(val):
        """Convert numeric strings to floats for safe comparisons."""
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                return val
        return val

    def _convert_dict(d):
        return {k: _maybe_float(v) for k, v in (d or {}).items()}

    scope = _convert_dict(vitals)
    scope["vitals"] = _convert_dict(vitals)
    if thresholds:
        scope.update(_convert_dict(thresholds))
    instructions = []
    for _, row in tree_df.iterrows():
        rid = row.get("id")
        if not isinstance(rid, str):
            continue
        if prefixes and not any(rid.startswith(p) for p in prefixes):
            continue
        if row.get("phase(acute=a, reevaluate=r)", "a") != phase:
            continue
        cond_ok = False
        # YAML-style: condition expression
        cond_str = row.get("condition")
        if cond_str is not None:
            try:
                cond_ok = bool(eval(str(cond_str), {"__builtins__": {}}, scope))
            except Exception:
                cond_ok = False
        # Excel-style: use row_matches helper
        elif row.get("項目") is not None and row.get("条件") is not None:
            main_item = row.get("項目")
            primary_value = vitals.get(main_item)
            if primary_value is not None:
                try:
                    cond_ok = row_matches(row, primary_value, vitals, thresholds)
                except Exception:
                    cond_ok = False
        if not cond_ok:
            continue
        comment = row.get("備考", "")
        try:
            if pd.isna(comment):
                comment = ""
        except Exception:
            pass
        instructions.append({
            "id": rid,
            "instruction": row.get("介入", ""),
            "pause_min": row.get("ポーズ(min)", ""),
            "next_id": row.get("再評価用NextID"),
            "comment": comment,
        })
    return instructions
