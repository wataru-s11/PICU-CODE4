import pandas as pd
import ast
import operator as op
from common.rule_engine import evaluate_rules


_CMP_OPS = {
    ast.Lt: op.lt,
    ast.LtE: op.le,
    ast.Gt: op.gt,
    ast.GtE: op.ge,
    ast.Eq: op.eq,
    ast.NotEq: op.ne,
}

_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}


def _eval_ast(node):
    """Evaluate a limited AST representing a numeric expression or comparison."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Num):  # pragma: no cover (for Python <3.8)
        return node.n
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _eval_ast(node.operand)
        return +val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.Compare):
        left = _eval_ast(node.left)
        for op_node, comp in zip(node.ops, node.comparators):
            right = _eval_ast(comp)
            if not _CMP_OPS[type(op_node)](left, right):
                return False
            left = right
        return True
    raise ValueError(f"Unsupported expression: {ast.dump(node, include_attributes=False)}")


def evaluate_numeric_cond(expr: str) -> bool:
    """Safely evaluate a numeric comparison expression without using eval."""
    try:
        tree = ast.parse(expr, mode="eval")
        return bool(_eval_ast(tree.body))
    except Exception:
        return False


def _replace_placeholders(cond: str, value, thresholds):
    s = str(cond)
    for k, v in thresholds.items():
        s = s.replace(f"{{{{{k}}}}}", str(v))
    return s.replace("value", str(value))

def evaluate_cvp(vitals, tree_df, thresholds, phase='a'):
    """Evaluate CVP related rules from tree data.

    Supports both the legacy Excel-format rows and the newer YAML-based
    ``condition`` expressions.  When ``condition`` is present, the generic
    :func:`common.rule_engine.evaluate_rules` is delegated to, otherwise the
    former numeric-comparison logic is used.
    """
    if tree_df is None or tree_df.empty:
        return []

    # YAML-based tree: delegate to generic evaluator
    first_row = next(tree_df.iterrows())[1] if hasattr(tree_df, "iterrows") else None
    if first_row is not None and first_row.get("condition") is not None:
        return evaluate_rules(vitals, tree_df, ["CVP"], thresholds, phase)

    instructions = []

    for _, row in tree_df.iterrows():
        if row.get('phase(acute=a, reevaluate=r)', 'a') != phase:
            continue

        main_item = row.get('項目')
        if not main_item:
            continue
        main_value = vitals.get(main_item, None)
        try:
            if main_value is None or (isinstance(main_value, float) and pd.isna(main_value)):
                continue
        except Exception:
            pass

        # --- 条件評価 ---
        main_cond = row.get('条件', '')
        if not main_cond:
            continue
        if not evaluate_numeric_cond(_replace_placeholders(main_cond, main_value, thresholds)):
            continue

        ok = True
        # 追加条件1..4
        for i in range(1, 5):
            add_item = row.get(f"追加項目{i}")
            add_cond = row.get(f"追加条件{i}")
            if pd.isna(add_item) or pd.isna(add_cond):
                continue
            add_val = vitals.get(add_item, None)
            try:
                if add_val is None or (isinstance(add_val, float) and pd.isna(add_val)):
                    continue
            except Exception:
                pass

            if add_cond in ["Y", "N"]:
                ok = (str(add_val) == add_cond)
            else:
                expr = _replace_placeholders(add_cond, add_val, thresholds)
                ok = evaluate_numeric_cond(expr)
            if not ok:
                break

        if ok:
            instructions.append({
                'id': row.get('id'),
                'instruction': row.get('介入', ''),
                'comment': row.get('備考', ''),
                'pause_min': row.get('ポーズ(min)', ''),
            })

    return instructions
