import pandas as pd
import re
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def _parse_condition(cond: str, item: str) -> str:
    """Convert a small DSL used in tree.yaml into a Python expression."""
    s = str(cond).strip()
    s = s.replace("{{", "").replace("}}", "")
    # Replace direct dictionary access (vitals['X']) with a safe getter to avoid
    # ``KeyError`` when a vital sign is missing from the ``vitals`` dict.  This
    # mirrors how conditions written using ``value`` are expanded and keeps rule
    # evaluation robust even if some measurements are unavailable.
    s = re.sub(r"vitals\[['\"]([^'\"\]]+)['\"]\]", r"vitals.get('\1')", s)

    if s in ("True", "False"):
        return s

    if s in ("Y", "N"):
        return f"vitals.get('{item}') == '{s}'"

    m = re.match(r'vitals\["([^"\']+)"\]\s*([0-9.,\-]+)\s*(>=|<=|>|<|=)', s)
    if m:
        var, val, op = m.groups()
        op = {'=': '==', '>': '>', '<': '<', '>=': '>=', '<=': '<='}[op]
        nums = re.findall(r'[0-9]+(?:\.[0-9]+)?', val)
        if len(nums) == 1:
            return f"vitals.get('{var}') {op} {nums[0]}"
        if '-' in val and len(nums) == 2:
            return f"{nums[0]} <= vitals.get('{var}') <= {nums[1]}"
        return f"vitals.get('{var}') in [{', '.join(nums)}]"

    if s.startswith('value'):
        m = re.match(r'value\s*(>=|<=|>|<|=)\s*(.+)', s)
        if m:
            op, rhs = m.groups()
            op = {'=': '==', '>': '>', '<': '<', '>=': '>=', '<=': '<='}[op]
            nums = re.findall(r'[0-9]+(?:\.[0-9]+)?', rhs)
            if len(nums) == 1:
                return f"vitals.get('{item}') {op} {nums[0]}"
            if '-' in rhs and len(nums) == 2:
                return f"{nums[0]} <= vitals.get('{item}') <= {nums[1]}"
            if not nums and rhs.strip():
                return f"vitals.get('{item}') {op} {rhs.strip()}"
            return f"vitals.get('{item}') in [{', '.join(nums)}]"

    return s


def _parse_actions(actions):
    pause = None
    next_id = None
    for act in actions or []:
        if not isinstance(act, str):
            continue
        m = re.search(r'POSE_(\d+)', act)
        if m and pause is None:
            pause = float(m.group(1))
        if act.startswith('NEXT:') and next_id is None:
            frag = act.split(':', 1)[1].strip()
            frag = re.split(r'[，,\s]', frag)[0]
            if frag and frag not in ('なし', 'None'):
                next_id = frag
    return pause, next_id


def load_tree(path):
    p = Path(path)
    if p.suffix.lower() in ('.yaml', '.yml'):
        if yaml is None:
            raise RuntimeError("yaml module not available")
        with open(p, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        rows = []
        for rule in data.get('rules', []):
            rid = rule.get('id')
            tags = [t for t in (rule.get('tags') or []) if isinstance(t, str)]
            phases = []
            items = []
            for t in tags:
                for part in t.split(','):
                    part = part.strip()
                    if part in ('a', 'r'):
                        phases.append(part)
                    else:
                        items.append(part)
            item = items[0] if items else ''
            cond = _parse_condition(rule.get('when', 'True'), item)
            pause, nxt = _parse_actions(rule.get('actions'))
            if not phases:
                phases = ['a']
            for ph in phases:
                rows.append({
                    'id': rid,
                    'phase(acute=a, reevaluate=r)': ph,
                    'condition': cond,
                    '介入': rule.get('message', ''),
                    'ポーズ(min)': pause,
                    '再評価用NextID': nxt,
                    '備考': '',
                })
        return pd.DataFrame(rows)
    return pd.read_excel(p, sheet_name=0)

def row_matches(row, primary_value, vitals, thresholds=None):
    """
    tree.xlsx の1行(row)と最新の値を比較し、条件がすべて一致するか判定
    - primary_value: row['項目'] に対応する値（例: SpO2）
    - vitals: dict。全バイタル。
    - thresholds: dict。しきい値（row内で {{KEY}} 参照可）
    """
    scope = dict(vitals)
    if thresholds:
        scope.update(thresholds)

    def _resolve(val):
        if isinstance(val, str):
            pattern = re.compile(r"\{\{([^}]+)\}\}")
            val = pattern.sub(lambda m: str(scope.get(m.group(1), m.group(0))), val)
        return val

    # 主条件
    main_threshold = _resolve(row["閾値(記入なしはユーザー設定・固定値は記入）"])
    if not _compare(primary_value, row["比較"], main_threshold):
        return False

    # 追加条件（最大4つ）
    for i in range(1, 5):
        key = row.get(f"追加条件項目{i}")
        op = row.get(f"追加条件比較{i}")
        val = row.get(f"追加条件閾値{i}")

        if pd.isna(key) or pd.isna(op) or pd.isna(val):
            continue  # 条件なし

        key = str(key)
        if key not in scope:
            return False  # 測定値・しきい値がない

        if not _compare(scope[key], op, _resolve(val)):
            return False

    return True

def _compare(a, operator, b):
    try:
        a = float(a)
        b = float(b)
    except:
        return False

    if operator == ">":
        return a > b
    if operator == "<":
        return a < b
    if operator == "=":
        return a == b
    if operator == ">=":
        return a >= b
    if operator == "<=":
        return a <= b
    return False  # 未知の演算子
