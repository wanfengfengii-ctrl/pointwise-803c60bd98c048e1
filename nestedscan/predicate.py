"""谓词编译与 SQL 三值逻辑求值。

支持的谓词形式::

    ["eq", 路径, int|str]
    ["and", p, q]
    ["not", p]
    ["any", 列表路径, p]      # p 内路径相对于列表元素

三值逻辑常量: TRUE / FALSE / UNKNOWN。空列表为假，null 列表为未知，
只有求值为 TRUE 的行才命中。
"""

from .errors import InvalidPathError, InvalidPredicateError
from .schema import describe_leaf, resolve_path

TRUE = "TRUE"
FALSE = "FALSE"
UNKNOWN = "UNKNOWN"

# 导航时的“未投影/结构不存在”哨兵，区别于 None（SQL null）
_MISSING = object()


def compile_predicate(where, root):
    """把用户谓词编译为带解析节点的内部结构，并做全部静态校验。"""
    if where is None:
        return None
    return _compile(where, root, base=[])


def _compile(expr, root, base):
    if not isinstance(expr, list) or not expr:
        raise InvalidPredicateError(f"谓词必须是非空数组，实际 {expr!r}")
    op = expr[0]
    if op == "eq":
        if len(expr) != 3:
            raise InvalidPredicateError('eq 谓词必须是 ["eq", 路径, 字面量]')
        rel_path = expr[1]
        # 合法路径：非空数组；在 any 元素作用域内允许裸 [] 表示当前元素值
        if not isinstance(rel_path, list) or not rel_path:
            raise InvalidPredicateError(
                f"eq 路径必须是非空数组（可用 [] 表示当前值），实际 {rel_path!r}"
            )
        is_current = rel_path == [[]]
        if is_current:
            # 当前元素值：full_path 复用基准（已定位到元素叶）
            full_path = list(base)
            if not full_path:
                raise InvalidPredicateError(
                    "裸 [] 路径只能出现在 any 的元素作用域内"
                )
        else:
            full_path = _join_path(base, rel_path)
        leaf = resolve_path(root, full_path)
        _check_literal(leaf, expr[2])
        rel = [] if is_current else _strip_base(base, rel_path)
        return {"op": "eq", "path": full_path, "rel": rel,
                "leaf": leaf, "leaf_name": leaf.name,
                "literal": expr[2]}
    if op == "and":
        if len(expr) != 3:
            raise InvalidPredicateError('and 谓词必须是 ["and", p, q]')
        return {"op": "and",
                "left": _compile(expr[1], root, base),
                "right": _compile(expr[2], root, base)}
    if op == "not":
        if len(expr) != 2:
            raise InvalidPredicateError('not 谓词必须是 ["not", p]')
        return {"op": "not", "child": _compile(expr[1], root, base)}
    if op == "any":
        if len(expr) != 3:
            raise InvalidPredicateError('any 谓词必须是 ["any", 列表路径, p]')
        list_path = _join_path(base, expr[1])
        node = _resolve_node(root, list_path)
        if node.kind != "list":
            raise InvalidPathError(
                f"any 的路径 {list_path} 必须指向列表节点，实际为 {node.kind}"
            )
        rel = _strip_base(base, expr[1])
        # 元素内谓词的基准路径 = 列表路径 + []
        inner_base = list_path + [[]]
        return {"op": "any", "path": list_path, "rel": rel,
                "list_node": node,
                "child": _compile(expr[2], root, inner_base)}
    raise InvalidPredicateError(f"未知谓词操作符 {op!r}")


def _strip_base(base, rel_path):
    """计算相对当前作用域的导航段：去掉基准前缀，并丢弃 [] 占位。

    返回的字段名序列用于在元素作用域（物化 dict）内取值；空序列表示
    作用域本身就是目标（leaf 元素时由求值器按叶名解包）。
    """
    return [seg for seg in rel_path if seg != []]


def _join_path(base, rel):
    # [] 是合法的“当前值”路径（非空数组，唯一元素为空列表）
    if not isinstance(rel, list) or not rel:
        raise InvalidPredicateError(f"谓词路径必须是非空数组，实际 {rel!r}")
    joined = list(base)
    for seg in rel:
        if seg == []:
            # 基准末尾已是元素作用域时，内层 [] 表示当前元素，合并去重
            if joined and joined[-1] == []:
                continue
            joined.append([])
        else:
            joined.append(seg)
    if not joined:
        raise InvalidPredicateError("谓词路径解析为空")
    return joined


def _resolve_node(root, path):
    """解析路径到任意节点（用于 any 的列表定位）。"""
    node = root
    for seg in path:
        if seg == []:
            if node.kind != "list":
                raise InvalidPathError("[] 只能作用于列表节点")
            node = node.element_node
            continue
        if node.kind in ("root", "struct"):
            found = None
            for child in node.children:
                if child.name == seg:
                    found = child
                    break
            if found is None:
                raise InvalidPathError(
                    f"节点 {node.name!r} 下不存在字段 {seg!r}"
                )
            node = found
        elif node.kind == "list":
            # 嵌套 list：外层 [] 已进入元素，该元素本身又是 list 节点，
            # 用内层 list 名继续导航是合法的；否则拒绝跨元素直接取字段。
            if node.name != seg:
                raise InvalidPathError(
                    f"列表 {node.name!r} 后必须先经过 [] 才能访问 {seg!r}"
                )
        else:
            raise InvalidPathError(f"叶子 {node.name!r} 下没有字段 {seg!r}")
    return node


def _check_literal(leaf, literal):
    kind = describe_leaf(leaf)
    if kind == "INT64":
        if not isinstance(literal, int) or isinstance(literal, bool):
            raise InvalidPredicateError(
                f"INT64 列的 eq 字面量必须是整数，实际 {literal!r}"
            )
    else:
        if not isinstance(literal, str):
            raise InvalidPredicateError(
                f"UTF8 列的 eq 字面量必须是字符串，实际 {literal!r}"
            )


def collect_predicate_leaves(compiled):
    """收集谓词引用的全部物理叶节点（保持出现顺序、去重）。"""
    leaves = []
    seen = set()

    def walk(node):
        if node is None:
            return
        if node["op"] == "eq":
            if node["leaf"].index not in seen:
                seen.add(node["leaf"].index)
                leaves.append(node["leaf"])
        elif node["op"] == "and":
            walk(node["left"])
            walk(node["right"])
        elif node["op"] == "not":
            walk(node["child"])
        elif node["op"] == "any":
            walk(node["child"])

    walk(compiled)
    return leaves


# ---------------------------------------------------------------------
# 三值逻辑
# ---------------------------------------------------------------------

def _and3(a, b):
    if a == FALSE or b == FALSE:
        return FALSE
    if a == UNKNOWN or b == UNKNOWN:
        return UNKNOWN
    return TRUE


def _not3(a):
    if a is TRUE:
        return FALSE
    if a is FALSE:
        return TRUE
    return UNKNOWN


def _eq3(value, literal):
    if value is None:                 # SQL null
        return UNKNOWN
    if isinstance(literal, int):
        if isinstance(value, bool) or not isinstance(value, int):
            return FALSE
    elif not isinstance(value, type(literal)):
        return FALSE
    return TRUE if value == literal else FALSE


# ---------------------------------------------------------------------
# 在物化结构上按“作用域 + 相对路径”导航
# ---------------------------------------------------------------------

def _navigate(scope, rel_path):
    """从作用域对象出发沿相对字段名导航（仅穿越 dict，不进入 list）。"""
    cur = scope
    for seg in rel_path:
        if cur is None:
            return _MISSING
        if isinstance(cur, dict):
            if seg not in cur:
                return _MISSING
            cur = cur[seg]
        else:
            return _MISSING
    return cur


def evaluate(compiled, row):
    """在物化行（嵌套 dict/list，含 None）上求三值逻辑。"""
    return _eval(compiled, row)


def _eval(compiled, scope):
    op = compiled["op"]
    if op == "eq":
        rel = compiled["rel"]
        if not rel:
            # 当前作用域即目标：leaf 元素被物化为 {leaf_name: value}
            if isinstance(scope, dict) and compiled["leaf_name"] in scope:
                value = scope[compiled["leaf_name"]]
            else:
                value = scope
        else:
            value = _navigate(scope, rel)
        if value is _MISSING:
            return UNKNOWN
        return _eq3(value, compiled["literal"])
    if op == "and":
        return _and3(_eval(compiled["left"], scope),
                     _eval(compiled["right"], scope))
    if op == "not":
        return _not3(_eval(compiled["child"], scope))
    if op == "any":
        return _eval_any(compiled, scope)
    raise InvalidPredicateError(f"内部谓词类型 {op!r} 无法求值")


def _eval_any(compiled, scope):
    """对 rel 路径末端的列表做存在性判定，支持路径中嵌套列表。

    rel 是相对当前作用域的字段名序列（不含 []）。物化结构里列表值挂在
    同名键下，元素是 dict（struct 元素）或 {名: 值}（leaf/list 元素）。
    沿路径逐段下沉：dict 取键；当某段的值是 list 时，对每个元素递归处理
    剩余路径，最终在末端列表上应用子谓词。
    """
    return _any_along(scope, compiled["rel"], compiled["child"])


def _any_along(scope, rel, child):
    if not rel:
        # 路径走完：scope 是末端列表的“单个元素”（leaf 包装 dict /
        # struct dict / None），直接应用子谓词。
        if scope is None:
            return UNKNOWN
        return _eval(child, scope)

    seg, rest = rel[0], rel[1:]
    if scope is None:
        return UNKNOWN
    if not isinstance(scope, dict) or seg not in scope:
        return UNKNOWN
    value = scope[seg]
    if value is None:
        return UNKNOWN                      # null 列表/字段 -> 未知
    if isinstance(value, list):
        if len(value) == 0:
            return FALSE                   # 空列表 -> 假
        result = UNKNOWN
        for element in value:
            sub = UNKNOWN if element is None else _any_along(element, rest, child)
            if sub == TRUE:
                return TRUE
            if sub == UNKNOWN:
                result = UNKNOWN
        return result
    # 普通 dict 字段，继续下沉
    return _any_along(value, rest, child)
