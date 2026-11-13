"""Schema 树构建、标准三层 LIST 识别与逻辑叶路径解析。

物理 schema（Parquet SchemaElement 扁平数组）被还原为节点树：

* 根节点: kind="root"
* 结构体: kind="struct"
* 列表:   kind="list"，对应标准三层
          group<LIST> -> repeated group<list> -> element
* 叶子:   kind="leaf"，仅允许 INT64 与 UTF8（STRING / 带 UTF8 注解的
          BYTE_ARRAY）

逻辑路径是字段名数组；空数组 ``[]`` 表示列表的“当前元素”。
"""

from .errors import (
    CorruptMetadataError,
    InvalidPathError,
    UnsupportedFormatError,
)
from .metadata import (
    CONV_LIST,
    CONV_UTF8,
    REP_OPTIONAL,
    REP_REPEATED,
    REP_REQUIRED,
    TYPE_BYTE_ARRAY,
    TYPE_INT64,
)


class Node:
    __slots__ = (
        "name", "kind", "repetition", "physical_type", "is_utf8",
        "children", "parent", "max_def", "max_rep", "depth",
        "leaf_children", "element_node", "list_wrapper", "repeated_group",
        "index", "projected", "root_ref",
    )

    def __init__(self, name, kind, repetition=None):
        self.name = name
        self.kind = kind            # root/struct/list/leaf
        self.repetition = repetition  # required/optional/repeated
        self.physical_type = None
        self.is_utf8 = False
        self.children = []
        self.parent = None
        self.max_def = 0
        self.max_rep = 0
        self.depth = 0
        self.leaf_children = []
        # list 节点专用
        self.element_node = None    # 元素节点（repeated group 的唯一子）
        self.list_wrapper = None    # 带 LIST 注解的 optional group 节点
        self.repeated_group = None  # repeated group<list> 节点
        self.index = -1
        self.projected = False
        self.root_ref = None


def _build_tree(schema_elements):
    """把扁平 SchemaElement 数组还原成节点树（不做深度/列表标注）。"""
    if not schema_elements:
        raise CorruptMetadataError("schema 为空")
    root_el = schema_elements[0]
    if root_el.repetition_type is not None:
        raise CorruptMetadataError("schema 根元素不应带 repetition_type")
    root = Node(root_el.name, "root")
    root.index = 0
    root.root_ref = root

    # 用栈按 num_children 还原嵌套
    stack = [(root, root_el.num_children or 0)]
    idx = 1
    while idx < len(schema_elements):
        parent, remaining = stack[-1]
        if remaining == 0:
            stack.pop()
            continue
        el = schema_elements[idx]
        node = Node(el.name, "struct", el.repetition_type)
        node.index = idx
        node.parent = parent
        node.root_ref = root
        parent.children.append(node)
        if el.type is not None:
            node.kind = "leaf"
            node.physical_type = el.type
            node.is_utf8 = (
                el.converted_type == CONV_UTF8 or el.logical_type == "STRING"
            )
        stack[-1] = (parent, remaining - 1)
        if el.num_children:
            stack.append((node, el.num_children))
        idx += 1

    # 栈必须恰好清空
    while stack and stack[-1][1] == 0:
        stack.pop()
    if stack:
        raise CorruptMetadataError("SchemaElement 的 num_children 与元素数量不一致")
    return root


def _annotate(root, schema_elements):
    """计算深度、max definition/repetition level，并识别三层 LIST。"""

    def walk(node, depth, cur_def, cur_rep):
        node.depth = depth
        if node.kind == "root":
            node.max_def, node.max_rep = cur_def, cur_rep
        else:
            add_def = 1 if node.repetition in (REP_OPTIONAL, REP_REPEATED) else 0
            add_rep = 1 if node.repetition == REP_REPEATED else 0
            cur_def += add_def
            cur_rep += add_rep
            node.max_def, node.max_rep = cur_def, cur_rep

        if node.kind == "leaf":
            node.leaf_children = [node]
            _validate_leaf(node)
            return node.leaf_children

        # 列表识别：带 LIST 注解，或结构上恰好是标准三层
        list_info = _try_as_list(node, schema_elements)
        if list_info is not None:
            wrapper, repeated, element = list_info
            node.kind = "list"
            node.list_wrapper = wrapper
            node.repeated_group = repeated
            node.element_node = element
            # 走到这里时 cur_def/cur_rep 已计入 wrapper 自身；
            # 隐式 repeated group 再贡献一级 def 与一级 rep
            rep_add_def = 1 if repeated.repetition in (REP_OPTIONAL,
                                                        REP_REPEATED) else 0
            rep_add_rep = 1 if repeated.repetition == REP_REPEATED else 0
            repeated.max_def = cur_def + rep_add_def
            repeated.max_rep = cur_rep + rep_add_rep
            walk(element, depth + 1,
                 cur_def + rep_add_def, cur_rep + rep_add_rep)
            node.leaf_children = element.leaf_children
            return node.leaf_children

        if not node.children:
            raise CorruptMetadataError(
                f"组字段 {node.name!r} 没有子节点", None
            )
        leaves = []
        for child in node.children:
            leaves.extend(walk(child, depth + 1, cur_def, cur_rep))
        node.leaf_children = leaves
        return leaves

    walk(root, 0, 0, 0)
    return root


def _validate_leaf(node):
    t = node.physical_type
    if t == TYPE_INT64:
        return
    if t == TYPE_BYTE_ARRAY and node.is_utf8:
        return
    name = {
        0: "BOOLEAN", 1: "INT32", 2: "INT64", 3: "INT96",
        4: "FLOAT", 5: "DOUBLE", 6: "BYTE_ARRAY", 7: "FIXED_LEN_BYTE_ARRAY",
    }.get(t, str(t))
    raise UnsupportedFormatError(
        f"叶列 {node.name!r} 的物理类型 {name} 不受支持"
        "（仅支持 INT64 与 UTF8）"
    )


def _try_as_list(node, schema_elements):
    """若 node 是标准三层 LIST，返回 (wrapper, repeated_group, element)。

    标准三层::

        optional group <name> (LIST) {
          repeated group <list-name> {
            <element-type> <element-name>;
          }
        }

    注解缺失但结构完全一致的三层写法同样接受；两层的 repeated 原始写法
    不属于“标准三层 LIST”，拒绝处理。
    """
    if node.kind != "struct" or len(node.children) != 1:
        return None
    wrapper = node
    el_wrapper = schema_elements[wrapper.index]
    repeated = wrapper.children[0]
    if repeated.kind != "struct" or repeated.repetition != REP_REPEATED:
        return None
    if len(repeated.children) != 1:
        return None
    element = repeated.children[0]
    annotated = (
        el_wrapper.converted_type == CONV_LIST
        or el_wrapper.logical_type == "LIST"
    )
    structural = (
        wrapper.repetition in (REP_OPTIONAL, REP_REQUIRED)
        and element.repetition in (REP_OPTIONAL, REP_REQUIRED)
    )
    if not (annotated or structural):
        return None
    if not annotated and not structural:  # pragma: no cover - 防御
        return None
    # 显式非 LIST 注解则不按列表处理
    if (el_wrapper.converted_type is not None
            and el_wrapper.converted_type != CONV_LIST):
        return None
    if (el_wrapper.logical_type is not None
            and el_wrapper.logical_type != "LIST"):
        return None
    if not structural:
        raise UnsupportedFormatError(
            f"字段 {wrapper.name!r} 带 LIST 注解但不是标准三层 LIST 结构"
        )
    return wrapper, repeated, element


def build_schema(schema_elements):
    root = _build_tree(schema_elements)
    _annotate(root, schema_elements)
    return root


# ---------------------------------------------------------------------
# 逻辑路径解析
# ---------------------------------------------------------------------

def resolve_path(root, logical_path):
    """把逻辑路径（字段名数组，含 [] 占位）解析为节点。

    路径必须落在一个叶子上（叶路径），否则抛 InvalidPathError。
    """
    if not isinstance(logical_path, list) or not logical_path:
        raise InvalidPathError("逻辑路径必须是非空数组")
    node = root
    i = 0
    while i < len(logical_path):
        seg = logical_path[i]
        if seg == []:
            # 进入当前列表的元素；元素本身可能就是叶子
            if node.kind != "list":
                raise InvalidPathError(
                    f"路径段 [] 出现在非列表节点 {node.name!r} 上"
                )
            node = node.element_node
            i += 1
            # 元素是叶子时，路径必须在此结束
            if node.kind == "leaf" and i != len(logical_path):
                raise InvalidPathError(
                    f"列表元素 {node.name!r} 本身是叶子，路径不能继续："
                    f"{logical_path}"
                )
            continue
        if not isinstance(seg, str):
            raise InvalidPathError(f"路径段必须是字段名字符串或 []，实际 {seg!r}")
        if node.kind == "root":
            node = _find_child(node, seg)
        elif node.kind == "struct":
            node = _find_child(node, seg)
        elif node.kind == "list":
            # 嵌套 list：外层 [] 已进入元素，该元素本身又是一个 list 节点，
            # 此时用其字段名（内层 list 名）继续导航是合法的。
            if node.name == seg:
                pass  # 已在该 list 节点，保持
            else:
                raise InvalidPathError(
                    f"列表 {node.name!r} 后必须先经过 [] 才能访问元素字段 "
                    f"{seg!r}"
                )
        else:
            raise InvalidPathError(f"叶子节点 {node.name!r} 下没有字段 {seg!r}")
        i += 1
    if node.kind != "leaf":
        raise InvalidPathError(
            f"路径 {logical_path} 指向 {node.kind} 节点 {node.name!r}，不是叶路径"
        )
    return node


def _find_child(node, name):
    for child in node.children:
        if child.name == name:
            return child
    available = [c.name for c in node.children]
    raise InvalidPathError(
        f"节点 {node.name!r} 下不存在字段 {name!r}，可选字段 {available}"
    )


def collect_leaves(node):
    """返回节点下全部物理叶子（保持 schema 顺序）。"""
    return list(node.leaf_children)


def describe_leaf(node):
    """返回叶子的逻辑类型描述，供谓词字面量类型检查。"""
    if node.physical_type == TYPE_INT64:
        return "INT64"
    return "UTF8"
