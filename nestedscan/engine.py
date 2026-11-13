"""扫描引擎：把元数据、schema、谓词、物化器串成 scan(path, columns, where)。"""

from .errors import InvalidPathError, NestedScanError
from .io_utils import FileView
from .materializer import (
    NodeReader,
    build_streams,
    mark_projection,
)
from .metadata import read_footer
from .predicate import (
    TRUE,
    collect_predicate_leaves,
    compile_predicate,
    evaluate,
)
from .schema import build_schema, resolve_path


def scan_file(path, columns, where):
    # ---- 入参静态校验 -------------------------------------------------
    if not isinstance(columns, list) or not columns:
        raise InvalidPathError("columns 必须是非空逻辑叶路径列表")
    norm_columns = []
    for col in columns:
        if not isinstance(col, list) or not col:
            raise InvalidPathError(f"列路径必须是非空数组，实际 {col!r}")
        norm_columns.append(list(col))

    view = FileView(path)
    try:
        metadata, _ = read_footer(view)
        root = build_schema(metadata.schema)

        # ---- 解析投影列与谓词，汇总所需物理叶 -------------------------
        projected_leaves = []
        seen = set()
        for path in norm_columns:
            leaf = resolve_path(root, path)
            if leaf.index not in seen:
                seen.add(leaf.index)
                projected_leaves.append(leaf)

        compiled = compile_predicate(where, root)
        predicate_leaves = (
            collect_predicate_leaves(compiled) if compiled is not None else []
        )
        needed_leaves = list(projected_leaves)
        for leaf in predicate_leaves:
            if leaf.index not in seen:
                seen.add(leaf.index)
                needed_leaves.append(leaf)

        projected_set = set(leaf.index for leaf in projected_leaves)

        # ---- 逐行组扫描 ----------------------------------------------
        for rg_index, row_group in enumerate(metadata.row_groups):
            if row_group.num_rows is None:
                raise NestedScanError("行组缺少 num_rows")
            streams = build_streams(view, row_group, needed_leaves)
            try:
                # projected 标记决定输出裁剪；物化阶段保留全部所需叶，
                # 使谓词专用叶也能参与三值逻辑求值。
                mark_projection(root, projected_set)
                reader = _PrunedNodeReader(root, streams)
                for _ in range(row_group.num_rows):
                    full_row = reader.read_row()
                    if compiled is None or evaluate(compiled, full_row) == TRUE:
                        yield _project_view(root, full_row)
                for stream in streams.values():
                    stream.close()
            except BaseException:
                for stream in streams.values():
                    try:
                        stream.close()
                    except NestedScanError:
                        pass
                raise
    finally:
        view.close()


def _project_view(root, full_row):
    """按 schema 的 projected 标记递归剥离谓词专用叶，只保留投影结构。

    物化行为求值方便而包含了所有“所需叶”；这里把未投影的分支移除，
    保留投影叶及其祖先、列表位置、空列表与 null。
    """
    return _prune_row(root, full_row)


def _prune_row(node, value):
    if value is None:
        return None
    if node.kind == "leaf":
        return value
    if node.kind == "list":
        if not getattr(node, "projected", True):
            return None
        element = node.element_node
        result = []
        for elem in value:
            if elem is None:
                result.append(None)
                continue
            if element.kind == "struct":
                # struct 元素本身是 dict
                result.append(_prune_row(element, elem))
            else:
                # leaf / 嵌套 list 元素被物化为 {元素名: 内部值}
                inner = elem[element.name]
                if element.kind == "leaf":
                    result.append({element.name: inner})
                else:
                    result.append({element.name: _prune_row(element, inner)})
        return result
    if isinstance(value, dict):
        out = {}
        for child in node.children:
            if child.name in value and getattr(child, "projected", True):
                out[child.name] = _prune_row(child, value[child.name])
        return out
    return value


class _PrunedNodeReader(NodeReader):
    """带结构裁剪的节点读取器。

    所有层级共享同一个全量 streams 映射（以全局 leaf.index 为键），
    裁剪只体现在 children 列表：不含任何所需叶的子树不会建立读取器，
    因而其物理记录也不会被消费。
    """

    def __init__(self, node, all_streams):
        self.node = node
        self.kind = node.kind
        # 直接共享全量映射，保证 leaf.index 全局可索引
        self.streams = all_streams
        self.projected = getattr(node, "projected", True)
        if node.kind in ("struct", "root"):
            self.children = [
                _PrunedNodeReader(c, all_streams) for c in node.children
                if _subtree_needed(c, all_streams)
            ]
        elif node.kind == "list":
            self.element_reader = _PrunedNodeReader(node.element_node, all_streams)
        else:
            self.children = []


def _subtree_needed(node, all_streams):
    return any(leaf.index in all_streams for leaf in node.leaf_children)
