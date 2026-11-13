"""嵌套结构物化器。

依据 Parquet 的重复级（repetition level）/定义级（definition level）
把若干物理叶列的扁平记录流还原为嵌套 dict/list：

* 叶子读取器 :class:`LeafStream` 封装页边界，行/列表跨页时自动续页；
* 节点读取器 :class:`NodeReader` 递归消费子树，列表元素按重复级切分，
  空列表与 null 都由定义级直接区分，不裁掉任何 null 元素；
* 投影裁剪只保留被选叶所在的祖先分支，谓词专用叶不进入输出。
"""

from .errors import CorruptMetadataError, IllegalLevelError, TruncatedFileError
from .pages import LeafReader


class LeafStream:
    """叶列记录流；对上层提供 peek/consume。"""

    __slots__ = ("reader", "_rl", "_dl", "_value", "_loaded", "_delivered")

    def __init__(self, view, chunk, leaf):
        self.reader = LeafReader(view, chunk, leaf)
        self._rl = None
        self._dl = None
        self._value = None
        self._loaded = False
        # 已通过 consume 真正交付给上层的记录数（预取不计入）
        self._delivered = 0

    def _ensure(self):
        if not self._loaded:
            self._rl, self._dl, self._value = self.reader.next_record()
            self._loaded = True

    def peek_rl(self):
        self._ensure()
        return self._rl

    def peek_dl(self):
        self._ensure()
        return self._dl

    def is_exhausted(self):
        """列块所有记录都已交付时为 True（已预取待消费的记录仍算未交付）。"""
        return self._delivered >= self.reader.expected_total

    def consume(self):
        self._ensure()
        rec = (self._rl, self._dl, self._value)
        self._loaded = False
        self._rl = self._dl = self._value = None
        self._delivered += 1
        return rec

    def close(self):
        if self._delivered != self.reader.expected_total:
            raise TruncatedFileError(
                f"列块实际交付 {self._delivered} 条，"
                f"元数据声明 {self.reader.expected_total} 条",
                self.reader.next_offset,
            )


def mark_projection(node, projected_indices):
    """给节点子树打 projected 标记：任一后代叶被投影则为 True。

    projected_indices 是被投影叶节点的 index 集合。
    """
    if node.kind == "leaf":
        node.projected = node.index in projected_indices
        return node.projected
    if node.kind == "list":
        node.projected = mark_projection(node.element_node, projected_indices)
        return node.projected
    any_proj = False
    for child in node.children:
        if mark_projection(child, projected_indices):
            any_proj = True
    node.projected = any_proj
    return any_proj


class NodeReader:
    """按 schema 节点递归物化；streams 为 {叶节点 index: LeafStream}。"""

    def __init__(self, node, streams):
        self.node = node
        self.streams = streams
        self.kind = node.kind
        # 子节点读取器（list 的元素读取器单独保存）
        if node.kind == "struct" or node.kind == "root":
            self.children = [
                NodeReader(c, streams) for c in node.children
            ]
        elif node.kind == "list":
            self.element_reader = NodeReader(node.element_node, streams)
        self.projected = getattr(node, "projected", True)

    # -----------------------------------------------------------------
    def read_row(self):
        """读取一个顶层行（根节点的一次实例）。"""
        return self._read_instance(first_rl=0)

    def _read_instance(self, first_rl):
        """读取当前节点的一个实例。

        first_rl: 该实例第一条物理记录的重复级（由父结构 peek 得到）。
        """
        node = self.node
        if node.kind == "leaf":
            return self._read_leaf(first_rl)
        if node.kind in ("struct", "root"):
            return self._read_struct(first_rl)
        return self._read_list(first_rl)

    def _any_stream(self):
        for leaf in self.node.leaf_children:
            return self.streams[leaf.index]
        raise CorruptMetadataError("节点没有任何叶列", None)

    def _read_leaf(self, first_rl):
        stream = self.streams[self.node.index]
        rl, dl, value = stream.consume()
        if rl != first_rl:
            raise IllegalLevelError(
                f"叶 {self.node.name!r} 重复级 {rl} 与行首 {first_rl} 不一致",
                stream.reader.next_offset,
            )
        if dl < 0 or dl > self.node.max_def:
            raise IllegalLevelError(
                f"叶 {self.node.name!r} 定义级 {dl} 超出 max_def "
                f"{self.node.max_def}", stream.reader.next_offset,
            )
        if dl == self.node.max_def:
            return value
        return None

    def _read_struct(self, first_rl):
        """读取 struct/root 实例；null 时仍需消耗全部后代占位记录。"""
        node = self.node
        probe = self._any_stream()
        head_dl = probe.peek_dl()
        # 探针是后代叶列，其 DL 范围是叶的 max_def，不能用 struct 自身的
        leaf_max_def = probe.reader.max_def
        if head_dl < 0 or head_dl > leaf_max_def:
            raise IllegalLevelError(
                f"结构 {node.name!r} 探针定义级 {head_dl} 超出叶 max_def "
                f"{leaf_max_def}", probe.reader.next_offset,
            )
        is_null = (
            node.kind == "struct"
            and node.repetition is not None
            and head_dl < node.max_def
        )
        if is_null:
            # 消耗每个后代叶的一条占位记录，值与结构无关
            self._consume_null_placeholders(first_rl)
            return None
        result = {}
        for child_reader in self.children:
            # self.children 含全部“可达”子树（投影叶 + 谓词叶）；
            # 谓词专用叶也写入物化行供求值，输出时由引擎按 projected 裁剪。
            value = child_reader._read_instance(first_rl)
            result[child_reader.node.name] = value
        if node.kind == "root":
            return result
        return result

    def _consume_null_placeholders(self, expected_rl):
        for leaf in self.node.leaf_children:
            if leaf.index not in self.streams:
                continue
            stream = self.streams[leaf.index]
            rl, dl, _ = stream.consume()
            if rl != expected_rl:
                raise IllegalLevelError(
                    f"null 结构占位记录重复级 {rl}，期望 {expected_rl}",
                    stream.reader.next_offset,
                )

    def _read_list(self, first_rl):
        """读取三层 LIST 实例：null 列表 / 空列表 / 元素数组。

        每条物理记录的 DL 在标准三层下的含义：
          DL <  wrapper.max_def(=1) -> 列表为 null，一条占位记录，需消费
          DL == wrapper.max_def(=1) -> 空列表，一条占位记录，需消费
          DL >= repeated 层定义级(=2) -> 真实元素，按 RL 切分
        新行第一个元素的 RL 等于行首 RL（通常 0），后续同列表元素为 list_rl。
        """
        node = self.node
        repeated = node.repeated_group
        probe = self._any_stream()
        leaf_max_def = probe.reader.max_def
        head_dl = probe.peek_dl()
        if head_dl < 0 or head_dl > leaf_max_def:
            raise IllegalLevelError(
                f"列表 {node.name!r} 定义级 {head_dl} 超出叶 max_def "
                f"{leaf_max_def}", probe.reader.next_offset,
            )

        # null 列表：wrapper 未定义，消费每个叶流的一条占位记录
        if head_dl < node.list_wrapper.max_def:
            self._consume_null_placeholders(first_rl)
            return None

        # 空列表：wrapper 有定义但无 repeated 记录，同样消费一条占位
        if head_dl == node.list_wrapper.max_def:
            self._consume_null_placeholders(first_rl)
            return []

        # 非空列表：头记录一定属于本列表的第一个元素，先消费它；
        # 之后通过重复级判断后续记录是否仍属于本列表——
        # RL >= 本列表 repeated 层的 max_rep 表示同列表/更深嵌套的延续，
        # RL 回退到该阈值之下表示已到行或祖先结构边界。
        list_rl = repeated.max_rep
        elements = []

        def wrap(elem):
            # leaf / 嵌套 list 元素以字段名承载；struct 元素本身是 dict
            if self.node.element_node.kind in ("leaf", "list"):
                return {self.node.element_node.name: elem}
            return elem

        # 第一个元素
        elements.append(wrap(self.element_reader._read_instance(
            probe.peek_rl())))
        # 后续元素：探针仍有记录且重复级未回退到本列表层之下
        while not probe.is_exhausted() and probe.peek_rl() >= list_rl:
            cur_rl = probe.peek_rl()
            if cur_rl > list_rl and self.node.element_node.kind != "list":
                # 更深嵌套由内层 list 读取器处理；同层不应出现更高 RL
                raise IllegalLevelError(
                    f"列表 {node.name!r} 元素重复级 {cur_rl} 大于列表层 "
                    f"{list_rl}", probe.reader.next_offset,
                )
            elements.append(wrap(self.element_reader._read_instance(cur_rl)))
        return elements


def build_streams(view, row_group, needed_leaves):
    """为行组内需要的叶列建立 {leaf.index: LeafStream}，并校验列路径。"""
    streams = {}
    by_path = {}
    for chunk in row_group.columns:
        if chunk.meta_data is None:
            raise CorruptMetadataError("列块缺少元数据", None)
        by_path[tuple(chunk.meta_data.path_in_schema)] = chunk
    for leaf in needed_leaves:
        path = _physical_path(leaf)
        chunk = by_path.get(tuple(path))
        if chunk is None:
            raise CorruptMetadataError(
                f"行组缺少叶列 {path}", None
            )
        streams[leaf.index] = LeafStream(view, chunk, leaf)
    return streams


def _physical_path(leaf):
    """从根到叶的物理字段名序列（含 list wrapper 与 repeated group 名）。

    三层 LIST 的物理路径为 [wrapper 名, repeated group 名, 元素...]。
    element 的 parent 指向 list wrapper，回溯会成环，因此自顶向下搜索。
    """
    root = leaf.root_ref
    chain = []

    def search(node):
        if node is leaf:
            chain.append(node)
            return True
        if node.kind == "list":
            if search(node.element_node):
                chain.append(node)
                return True
            return False
        for child in node.children:
            if search(child):
                chain.append(node)
                return True
        return False

    search(root)
    chain.reverse()

    names = []
    for node in chain:
        if node.kind == "root":
            continue
        names.append(node.name)
        if node.kind == "list":
            names.append(node.repeated_group.name)
    return names
