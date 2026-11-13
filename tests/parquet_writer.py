"""测试夹具：仅用标准库生成嵌套 Parquet 文件。

这不是产品代码的一部分，而是为了在没有 pyarrow/pyspark 的环境里验证
nestedscan 自解码的正确性。支持：

* 嵌套 optional/required struct 与标准三层 LIST（可多层嵌套）；
* INT64 / UTF8 叶；
* PLAIN 与 RLE_DICTIONARY（含旧 PLAIN_DICTIONARY）数据页；
* DataPage V1 / V2；
* 多个小数据页（把列表/行切断到不同页面）；
* 受控损坏（截断、非法级值、字典越界、压缩页）。
"""

import json
import os
import struct

from nestedscan.metadata import (
    CONV_LIST,
    CONV_UTF8,
    ENC_PLAIN,
    ENC_PLAIN_DICTIONARY,
    ENC_RLE,
    ENC_RLE_DICTIONARY,
    CODEC_UNCOMPRESSED,
    PAGE_DATA,
    PAGE_DATA_V2,
    PAGE_DICTIONARY,
    REP_OPTIONAL,
    REP_REPEATED,
    REP_REQUIRED,
    TYPE_BYTE_ARRAY,
    TYPE_INT64,
)
from nestedscan.thrift import (
    CT_BINARY,
    CT_BYTE,
    CT_I32,
    CT_I64,
    CT_LIST,
    CT_STRUCT,
    CT_TRUE,
    CompactWriter,
)

# ---------------------------------------------------------------------
# 逻辑 schema 描述
# ---------------------------------------------------------------------

def leaf(name, physical_type, repetition=REP_OPTIONAL, utf8=False):
    return {"kind": "leaf", "name": name, "type": physical_type,
            "rep": repetition, "utf8": utf8}


def struct_node(name, children, repetition=REP_OPTIONAL):
    return {"kind": "struct", "name": name, "rep": repetition,
            "children": children}


def list3(name, element, repetition=REP_OPTIONAL, list_name="list"):
    """标准三层 LIST：optional group LIST { repeated group list { element } }。"""
    return {"kind": "list", "name": name, "rep": repetition,
            "list_name": list_name, "element": element}


# ---------------------------------------------------------------------
# 行 -> 物理记录（rl, dl, value）
# ---------------------------------------------------------------------

def flatten_rows(schema, rows):
    """把逻辑行集合展开为 {leaf_name: [(rl, dl, value), ...]}。

    value 仅在 dl==max_def 时有效，其余位置放 None。
    """
    max_def, max_rep = {}, {}
    _compute_levels(schema, 0, 0, max_def, max_rep)
    columns = {}
    _collect_leaves(schema, columns)

    for row in rows:
        _emit(schema, row, columns, first_rl=0,
              cur_def=0, cur_rep=0, max_def=max_def, max_rep=max_rep,
              ancestor_rls=[])
    return columns, max_def, max_rep


def _compute_levels(node, cur_def, cur_rep, max_def, max_rep):
    if node["kind"] == "root":
        max_def[node["name"]] = cur_def
        max_rep[node["name"]] = cur_rep
        for c in node["children"]:
            _compute_levels(c, cur_def, cur_rep, max_def, max_rep)
        return
    add_def = 1 if node["rep"] in (REP_OPTIONAL, REP_REPEATED) else 0
    add_rep = 1 if node["rep"] == REP_REPEATED else 0
    cur_def += add_def
    cur_rep += add_rep
    max_def[node["name"]] = cur_def
    max_rep[node["name"]] = cur_rep
    if node["kind"] == "leaf":
        return
    if node["kind"] == "list":
        # 标准三层：走到这里 cur_def/cur_rep 已计入 wrapper 自身。
        # 隐式 repeated group 再贡献一级 def 与一级 rep。
        repeated_def = cur_def + 1
        repeated_rep = cur_rep + 1
        _compute_levels(node["element"], repeated_def, repeated_rep,
                        max_def, max_rep)
    else:
        for c in node["children"]:
            _compute_levels(c, cur_def, cur_rep, max_def, max_rep)


def _collect_leaves(node, columns):
    if node["kind"] == "leaf":
        columns[node["name"]] = []
        return
    if node["kind"] == "list":
        _collect_leaves(node["element"], columns)
    else:
        for c in node.get("children", []):
            _collect_leaves(c, columns)


def _emit(node, value, columns, first_rl, cur_def, cur_rep,
          max_def, max_rep, ancestor_rls):
    if node["kind"] == "root":
        for c in node["children"]:
            _emit(c, value.get(c["name"]), columns, first_rl,
                  cur_def, cur_rep, max_def, max_rep, ancestor_rls)
        return

    add_def = 1 if node["rep"] in (REP_OPTIONAL, REP_REPEATED) else 0
    add_rep = 1 if node["rep"] == REP_REPEATED else 0

    if node["kind"] == "leaf":
        # 入口 cur_def 是叶子“自身累加前”的级别。
        # 非空：叶子有定义，DL = cur_def + add_def（应等于 max_def）。
        # null：叶子未定义，DL 停在 cur_def。
        if value is None:
            columns[node["name"]].append((first_rl, cur_def, None))
        else:
            columns[node["name"]].append(
                (first_rl, cur_def + add_def, value))
        return

    if node["kind"] == "struct":
        nd = cur_def + add_def
        nr = cur_rep + add_rep
        if value is None:
            # null struct：每个后代叶产出一条当前级值的占位
            _emit_null_placeholders(node, columns, first_rl, cur_def,
                                    max_def, max_rep, ancestor_rls)
            return
        for c in node["children"]:
            _emit(c, value.get(c["name"]), columns, first_rl,
                  nd, nr, max_def, max_rep, ancestor_rls)
        return

    # list（标准三层）：与 leaf/struct 不同，list 分支在此显式应用
    # wrapper 自身的 add_def/add_rep，再为隐式 repeated group 各加一级。
    wrapper_def = cur_def + add_def
    wrapper_rep = cur_rep + add_rep
    repeated = node["element"]
    list_rl = wrapper_rep + 1
    repeated_def = wrapper_def + 1
    if value is None:
        # 列表 null：wrapper 自身未定义，占位 DL 停在 wrapper 之前一级
        null_def = cur_def  # 尚未加 wrapper 的 add_def
        _emit_null_placeholders(node, columns, first_rl, null_def,
                                max_def, max_rep, ancestor_rls)
        return
    if len(value) == 0:
        # 空列表：wrapper 有定义但无 repeated 记录，每个叶一条
        # DL=wrapper 定义级 的占位记录
        _emit_empty_list_placeholders(node, columns, first_rl, wrapper_def,
                                      max_def, max_rep)
        return
    for i, element in enumerate(value):
        elem_rl = first_rl if i == 0 else list_rl
        _emit_list_element(repeated, element, columns, elem_rl,
                           repeated_def, list_rl, max_def, max_rep,
                           ancestor_rls + [list_rl])


def _emit_empty_list_placeholders(node, columns, rl, dl, max_def, max_rep):
    """空列表：为节点下每个叶产出一条停在 wrapper 定义级的占位记录。"""
    if node["kind"] == "leaf":
        columns[node["name"]].append((rl, dl, None))
    elif node["kind"] == "list":
        _emit_empty_list_placeholders(node["element"], columns, rl, dl,
                                      max_def, max_rep)
    else:
        for c in node.get("children", []):
            _emit_empty_list_placeholders(c, columns, rl, dl,
                                          max_def, max_rep)


def _emit_list_element(node, value, columns, elem_rl, cur_def, cur_rep,
                       max_def, max_rep, ancestor_rls):
    """node 是 repeated group 的唯一子节点（元素）。

    cur_def/cur_rep 已计入 wrapper 与隐式 repeated group 两层，是元素节点
    “自身累加之前”的级别。元素自身的 repetition 级别在本函数内显式累加。
    """
    add_def = 1 if node["rep"] in (REP_OPTIONAL, REP_REPEATED) else 0
    add_rep = 1 if node["rep"] == REP_REPEATED else 0

    if node["kind"] == "leaf":
        leaf_value = value.get(node["name"]) if isinstance(value, dict) else value
        if leaf_value is None:
            # null 元素：DL 停在元素自身之前（repeated 层）
            columns[node["name"]].append((elem_rl, cur_def, None))
        else:
            columns[node["name"]].append((elem_rl, cur_def + add_def,
                                          leaf_value))
        return
    # struct / 嵌套 list：struct 元素自身还有一级 repetition，先累加，
    # 再以“子节点自身累加前”的级别下发，与 _emit 的入口语义一致。
    if node["kind"] == "struct":
        # 元素 struct 在逻辑数据里被包成 {节点名: 字段dict}；
        # null 元素时该值为 None。
        struct_value = value.get(node["name"]) if isinstance(value, dict) else value
        nd = cur_def + add_def
        nr = cur_rep + add_rep
        if struct_value is None:
            _emit_null_placeholders(node, columns, elem_rl, cur_def,
                                    max_def, max_rep, ancestor_rls)
            return
        for c in node["children"]:
            _emit(c, struct_value.get(c["name"]), columns, elem_rl,
                  nd, nr, max_def, max_rep, ancestor_rls)
        return
    # 嵌套 list：作为外层 repeated 组的元素时，逻辑数据被包成
    # {内层list名: 列表值}，先解包再交给 _emit。
    list_value = value.get(node["name"]) if isinstance(value, dict) else value
    _emit(node, list_value, columns, elem_rl, cur_def, cur_rep,
          max_def, max_rep, ancestor_rls)


def _emit_null_placeholders(node, columns, rl, cur_def, max_def, max_rep,
                            ancestor_rls):
    if node["kind"] == "leaf":
        columns[node["name"]].append((rl, cur_def, None))
    elif node["kind"] == "list":
        _emit_null_placeholders(node["element"], columns, rl, cur_def,
                                max_def, max_rep, ancestor_rls)
    else:
        for c in node.get("children", []):
            _emit_null_placeholders(c, columns, rl, cur_def,
                                    max_def, max_rep, ancestor_rls)


# ---------------------------------------------------------------------
# 编码：RLE hybrid / PLAIN
# ---------------------------------------------------------------------

def bit_width(max_value):
    if max_value <= 0:
        return 0
    w = 0
    while (1 << w) <= max_value:
        w += 1
    return w


def encode_rle_hybrid(values, width):
    """RLE/bit-packing hybrid 编码（测试夹具用）。

    采用合法的纯 bit-packing 子集：所有值按 8 个一组 LSB-first 打包，
    最后一组不足 8 个时补零（填充位由解码器按 num_values 忽略）。
    不产生 RLE run，从而彻底避免“填充出现在流中间”的歧义。
    width==0 时每个组占 0 字节，仅靠 header 累计组数。
    """
    out = bytearray()
    pos = 0
    n = len(values)
    while pos < n:
        group = values[pos:pos + 8]
        if len(group) < 8:
            group = group + [0] * (8 - len(group))
        out.append(1 << 1)  # header：1 个 bit-packing 组
        if width > 0:
            _write_bitpacked(out, group, width)
        pos += 8
    return bytes(out)


def _write_varint(buf, n):
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            buf.append(b | 0x80)
        else:
            buf.append(b)
            return


def _write_packed(buf, value, width):
    nbytes = (width + 7) // 8
    for i in range(nbytes):
        buf.append((value >> (8 * i)) & 0xFF)


def _write_bitpacked(buf, values8, width):
    """写一个 bit-packing 组（8 个 width 位值），共 width 字节。

    与解码端对称：把 8 个值按低位优先拼成一个整数，再逐字节小端写出。
    """
    acc = 0
    for i, v in enumerate(values8):
        acc |= (v & ((1 << width) - 1)) << (i * width)
    for _ in range(width):
        buf.append(acc & 0xFF)
        acc >>= 8


def encode_levels_v1(values, max_level):
    # 与 Parquet 规范一致：max_level==0 时整个级值列省略（空体 + 长度 0）
    if max_level == 0:
        return struct.pack("<I", 0)
    body = encode_rle_hybrid(values, bit_width(max_level))
    return struct.pack("<I", len(body)) + body


def encode_plain_int64(values):
    return b"".join(struct.pack("<q", v) for v in values)


def encode_plain_bytearrays(values):
    out = bytearray()
    for s in values:
        data = s.encode("utf-8")
        out += struct.pack("<i", len(data))
        out += data
    return bytes(out)


def encode_plain(leaf_spec, values):
    if leaf_spec["type"] == TYPE_INT64:
        return encode_plain_int64(values)
    return encode_plain_bytearrays(values)


# ---------------------------------------------------------------------
# Thrift 元数据写出
# ---------------------------------------------------------------------

def _field(w, fid, ctype, last, writer):
    w.write_field_header(fid, ctype, last)
    writer(w)
    return fid


def write_schema_elements(w, root_name, schema):
    """按 Parquet 先序遍历生成扁平 SchemaElement 描述列表。"""
    elements = []

    def emit(node, is_root=False):
        if is_root:
            elements.append({
                "type": None, "type_length": None, "rep": None,
                "name": node["name"],
                "num_children": len(node["children"]),
                "conv": None, "logical": None,
            })
            for c in node["children"]:
                emit(c)
            return

        if node["kind"] == "leaf":
            elements.append({
                "type": node["type"], "type_length": None,
                "rep": node["rep"], "name": node["name"],
                "num_children": None,
                "conv": CONV_UTF8 if node.get("utf8") else None,
                "logical": "STRING" if node.get("utf8") else None,
            })
        elif node["kind"] == "struct":
            elements.append({
                "type": None, "type_length": None, "rep": node["rep"],
                "name": node["name"], "num_children": len(node["children"]),
                "conv": None, "logical": None,
            })
            for c in node["children"]:
                emit(c)
        elif node["kind"] == "list":
            # 第一层：带 LIST 注解的 wrapper group
            elements.append({
                "type": None, "type_length": None, "rep": node["rep"],
                "name": node["name"], "num_children": 1,
                "conv": CONV_LIST, "logical": "LIST",
            })
            # 第二层：repeated group<list-name>
            elements.append({
                "type": None, "type_length": None, "rep": REP_REPEATED,
                "name": node["list_name"], "num_children": 1,
                "conv": None, "logical": None,
            })
            # 第三层：元素
            emit(node["element"])

    root = {"kind": "root", "name": root_name,
            "children": schema if isinstance(schema, list)
            else schema["children"]}
    emit(root, is_root=True)
    return elements


def serialize_schema_element(el):
    w = CompactWriter()
    last = 0
    if el["type"] is not None:
        last = _field(w, 1, CT_I32, last, lambda x: x.write_zigzag(el["type"]))
    if el["type_length"] is not None:
        last = _field(w, 2, CT_I32, last,
                      lambda x: x.write_zigzag(el["type_length"]))
    if el["rep"] is not None:
        last = _field(w, 3, CT_I32, last, lambda x: x.write_zigzag(el["rep"]))
    last = _field(w, 4, CT_BINARY, last, lambda x: x.write_string(el["name"]))
    if el["num_children"] is not None:
        last = _field(w, 5, CT_I32, last,
                      lambda x: x.write_zigzag(el["num_children"]))
    if el["conv"] is not None:
        last = _field(w, 6, CT_I32, last, lambda x: x.write_zigzag(el["conv"]))
    if el["logical"] is not None:
        w.write_field_header(10, CT_STRUCT, last)
        # LogicalType union：内部字段 id 从 0 开始，可能 >15，
        # 必须用 delta=0 + zigzag 绝对 id 的非紧凑字段头
        union_fid = 1 if el["logical"] == "STRING" else 3
        w.parts.append(bytes([CT_STRUCT]))          # delta=0, type=struct
        w.write_zigzag(union_fid)                    # 绝对字段 id
        w.write_struct_stop()                        # 空 union 成员的 STOP
        w.write_struct_stop()                        # union 本身的 STOP
    w.write_struct_stop()
    return w.getvalue()


def serialize_logical_type(name):  # pragma: no cover - 已内联
    pass


def write_page_header(ph):
    w = CompactWriter()
    last = 0
    last = _field(w, 1, CT_I32, last, lambda x: x.write_zigzag(ph["type"]))
    last = _field(w, 2, CT_I32, last,
                  lambda x: x.write_zigzag(ph["uncompressed"]))
    last = _field(w, 3, CT_I32, last,
                  lambda x: x.write_zigzag(ph["compressed"]))
    if ph["type"] == PAGE_DATA:
        h = ph["v1"]
        w.write_field_header(5, CT_STRUCT, last)
        hw = CompactWriter()
        hl = 0
        hl = _field(hw, 1, CT_I32, hl, lambda x: x.write_zigzag(h["num_values"]))
        hl = _field(hw, 2, CT_I32, hl, lambda x: x.write_zigzag(h["encoding"]))
        hl = _field(hw, 3, CT_I32, hl, lambda x: x.write_zigzag(ENC_RLE))
        hl = _field(hw, 4, CT_I32, hl, lambda x: x.write_zigzag(ENC_RLE))
        hw.write_struct_stop()
        w.parts.append(hw.getvalue())
    elif ph["type"] == PAGE_DATA_V2:
        h = ph["v2"]
        w.write_field_header(8, CT_STRUCT, last)
        hw = CompactWriter()
        hl = 0
        hl = _field(hw, 1, CT_I32, hl, lambda x: x.write_zigzag(h["num_values"]))
        hl = _field(hw, 2, CT_I32, hl, lambda x: x.write_zigzag(h["num_nulls"]))
        hl = _field(hw, 3, CT_I32, hl, lambda x: x.write_zigzag(h["num_rows"]))
        hl = _field(hw, 4, CT_I32, hl, lambda x: x.write_zigzag(h["encoding"]))
        hl = _field(hw, 5, CT_I32, hl,
                    lambda x: x.write_zigzag(h["dl_bytes"]))
        hl = _field(hw, 6, CT_I32, hl,
                    lambda x: x.write_zigzag(h["rl_bytes"]))
        if h.get("is_compressed"):
            hw.write_field_header(7, CT_TRUE, hl)
        hw.write_struct_stop()
        w.parts.append(hw.getvalue())
    elif ph["type"] == PAGE_DICTIONARY:
        h = ph["dict"]
        w.write_field_header(7, CT_STRUCT, last)
        hw = CompactWriter()
        hl = 0
        hl = _field(hw, 1, CT_I32, hl, lambda x: x.write_zigzag(h["num_values"]))
        hl = _field(hw, 2, CT_I32, hl, lambda x: x.write_zigzag(h["encoding"]))
        hw.write_struct_stop()
        w.parts.append(hw.getvalue())
    w.write_struct_stop()
    return w.getvalue()


def serialize_column_metadata(md, physical_path):
    w = CompactWriter()
    last = 0
    last = _field(w, 1, CT_I32, last, lambda x: x.write_zigzag(md["type"]))
    # 2: encodings list<i32> —— 字段头与 list 头合二为一
    w.write_field_header(2, CT_LIST, last)
    w.write_list_header(CT_I32, len(md["encodings"]))
    for enc in md["encodings"]:
        w.write_zigzag(enc)
    last = 2
    # 3: path_in_schema list<string>
    w.write_field_header(3, CT_LIST, last)
    w.write_list_header(CT_BINARY, len(physical_path))
    for name in physical_path:
        w.write_string(name)
    last = 3
    last = _field(w, 4, CT_I32, last,
                  lambda x: x.write_zigzag(CODEC_UNCOMPRESSED))
    last = _field(w, 5, CT_I64, last, lambda x: x.write_zigzag(md["num_values"]))
    last = _field(w, 6, CT_I64, last,
                  lambda x: x.write_zigzag(md["total_uncompressed"]))
    last = _field(w, 7, CT_I64, last,
                  lambda x: x.write_zigzag(md["total_compressed"]))
    if md.get("dict_offset") is not None:
        last = _field(w, 11, CT_I64, last,
                      lambda x: x.write_zigzag(md["dict_offset"]))
    last = _field(w, 9, CT_I64, last,
                  lambda x: x.write_zigzag(md["data_offset"]))
    w.write_struct_stop()
    return w.getvalue()


def serialize_row_group(columns_md, num_rows, total_size):
    w = CompactWriter()
    # 1: list<ColumnChunk> —— 字段头与 list 头合二为一
    w.write_field_header(1, CT_LIST, 0)
    w.write_list_header(CT_STRUCT, len(columns_md))
    for md in columns_md:
        cw = CompactWriter()
        cw.write_field_header(2, CT_I64, 0)
        cw.write_zigzag(md["data_offset"])
        cw.write_field_header(3, CT_STRUCT, 2)
        cw.parts.append(serialize_column_metadata(
            md, md["physical_path"]
        ))
        cw.write_struct_stop()
        w.parts.append(cw.getvalue())
    last = 1
    last = _field(w, 2, CT_I64, last, lambda x: x.write_zigzag(total_size))
    last = _field(w, 3, CT_I64, last, lambda x: x.write_zigzag(num_rows))
    w.write_struct_stop()
    return w.getvalue()


def serialize_footer(version, schema_elements, row_groups_buf, num_rows):
    w = CompactWriter()
    last = 0
    last = _field(w, 1, CT_I32, last, lambda x: x.write_zigzag(version))
    # 2: list<SchemaElement> —— 字段头与 list 头合二为一
    w.write_field_header(2, CT_LIST, last)
    w.write_list_header(CT_STRUCT, len(schema_elements))
    for el in schema_elements:
        w.parts.append(serialize_schema_element(el))
    last = 2
    last = _field(w, 3, CT_I64, last, lambda x: x.write_zigzag(num_rows))
    # 4: list<RowGroup>
    w.write_field_header(4, CT_LIST, last)
    w.write_list_header(CT_STRUCT, len(row_groups_buf))
    for buf in row_groups_buf:
        w.parts.append(buf)
    w.write_struct_stop()
    return w.getvalue()


# ---------------------------------------------------------------------
# 列块写出
# ---------------------------------------------------------------------

def _chunk_records(records, page_value_cap):
    """按值数切页；保证切在记录边界（可切断一行/列表）。"""
    pages = []
    i = 0
    while i < len(records):
        pages.append(records[i:i + page_value_cap])
        i += page_value_cap
    return pages


def write_column_chunk(fout, leaf_spec, records, max_def, max_rep,
                       page_value_cap, page_version, use_dict,
                       corrupt=None):
    """写出一个列块，返回该列的 ColumnMetaData 描述。"""
    dict_offset = None
    dict_values = None
    encodings = [ENC_RLE]
    data_encoding = ENC_PLAIN

    non_null = [v for (_, dl, v) in records if dl == max_def]
    if use_dict:
        data_encoding = ENC_RLE_DICTIONARY
        # 字典按首次出现顺序去重
        seen = {}
        dict_values = []
        for v in non_null:
            if v not in seen:
                seen[v] = len(dict_values)
                dict_values.append(v)
        encodings = [ENC_RLE, ENC_RLE_DICTIONARY, ENC_PLAIN]
    else:
        encodings = [ENC_RLE, ENC_PLAIN]

    # ---- 字典页 -------------------------------------------------------
    if dict_values is not None:
        dict_offset = fout.tell()
        dict_payload = encode_plain(leaf_spec, dict_values)
        header = write_page_header({
            "type": PAGE_DICTIONARY,
            "uncompressed": len(dict_payload),
            "compressed": len(dict_payload),
            "dict": {"num_values": len(dict_values), "encoding": ENC_PLAIN},
        })
        fout.write(header)
        fout.write(dict_payload)

    data_offset = fout.tell()
    pages = _chunk_records(records, page_value_cap)
    total_size = 0

    for page_records in pages:
        rls = [r[0] for r in page_records]
        dls = [r[1] for r in page_records]
        page_values = [r[2] for r in page_records if r[1] == max_def]
        num_nulls = len(page_records) - len(page_values)

        if page_version == "v1":
            payload = bytearray()
            payload += encode_levels_v1(rls, max_rep)
            payload += encode_levels_v1(dls, max_def)
            if data_encoding == ENC_PLAIN:
                payload += encode_plain(leaf_spec, page_values)
            else:
                width = bit_width(max(len(dict_values) - 1, 0))
                payload.append(width)
                indices = [seen[v] for v in page_values]
                payload += encode_rle_hybrid(indices, width)
            ph = {
                "type": PAGE_DATA,
                "uncompressed": len(payload),
                "compressed": len(payload),
                "v1": {"num_values": len(page_records),
                       "encoding": data_encoding},
            }
            header = write_page_header(ph)
            fout.write(header)
            fout.write(bytes(payload))
            total_size += len(header) + len(payload)
        else:
            rl_body = b"" if max_rep == 0 else encode_rle_hybrid(
                rls, bit_width(max_rep))
            dl_body = b"" if max_def == 0 else encode_rle_hybrid(
                dls, bit_width(max_def))
            value_part = bytearray()
            if data_encoding == ENC_PLAIN:
                value_part += encode_plain(leaf_spec, page_values)
            else:
                width = bit_width(max(len(dict_values) - 1, 0))
                value_part.append(width)
                indices = [seen[v] for v in page_values]
                value_part += encode_rle_hybrid(indices, width)
            payload = rl_body + dl_body + bytes(value_part)
            ph = {
                "type": PAGE_DATA_V2,
                "uncompressed": len(payload),
                "compressed": len(payload),
                "v2": {"num_values": len(page_records),
                       "num_nulls": num_nulls,
                       "num_rows": _count_rows(rls),
                       "encoding": data_encoding,
                       "dl_bytes": len(dl_body),
                       "rl_bytes": len(rl_body),
                       "is_compressed": corrupt == "compressed"},
            }
            header = write_page_header(ph)
            fout.write(header)
            fout.write(bytes(payload))
            total_size += len(header) + len(payload)

    return {
        "type": leaf_spec["type"],
        "encodings": encodings,
        "num_values": len(records),
        "total_uncompressed": total_size,
        "total_compressed": total_size,
        "data_offset": data_offset,
        "dict_offset": dict_offset,
    }


def _count_rows(rls):
    # 页内新顶层行：rl==0；空页按 0 行
    return sum(1 for rl in rls if rl == 0)


def physical_path(schema_node, leaf_name):
    """还原叶列的 path_in_schema（含 list wrapper 与 repeated group 名）。"""

    def walk(node, prefix):
        if node["kind"] == "leaf":
            return prefix + [node["name"]] if node["name"] == leaf_name else None
        if node["kind"] == "root":
            for c in node["children"]:
                found = walk(c, [])
                if found is not None:
                    return found
            return None
        if node["kind"] == "list":
            base = prefix + [node["name"], node["list_name"]]
            return walk(node["element"], base)
        # struct
        for c in node.get("children", []):
            found = walk(c, prefix + [node["name"]])
            if found is not None:
                return found
        return None

    result = walk(schema_node, [])
    return result if result is not None else []


def write_parquet(path, root_name, schema, rows, *,
                  page_value_cap=100000, page_version="v1",
                  use_dict=False, corrupt=None, version=1,
                  row_group_rows=None):
    """生成完整 Parquet 文件。

    row_group_rows 给定时按该大小把 rows 切成多个行组；corrupt:
    None/'truncate'/'illegal_level'/'compressed'。
    """
    root = {"kind": "root", "name": root_name, "children": schema}
    leaf_specs = {}
    _index_leaves(root, leaf_specs)
    schema_elements = write_schema_elements(None, root_name, schema)

    if row_group_rows is None:
        row_groups = [rows]
    else:
        row_groups = [rows[i:i + row_group_rows]
                      for i in range(0, len(rows), row_group_rows)]

    with open(path, "wb") as fout:
        fout.write(b"PAR1")
        row_group_bufs = []
        total_rows = 0
        for rg_rows in row_groups:
            columns, max_def, max_rep = flatten_rows(root, rg_rows)
            columns_md = []
            total_size = 0
            for leaf_name, records in columns.items():
                spec = leaf_specs[leaf_name]
                md = write_column_chunk(
                    fout, spec, records,
                    max_def[leaf_name], max_rep[leaf_name],
                    page_value_cap, page_version, use_dict,
                    corrupt=corrupt,
                )
                md["physical_path"] = physical_path(root, leaf_name)
                columns_md.append(md)
                total_size += md["total_compressed"]
            row_group_bufs.append(
                serialize_row_group(columns_md, len(rg_rows), total_size))
            total_rows += len(rg_rows)

        footer = serialize_footer(version, schema_elements,
                                  row_group_bufs, total_rows)
        footer_offset = fout.tell()
        fout.write(footer)
        fout.write(struct.pack("<I", len(footer)))
        fout.write(b"PAR1")
        end = fout.tell()

    if corrupt == "truncate":
        with open(path, "r+b") as f:
            f.truncate(end - 9)  # 砍掉尾魔数与部分长度
    elif corrupt == "illegal_level":
        _inject_illegal_level(path)
    return end


def _inject_illegal_level(path):
    """字节级注入非法定义级：定位第一个数据页的 DL 流，把第一个级值
    数据字节改成全 1，使按 schema 位宽解出的 DL 超过该叶 max_def。

    保持页大小、级值长度与非空值计数不变，只破坏级值取值范围。
    """
    from nestedscan.metadata import parse_page_header, PAGE_DATA
    with open(path, "r+b") as f:
        f.seek(0, 2)
        file_size = f.tell()
        offset = 4  # 跳过文件头魔数
        ph, hsize = parse_page_header(
            _read_at(f, offset, min(64, file_size - offset)), offset)
        # 跳过可能存在的字典页
        while ph.type != PAGE_DATA:
            offset += hsize + ph.compressed_page_size
            ph, hsize = parse_page_header(
                _read_at(f, offset, min(64, file_size - offset)), offset)
        payload_start = offset + hsize
        payload = bytearray(_read_at(f, payload_start,
                                     ph.compressed_page_size))
        pos = 0
        rl_len = struct.unpack_from("<I", payload, pos)[0]
        pos += 4 + rl_len
        dl_len = struct.unpack_from("<I", payload, pos)[0]
        pos += 4
        dl_body = payload[pos:pos + dl_len]
        if dl_len <= 1:
            raise RuntimeError("目标列没有可注入的定义级数据字节")
        # 首字节是 hybrid 头，其后第一个级值数据字节置全 1
        dl_body[1] = 0xFF
        payload[pos:pos + dl_len] = dl_body
        f.seek(payload_start)
        f.write(bytes(payload))


def _read_at(f, offset, count):
    f.seek(offset)
    return f.read(count)


def _index_leaves(node, out):
    if node["kind"] == "leaf":
        out[node["name"]] = node
    elif node["kind"] == "list":
        _index_leaves(node["element"], out)
    else:
        for c in node.get("children", []):
            _index_leaves(c, out)
