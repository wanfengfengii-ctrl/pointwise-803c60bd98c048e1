"""Parquet 元数据解析：文件页脚、行组、列块、页头。

仅解析扫描所需字段；未知字段通过 Thrift skip 跳过，保持前向兼容。
枚举值按 parquet.thrift 中的编号保留为整数常量。
"""

import struct

from .errors import CorruptMetadataError, TruncatedFileError, UnsupportedFormatError
from .thrift import (
    CompactReader,
    CT_BINARY,
    CT_BYTE,
    CT_DOUBLE,
    CT_FALSE,
    CT_I16,
    CT_I32,
    CT_I64,
    CT_LIST,
    CT_MAP,
    CT_STRUCT,
    CT_TRUE,
)

PARQUET_MAGIC = b"PAR1"

# Type
TYPE_BOOLEAN = 0
TYPE_INT32 = 1
TYPE_INT64 = 2
TYPE_INT96 = 3
TYPE_FLOAT = 4
TYPE_DOUBLE = 5
TYPE_BYTE_ARRAY = 6
TYPE_FIXED_LEN_BYTE_ARRAY = 7

# FieldRepetitionType
REP_REQUIRED = 0
REP_OPTIONAL = 1
REP_REPEATED = 2

# ConvertedType
CONV_UTF8 = 0
CONV_LIST = 3

# Encoding
ENC_PLAIN = 0
ENC_PLAIN_DICTIONARY = 2
ENC_RLE = 3
ENC_BIT_PACKED = 4
ENC_RLE_DICTIONARY = 8

# CompressionCodec
CODEC_UNCOMPRESSED = 0

# PageType
PAGE_DATA = 0
PAGE_INDEX = 1
PAGE_DICTIONARY = 2
PAGE_DATA_V2 = 3


class SchemaElement:
    __slots__ = (
        "type", "type_length", "repetition_type", "name",
        "num_children", "converted_type", "scale", "precision",
        "field_id", "logical_type",
    )

    def __init__(self):
        self.type = None
        self.type_length = None
        self.repetition_type = None
        self.name = None
        self.num_children = None
        self.converted_type = None
        self.scale = None
        self.precision = None
        self.field_id = None
        # 字符串名，当前仅识别 "STRING"/"LIST"；其余为 None
        self.logical_type = None


class ColumnMetaData:
    __slots__ = (
        "type", "encodings", "path_in_schema", "codec", "num_values",
        "total_uncompressed_size", "total_compressed_size",
        "data_page_offset", "index_page_offset", "dictionary_page_offset",
    )

    def __init__(self):
        self.type = None
        self.encodings = []
        self.path_in_schema = []
        self.codec = None
        self.num_values = None
        self.total_uncompressed_size = None
        self.total_compressed_size = None
        self.data_page_offset = None
        self.index_page_offset = None
        self.dictionary_page_offset = None


class ColumnChunk:
    __slots__ = ("file_path", "file_offset", "meta_data")

    def __init__(self):
        self.file_path = None
        self.file_offset = None
        self.meta_data = None


class RowGroup:
    __slots__ = ("columns", "total_byte_size", "num_rows", "file_offset",
                 "total_compressed_size", "ordinal")

    def __init__(self):
        self.columns = []
        self.total_byte_size = None
        self.num_rows = None
        self.file_offset = None
        self.total_compressed_size = None
        self.ordinal = None


class FileMetaData:
    __slots__ = ("version", "schema", "num_rows", "row_groups",
                 "created_by", "key_value_metadata")

    def __init__(self):
        self.version = None
        self.schema = []
        self.num_rows = None
        self.row_groups = []
        self.created_by = None
        self.key_value_metadata = None


class DataPageHeader:
    __slots__ = ("num_values", "encoding", "definition_level_encoding",
                 "repetition_level_encoding")

    def __init__(self):
        self.num_values = None
        self.encoding = None
        self.definition_level_encoding = None
        self.repetition_level_encoding = None


class DataPageHeaderV2:
    __slots__ = ("num_values", "num_nulls", "num_rows", "encoding",
                 "definition_levels_byte_length",
                 "repetition_levels_byte_length", "is_compressed")

    def __init__(self):
        self.num_values = None
        self.num_nulls = None
        self.num_rows = None
        self.encoding = None
        self.definition_levels_byte_length = None
        self.repetition_levels_byte_length = None
        self.is_compressed = False


class DictionaryPageHeader:
    __slots__ = ("num_values", "encoding", "is_sorted")

    def __init__(self):
        self.num_values = None
        self.encoding = None
        self.is_sorted = False


class PageHeader:
    __slots__ = ("type", "uncompressed_page_size", "compressed_page_size",
                 "crc", "data_page_header", "dictionary_page_header",
                 "data_page_header_v2", "header_offset")

    def __init__(self):
        self.type = None
        self.uncompressed_page_size = None
        self.compressed_page_size = None
        self.crc = None
        self.data_page_header = None
        self.dictionary_page_header = None
        self.data_page_header_v2 = None
        self.header_offset = None


# ---------------------------------------------------------------------
# 基础读取小工具
# ---------------------------------------------------------------------

def _read_scalar(reader, ctype):
    """读取一个标量字段（bool/整数/浮点/字符串）。"""
    if ctype == CT_TRUE:
        return True
    if ctype == CT_FALSE:
        return False
    if ctype == CT_BYTE:
        return reader.read_byte()
    if ctype in (CT_I16, CT_I32, CT_I64):
        return reader.read_zigzag()
    if ctype == 7:  # CT_DOUBLE
        return reader.read_double()
    if ctype == CT_BINARY:
        return reader.read_string()
    raise CorruptMetadataError(
        f"标量字段的 compact 类型 {ctype} 不合法", reader.abs_offset
    )


def _expect_ctype(reader, ctype, expected, what):
    if ctype != expected:
        raise CorruptMetadataError(
            f"{what} 需要 compact 类型 {expected}，实际为 {ctype}",
            reader.abs_offset,
        )


def _read_i32_list(reader, ctype):
    _expect_ctype(reader, ctype, CT_LIST, "i32 列表")
    etype, size = reader.read_list_header()
    return [_read_scalar(reader, etype) for _ in range(size)]


def _read_string_list(reader, ctype):
    _expect_ctype(reader, ctype, CT_LIST, "string 列表")
    etype, size = reader.read_list_header()
    return [reader.read_string() if etype == CT_BINARY
            else _read_scalar(reader, etype) for _ in range(size)]


# ---------------------------------------------------------------------
# 各结构解析
# ---------------------------------------------------------------------

def _parse_logical_type(reader):
    """LogicalType 是 union：每个 field 是一个空/带参 struct。"""
    name = None
    last = 0
    while True:
        fid, ctype = reader.read_field_header(last)
        if ctype == 0:  # STOP
            return name
        last = fid
        if fid == 1:  # StringType
            name = "STRING"
        elif fid == 3:  # ListType
            name = "LIST"
        # 其余 union 成员（DECIMAL/TIME/...）本扫描器不使用
        reader.skip_value(ctype)


def parse_schema_element(buf, base_offset):
    reader = CompactReader(buf, base_offset)
    se = _parse_inline_schema_element(reader)
    if reader.remaining() != 0:
        raise CorruptMetadataError(
            "SchemaElement 后存在多余字节", reader.abs_offset
        )
    return se


def _parse_inline_schema_element(reader):
    se = SchemaElement()
    last = 0
    while True:
        fid, ctype = reader.read_field_header(last)
        if ctype == 0:
            break
        last = fid
        if fid == 1:
            se.type = _read_scalar(reader, ctype)
        elif fid == 2:
            se.type_length = _read_scalar(reader, ctype)
        elif fid == 3:
            se.repetition_type = _read_scalar(reader, ctype)
        elif fid == 4:
            se.name = _read_scalar(reader, ctype)
        elif fid == 5:
            se.num_children = _read_scalar(reader, ctype)
        elif fid == 6:
            se.converted_type = _read_scalar(reader, ctype)
        elif fid in (7, 8, 9):
            # scale / precision / field_id
            setattr(se, ("scale", "precision", "field_id")[fid - 7],
                    _read_scalar(reader, ctype))
        elif fid == 10:
            _expect_ctype(reader, ctype, CT_STRUCT, "LogicalType")
            se.logical_type = _parse_logical_type(reader)
        else:
            reader.skip_value(ctype)
    if se.name is None:
        raise CorruptMetadataError("SchemaElement 缺少 name", reader.abs_offset)
    return se


def _parse_column_metadata(reader):
    md = ColumnMetaData()
    last = 0
    while True:
        fid, ctype = reader.read_field_header(last)
        if ctype == 0:
            break
        last = fid
        if fid == 1:
            md.type = _read_scalar(reader, ctype)
        elif fid == 2:
            md.encodings = _read_i32_list(reader, ctype)
        elif fid == 3:
            md.path_in_schema = _read_string_list(reader, ctype)
        elif fid == 4:
            md.codec = _read_scalar(reader, ctype)
        elif fid == 5:
            md.num_values = _read_scalar(reader, ctype)
        elif fid == 6:
            md.total_uncompressed_size = _read_scalar(reader, ctype)
        elif fid == 7:
            md.total_compressed_size = _read_scalar(reader, ctype)
        elif fid == 9:
            md.data_page_offset = _read_scalar(reader, ctype)
        elif fid == 10:
            md.index_page_offset = _read_scalar(reader, ctype)
        elif fid == 11:
            md.dictionary_page_offset = _read_scalar(reader, ctype)
        else:
            reader.skip_value(ctype)
    return md


def _parse_column_chunk(reader):
    cc = ColumnChunk()
    last = 0
    while True:
        fid, ctype = reader.read_field_header(last)
        if ctype == 0:
            break
        last = fid
        if fid == 1:
            cc.file_path = _read_scalar(reader, ctype)
        elif fid == 2:
            cc.file_offset = _read_scalar(reader, ctype)
        elif fid == 3:
            _expect_ctype(reader, ctype, CT_STRUCT, "ColumnMetaData")
            cc.meta_data = _parse_column_metadata(reader)
        else:
            reader.skip_value(ctype)
    return cc


def _parse_row_group(reader):
    rg = RowGroup()
    last = 0
    while True:
        fid, ctype = reader.read_field_header(last)
        if ctype == 0:
            break
        last = fid
        if fid == 1:
            _expect_ctype(reader, ctype, CT_LIST, "RowGroup.columns")
            etype, size = reader.read_list_header()
            for _ in range(size):
                _expect_ctype(reader, etype, CT_STRUCT, "ColumnChunk")
                rg.columns.append(_parse_column_chunk(reader))
        elif fid == 2:
            rg.total_byte_size = _read_scalar(reader, ctype)
        elif fid == 3:
            rg.num_rows = _read_scalar(reader, ctype)
        elif fid == 5:
            rg.file_offset = _read_scalar(reader, ctype)
        elif fid == 6:
            rg.total_compressed_size = _read_scalar(reader, ctype)
        elif fid == 7:
            rg.ordinal = _read_scalar(reader, ctype)
        else:
            reader.skip_value(ctype)
    return rg


def _parse_data_page_header(reader):
    h = DataPageHeader()
    last = 0
    while True:
        fid, ctype = reader.read_field_header(last)
        if ctype == 0:
            break
        last = fid
        if fid == 1:
            h.num_values = _read_scalar(reader, ctype)
        elif fid == 2:
            h.encoding = _read_scalar(reader, ctype)
        elif fid == 3:
            h.definition_level_encoding = _read_scalar(reader, ctype)
        elif fid == 4:
            h.repetition_level_encoding = _read_scalar(reader, ctype)
        else:
            reader.skip_value(ctype)
    return h


def _parse_data_page_header_v2(reader):
    h = DataPageHeaderV2()
    last = 0
    while True:
        fid, ctype = reader.read_field_header(last)
        if ctype == 0:
            break
        last = fid
        if fid == 1:
            h.num_values = _read_scalar(reader, ctype)
        elif fid == 2:
            h.num_nulls = _read_scalar(reader, ctype)
        elif fid == 3:
            h.num_rows = _read_scalar(reader, ctype)
        elif fid == 4:
            h.encoding = _read_scalar(reader, ctype)
        elif fid == 5:
            # V2 中为 definition_levels_byte_length（i32），不是编码
            h.definition_levels_byte_length = _read_scalar(reader, ctype)
        elif fid == 6:
            # V2 中为 repetition_levels_byte_length（i32）
            h.repetition_levels_byte_length = _read_scalar(reader, ctype)
        elif fid == 7:
            h.is_compressed = _read_scalar(reader, ctype)
        else:
            reader.skip_value(ctype)
    return h


def _parse_dictionary_page_header(reader):
    h = DictionaryPageHeader()
    last = 0
    while True:
        fid, ctype = reader.read_field_header(last)
        if ctype == 0:
            break
        last = fid
        if fid == 1:
            h.num_values = _read_scalar(reader, ctype)
        elif fid == 2:
            h.encoding = _read_scalar(reader, ctype)
        elif fid == 3:
            h.is_sorted = _read_scalar(reader, ctype)
        else:
            reader.skip_value(ctype)
    return h


def parse_page_header(buf, base_offset):
    """解析一个页头；返回 PageHeader（header_offset 记录其文件位置）。"""
    reader = CompactReader(buf, base_offset)
    ph = PageHeader()
    ph.header_offset = base_offset
    last = 0
    while True:
        fid, ctype = reader.read_field_header(last)
        if ctype == 0:
            break
        last = fid
        if fid == 1:
            ph.type = _read_scalar(reader, ctype)
        elif fid == 2:
            ph.uncompressed_page_size = _read_scalar(reader, ctype)
        elif fid == 3:
            ph.compressed_page_size = _read_scalar(reader, ctype)
        elif fid == 4:
            ph.crc = _read_scalar(reader, ctype)
        elif fid == 5:
            _expect_ctype(reader, ctype, CT_STRUCT, "DataPageHeader")
            ph.data_page_header = _parse_data_page_header(reader)
        elif fid == 7:
            _expect_ctype(reader, ctype, CT_STRUCT, "DictionaryPageHeader")
            ph.dictionary_page_header = _parse_dictionary_page_header(reader)
        elif fid == 8:
            _expect_ctype(reader, ctype, CT_STRUCT, "DataPageHeaderV2")
            ph.data_page_header_v2 = _parse_data_page_header_v2(reader)
        else:
            # IndexPageHeader 等本实现不使用
            reader.skip_value(ctype)
    header_size = reader.pos
    return ph, header_size


def parse_footer(buf, footer_offset):
    """解析 FileMetaData（buf 为页脚 thrift 字节）。"""
    reader = CompactReader(buf, footer_offset)
    md = FileMetaData()
    last = 0
    while True:
        fid, ctype = reader.read_field_header(last)
        if ctype == 0:
            break
        last = fid
        if fid == 1:
            md.version = _read_scalar(reader, ctype)
        elif fid == 2:
            _expect_ctype(reader, ctype, CT_LIST, "FileMetaData.schema")
            etype, size = reader.read_list_header()
            for _ in range(size):
                _expect_ctype(reader, etype, CT_STRUCT, "SchemaElement")
                md.schema.append(_parse_inline_schema_element(reader))
        elif fid == 3:
            md.num_rows = _read_scalar(reader, ctype)
        elif fid == 4:
            _expect_ctype(reader, ctype, CT_LIST, "FileMetaData.row_groups")
            etype, size = reader.read_list_header()
            for _ in range(size):
                _expect_ctype(reader, etype, CT_STRUCT, "RowGroup")
                md.row_groups.append(_parse_row_group(reader))
        elif fid == 6:
            md.created_by = _read_scalar(reader, ctype)
        elif fid == 5:
            # KeyValue 列表，本实现不使用
            reader.skip_value(ctype)
        else:
            reader.skip_value(ctype)
    return md


def _extract_struct(buf, pos):
    """从 buf[pos:] 取出一个完整 compact struct 的字节（含 STOP）。

    保留作为通用工具；当前页脚 schema 解析已改为内联读取。
    """
    reader = CompactReader(buf, 0)
    reader.pos = pos
    reader.skip_struct()
    return buf[pos:reader.pos]


# ---------------------------------------------------------------------
# 文件级入口
# ---------------------------------------------------------------------

def read_footer(view):
    """读取并校验文件尾部，返回 (FileMetaData, 页脚起始偏移)。"""
    if view.size < 8:
        raise TruncatedFileError("文件不足 8 字节，不可能是 Parquet", 0)
    head = view.read_at(0, 4)
    if head != PARQUET_MAGIC:
        raise CorruptMetadataError(
            f"文件头魔数非法：{head!r}，期望 PAR1", 0
        )
    tail = view.read_at(view.size - 8, 8)
    if tail[4:] != PARQUET_MAGIC:
        raise CorruptMetadataError(
            f"文件尾魔数非法：{tail[4:]!r}，期望 PAR1", view.size - 4
        )
    footer_len = struct.unpack("<I", tail[:4])[0]
    if footer_len <= 0 or footer_len > view.size - 8:
        raise CorruptMetadataError(
            f"页脚长度 {footer_len} 非法（文件大小 {view.size}）",
            view.size - 8,
        )
    footer_offset = view.size - 8 - footer_len
    footer_buf = view.read_at(footer_offset, footer_len)
    metadata = parse_footer(footer_buf, footer_offset)
    if metadata.version is None or not metadata.schema or not metadata.row_groups:
        raise CorruptMetadataError(
            "FileMetaData 缺少 version/schema/row_groups", footer_offset
        )
    if metadata.version not in (1, 2):
        raise UnsupportedFormatError(
            f"不支持的 Parquet 版本 {metadata.version}", footer_offset
        )
    return metadata, footer_offset
