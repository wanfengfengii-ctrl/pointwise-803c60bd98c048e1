"""列块逐页读取器。

每个被投影/谓词引用的物理叶列对应一个 :class:`LeafReader`，它在列块内
顺序消费字典页与数据页（V1/V2），对外只暴露“下一条记录”的级值与值。
页面边界对上层透明：一行或一个列表被切到不同页面时，读取器自动续页，
保证谓词始终在同一元素路径内连续求值。
"""

from .encoding import (
    ByteReader,
    decode_data_values,
    decode_dictionary_page,
    decode_levels_v1,
    decode_levels_v2,
)
from .errors import (
    CorruptMetadataError,
    TruncatedFileError,
    UnsupportedFormatError,
)
from .metadata import (
    CODEC_UNCOMPRESSED,
    ENC_RLE,
    PAGE_DATA,
    PAGE_DATA_V2,
    PAGE_DICTIONARY,
    PAGE_INDEX,
    parse_page_header,
)

# V1 级值编码允许值：RLE（hybrid 以 RLE 编码编号登记）
_ALLOWED_LEVEL_ENCODING = ENC_RLE


class LeafReader:
    def __init__(self, view, column_chunk, leaf_node):
        self.view = view
        self.chunk = column_chunk
        self.meta = column_chunk.meta_data
        self.leaf = leaf_node
        self.physical_type = leaf_node.physical_type
        self.is_utf8 = leaf_node.is_utf8
        self.max_rep = leaf_node.max_rep
        self.max_def = leaf_node.max_def

        if self.meta is None:
            raise CorruptMetadataError("ColumnChunk 缺少 ColumnMetaData")
        if self.meta.codec != CODEC_UNCOMPRESSED:
            raise UnsupportedFormatError(
                f"列 {self.meta.path_in_schema} 使用压缩编码 {self.meta.codec}，"
                "仅支持未压缩数据", self.meta.data_page_offset,
            )
        if self.meta.type != leaf_node.physical_type:
            raise CorruptMetadataError(
                f"列元数据类型 {self.meta.type} 与 schema 叶类型 "
                f"{leaf_node.physical_type} 不一致", self.meta.data_page_offset,
            )

        # 列块起点：有字典页时从字典页偏移开始，否则从数据页偏移开始
        if self.meta.dictionary_page_offset is not None:
            self.next_offset = self.meta.dictionary_page_offset
        else:
            if self.meta.data_page_offset is None:
                raise CorruptMetadataError(
                    "ColumnMetaData 缺少 data_page_offset", None
                )
            self.next_offset = self.meta.data_page_offset

        self.dict_values = None
        self.dictionary_seen = False
        self.data_seen = False

        # 当前数据页的队列
        self.rl_queue = []
        self.dl_queue = []
        self.value_queue = []
        self.records_consumed = 0
        self.expected_total = self.meta.num_values
        self.exhausted = False

    # -----------------------------------------------------------------
    def next_record(self):
        """返回下一条 (rep_level, def_level, value_or_None)。

        空列表/null 只消耗级值、不消耗值；非空叶子返回解出的 PLAIN/字典值。
        """
        if not self.rl_queue:
            if self.exhausted:
                raise TruncatedFileError(
                    "列块值数量少于元数据声明，行数据被截断",
                    self.next_offset,
                )
            self._load_next_data_page()
        rl = self.rl_queue.pop(0)
        dl = self.dl_queue.pop(0)
        value = None
        if dl == self.max_def:
            if not self.value_queue:
                raise TruncatedFileError(
                    "数据页非空值数量不足", self.next_offset
                )
            value = self.value_queue.pop(0)
        self.records_consumed += 1
        if self.records_consumed == self.expected_total:
            self.exhausted = True
        return rl, dl, value

    def ensure_finished(self):
        if self.records_consumed != self.expected_total:
            raise TruncatedFileError(
                f"列块实际记录 {self.records_consumed} 条，"
                f"元数据声明 {self.expected_total} 条",
                self.next_offset,
            )
        if not self.exhausted and (self.rl_queue or self.value_queue):
            raise CorruptMetadataError(
                "列块消费完毕后页内仍有残留数据", self.next_offset
            )

    # -----------------------------------------------------------------
    def _read_page(self):
        """读取下一个页头与完整页载荷，返回 (PageHeader, payload_bytes)。"""
        offset = self.next_offset
        # 页头 thrift 没有长度前缀：用递增缓冲试探解析，直到读出 STOP
        ph, header_size = self._parse_header_with_growth(offset)
        if ph.compressed_page_size is None or ph.uncompressed_page_size is None:
            raise CorruptMetadataError("页头缺少页大小字段", offset)
        payload_len = ph.compressed_page_size
        payload_start = offset + header_size
        self.view.require_inside(payload_start, payload_len)
        payload = self.view.read_at(payload_start, payload_len)
        # 下一页紧跟当前页
        self.next_offset = payload_start + payload_len
        if ph.uncompressed_page_size != payload_len:
            # 未压缩时两个大小必须相等
            raise CorruptMetadataError(
                f"未压缩页的 compressed={payload_len} 与 "
                f"uncompressed={ph.uncompressed_page_size} 不一致",
                offset,
            )
        return ph, payload, payload_start

    def _parse_header_with_growth(self, offset):
        size = 64
        while True:
            available = self.view.size - offset
            if size > available:
                size = available
            if size <= 0:
                raise TruncatedFileError("页头位置超出文件", offset)
            header_buf = self.view.read_at(offset, size)
            try:
                return parse_page_header(header_buf, offset)
            except TruncatedFileError:
                if size >= available:
                    # 已经读到文件尾仍不完整：真实截断
                    raise
                size = min(size * 4, available)

    def _load_next_data_page(self):
        """跳过可能的字典/索引页，载入并解码下一个数据页。"""
        while True:
            ph, payload, payload_start = self._read_page()
            if ph.type == PAGE_DICTIONARY:
                self._handle_dictionary_page(ph, payload, payload_start)
                continue
            if ph.type == PAGE_INDEX:
                # 索引页不参与扫描，载荷已按页大小跳过
                continue
            if ph.type == PAGE_DATA:
                self._decode_v1(ph, payload, payload_start)
                self.data_seen = True
                return
            if ph.type == PAGE_DATA_V2:
                self._decode_v2(ph, payload, payload_start)
                self.data_seen = True
                return
            raise UnsupportedFormatError(
                f"未知页类型 {ph.type}", ph.header_offset
            )

    def _handle_dictionary_page(self, ph, payload, payload_start):
        if self.data_seen:
            raise CorruptMetadataError(
                "字典页出现在数据页之后", ph.header_offset
            )
        if self.dictionary_seen:
            raise CorruptMetadataError(
                "一个列块出现多个字典页", ph.header_offset
            )
        self.dictionary_seen = True
        header = ph.dictionary_page_header
        if header is None:
            raise CorruptMetadataError(
                "字典页缺少 DictionaryPageHeader", ph.header_offset
            )
        self.dict_values = decode_dictionary_page(
            payload, payload_start, header,
            self.physical_type, self.is_utf8,
        )

    # -----------------------------------------------------------------
    def _decode_v1(self, ph, payload, payload_start):
        header = ph.data_page_header
        if header is None:
            raise CorruptMetadataError(
                "V1 数据页缺少 DataPageHeader", ph.header_offset
            )
        if header.definition_level_encoding != _ALLOWED_LEVEL_ENCODING \
                or header.repetition_level_encoding != _ALLOWED_LEVEL_ENCODING:
            raise UnsupportedFormatError(
                "V1 级值仅支持 RLE/bit-packing hybrid，不支持旧版 BIT_PACKED",
                ph.header_offset,
            )
        num_values = header.num_values
        if num_values < 0:
            raise CorruptMetadataError("数据页 num_values 为负", ph.header_offset)
        reader = ByteReader(payload, payload_start)

        rl_start = reader.abs_offset
        rls = decode_levels_v1(reader, self.max_rep, num_values, rl_start)
        dl_start = reader.abs_offset
        dls = decode_levels_v1(reader, self.max_def, num_values, dl_start)

        num_non_null = sum(1 for d in dls if d == self.max_def)
        values = decode_data_values(
            reader, header.encoding, self.physical_type, self.is_utf8,
            num_non_null, self.dict_values, payload_start,
        )
        if reader.remaining() != 0:
            raise TruncatedFileError(
                f"V1 数据页解码后残留 {reader.remaining()} 字节",
                reader.abs_offset,
            )
        self._stage_page(rls, dls, values, num_values)

    def _decode_v2(self, ph, payload, payload_start):
        header = ph.data_page_header_v2
        if header is None:
            raise CorruptMetadataError(
                "V2 数据页缺少 DataPageHeaderV2", ph.header_offset
            )
        if header.is_compressed:
            raise UnsupportedFormatError(
                "V2 数据页声明已压缩，仅支持未压缩数据", ph.header_offset
            )
        # V2 的级值编码固定为 RLE/bit-packing hybrid；页头不再登记编码
        num_values = header.num_values
        if num_values < 0:
            raise CorruptMetadataError("V2 数据页 num_values 为负", ph.header_offset)
        reader = ByteReader(payload, payload_start)

        rl_len = header.repetition_levels_byte_length
        dl_len = header.definition_levels_byte_length
        if rl_len is None or dl_len is None or rl_len < 0 or dl_len < 0:
            raise CorruptMetadataError(
                "V2 页头缺少合法的级值字节长度", ph.header_offset
            )
        rl_start = reader.abs_offset
        rls = decode_levels_v2(reader, self.max_rep, num_values, rl_len, rl_start)
        dl_start = reader.abs_offset
        dls = decode_levels_v2(reader, self.max_def, num_values, dl_len, dl_start)

        num_non_null = sum(1 for d in dls if d == self.max_def)
        if header.num_nulls is not None and header.num_nulls != num_values - num_non_null:
            raise CorruptMetadataError(
                f"V2 num_nulls={header.num_nulls} 与定义级推出的空值数 "
                f"{num_values - num_non_null} 不一致",
                ph.header_offset,
            )
        values = decode_data_values(
            reader, header.encoding, self.physical_type, self.is_utf8,
            num_non_null, self.dict_values, payload_start,
        )
        if reader.remaining() != 0:
            raise TruncatedFileError(
                f"V2 数据页解码后残留 {reader.remaining()} 字节",
                reader.abs_offset,
            )
        self._stage_page(rls, dls, values, num_values)

    def _stage_page(self, rls, dls, values, num_values):
        if len(rls) != num_values or len(dls) != num_values:
            raise CorruptMetadataError(
                "数据页级值长度与 num_values 不一致", None
            )
        if len(values) != sum(1 for d in dls if d == self.max_def):
            raise CorruptMetadataError(
                "数据页非空值数量与定义级不匹配", None
            )
        self.rl_queue = rls
        self.dl_queue = dls
        self.value_queue = values
