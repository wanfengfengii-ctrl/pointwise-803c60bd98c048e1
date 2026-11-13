"""Parquet 编码解码：RLE/bit-packing 混合（级值与字典索引）与 PLAIN。

只实现题目要求的未压缩 PLAIN 与 RLE_DICTIONARY（含旧称 PLAIN_DICTIONARY）。
所有读取都带文件绝对偏移，截断即抛 :class:`TruncatedFileError`。
"""

import struct

from .errors import (
    DictionaryOutOfBoundsError,
    IllegalLevelError,
    TruncatedFileError,
    UnsupportedFormatError,
)
from .metadata import (
    ENC_BIT_PACKED,
    ENC_PLAIN,
    ENC_PLAIN_DICTIONARY,
    ENC_RLE,
    ENC_RLE_DICTIONARY,
    TYPE_BYTE_ARRAY,
    TYPE_INT64,
)


class ByteReader:
    """带绝对偏移的字节游标（用于页内解码）。"""

    __slots__ = ("buf", "base", "pos")

    def __init__(self, buf, base_offset=0):
        self.buf = buf
        self.base = base_offset
        self.pos = 0

    @property
    def abs_offset(self):
        return self.base + self.pos

    def remaining(self):
        return len(self.buf) - self.pos

    def read(self, n):
        if n > len(self.buf) - self.pos:
            raise TruncatedFileError(
                f"页内需要 {n} 字节，剩余 {len(self.buf) - self.pos} 字节",
                self.abs_offset,
            )
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def read_byte(self):
        if self.pos >= len(self.buf):
            raise TruncatedFileError("页内字节被截断", self.abs_offset)
        b = self.buf[self.pos]
        self.pos += 1
        return b

    def read_varint(self):
        result = 0
        shift = 0
        start = self.abs_offset
        while True:
            if self.pos >= len(self.buf):
                raise TruncatedFileError("RLE 头 varint 被截断", start)
            b = self.buf[self.pos]
            self.pos += 1
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                return result
            shift += 7
            if shift > 35:
                raise TruncatedFileError("RLE 头 varint 过长", start)

    def read_int32(self):
        return struct.unpack("<i", self.read(4))[0]


# ---------------------------------------------------------------------
# RLE / bit-packing hybrid
# ---------------------------------------------------------------------

def decode_rle_hybrid(buf, base_offset, bit_width, num_values,
                      validate_max=None):
    """解码 RLE/bit-packing hybrid 编码的整数序列。

    参数:
        bit_width: 每个值的比特宽度（0..32）。
        num_values: 期望的值总数（级值长度或页 num_values）。
        validate_max: 若给出，每个解码值必须 <= 该值，否则按“非法级值”报错。

    编码格式: 前置 4 字节小端长度（数据页级值头）后接 hybrid 体；
    本函数直接接收 hybrid 体（不含长度前缀）。
    """
    if bit_width < 0 or bit_width > 32:
        raise UnsupportedFormatError(
            f"RLE hybrid 比特宽度 {bit_width} 非法", base_offset
        )
    reader = ByteReader(buf, base_offset)
    values = []
    if bit_width == 0:
        # 宽度 0：所有值隐含为 0。流要么为空，要么由若干 bit-pack 组头
        # （每组 0 字节）或零值 RLE run 组成。
        real_count = 0
        if reader.remaining() == 0:
            values = [0] * num_values
        else:
            while reader.remaining() > 0 and real_count < num_values:
                header = reader.read_varint()
                if header & 1:
                    # RLE run
                    count = header >> 1
                    zero = reader.read_byte()
                    if zero != 0:
                        raise IllegalLevelError(
                            f"比特宽度 0 的流出现非零值 {zero}",
                            reader.abs_offset,
                        )
                    values.extend([0] * count)
                    real_count += count
                else:
                    # bit-pack 组：宽度 0，组体为空
                    groups = header >> 1
                    if groups <= 0:
                        raise TruncatedFileError(
                            "宽度 0 的 bit-pack 组数为 0", reader.abs_offset
                        )
                    take = min(groups * 8, num_values - real_count)
                    values.extend([0] * take)
                    real_count += take
            if real_count != num_values or reader.remaining() != 0:
                raise TruncatedFileError(
                    f"宽度 0 hybrid 流计数 {real_count}/{num_values}，"
                    f"剩余 {reader.remaining()} 字节", reader.abs_offset,
                )
        _validate_levels(values, validate_max, base_offset)
        return values

    # 两遍解码：先把所有 run 的原始槽位读出，再按 num_values 收口。
    # RLE run 的每个槽都是真实值；bit-pack 组物理含 8*groups 槽，
    # 只有流的最后一个 bit-pack 组允许尾部填充，其余必须是满的真实值。
    raw_runs = []  # ("rle", [values...]) / ("bitpack", [values...])
    while reader.remaining() > 0:
        header = reader.read_varint()
        if header & 1:
            count = header >> 1
            if count <= 0:
                raise TruncatedFileError(
                    "RLE run 重复次数为 0", reader.abs_offset
                )
            packed = _read_packed_value(reader, bit_width)
            raw_runs.append(("rle", [packed] * count))
        else:
            groups = header >> 1
            if groups <= 0:
                raise TruncatedFileError(
                    "bit-packing run 组数为 0", reader.abs_offset
                )
            slots = groups * 8
            chunk = reader.read(groups * bit_width)
            raw_runs.append(("bitpack", _unpack_lsb(chunk, bit_width, slots)))

    values = []
    emitted = 0
    last_idx = len(raw_runs) - 1
    for idx, (kind, vals) in enumerate(raw_runs):
        if kind == "rle":
            if emitted + len(vals) > num_values:
                raise TruncatedFileError(
                    f"RLE run 使值数超过声明 {num_values}", reader.abs_offset
                )
            values.extend(vals)
            emitted += len(vals)
        else:
            if idx == last_idx:
                # 最后一个 run：只取还缺的值，其余视为填充
                take = max(0, num_values - emitted)
                if take > len(vals):
                    raise TruncatedFileError(
                        "末尾 bit-pack 组真实值不足", reader.abs_offset
                    )
                values.extend(vals[:take])
                emitted += take
            else:
                # 中间 bit-pack 组：必须全部是真实值
                if emitted + len(vals) > num_values:
                    raise TruncatedFileError(
                        "流中间的 bit-pack 组出现填充位，值数超限",
                        reader.abs_offset,
                    )
                values.extend(vals)
                emitted += len(vals)

    if emitted != num_values:
        raise TruncatedFileError(
            f"hybrid 解码得到 {emitted} 个值，期望 {num_values}",
            reader.abs_offset,
        )
    _validate_levels(values, validate_max, base_offset)
    return values


def _validate_levels(values, validate_max, base_offset):
    if validate_max is None:
        return
    for v in values:
        if v < 0 or v > validate_max:
            raise IllegalLevelError(
                f"级值 {v} 超出允许范围 [0, {validate_max}]", base_offset
            )


def _read_packed_value(reader, bit_width):
    """读取一个按 bit_width 位 LSB 打包的单值。"""
    nbytes = (bit_width + 7) // 8
    chunk = reader.read(nbytes)
    val = 0
    for i, b in enumerate(chunk):
        val |= b << (8 * i)
    val &= (1 << bit_width) - 1
    return val


def _unpack_lsb(chunk, bit_width, count):
    """LSB-first bit unpacking：chunk 为 count 个 bit_width 位值的紧凑打包。

    Parquet bit-packing 把所有值按低位优先连续铺放：第 i 个值占据
    比特区间 [i*width, (i+1)*width)，跨字节时低位在低字节。
    """
    if bit_width == 0:
        return [0] * count
    total_bits = bit_width * count
    if len(chunk) * 8 < total_bits:
        raise TruncatedFileError(
            f"bit-packing 需要 {total_bits} 位，实际 {len(chunk) * 8} 位",
            None,
        )
    # 拼成一个小端整数，低位在前；再按 width 逐一切出
    acc = 0
    for idx, byte in enumerate(chunk):
        acc |= byte << (8 * idx)
    mask = (1 << bit_width) - 1
    out = [0] * count
    for i in range(count):
        out[i] = (acc >> (i * bit_width)) & mask
    return out


def decode_levels_v1(reader, max_level, num_values, level_offset):
    """解码 V1 数据页的一个级值列（RL 或 DL）。

    V1 格式: 4 字节小端长度 + hybrid 体；max_level==0 时长度必为 0、
    没有 hybrid 体，但 4 字节长度前缀仍然存在，必须消费以推进游标。
    """
    length = reader.read_int32()
    if length < 0:
        raise TruncatedFileError("级值长度为负", reader.abs_offset)
    if max_level == 0:
        if length != 0:
            raise TruncatedFileError(
                f"max_level=0 时级值字节长度应为 0，实际 {length}",
                reader.abs_offset,
            )
        return [0] * num_values
    body = reader.read(length)
    return decode_rle_hybrid(
        body, level_offset, _bit_width(max_level), num_values,
        validate_max=max_level,
    )


def decode_levels_v2(reader, max_level, num_values, level_byte_length,
                     level_start_offset):
    """解码 V2 数据页的一个级值列（长度在页头给出，无内嵌长度前缀）。"""
    if max_level == 0:
        if level_byte_length != 0:
            raise TruncatedFileError(
                "max_level=0 时 V2 级值字节长度应为 0", level_start_offset
            )
        return [0] * num_values
    body = reader.read(level_byte_length)
    return decode_rle_hybrid(
        body, level_start_offset, _bit_width(max_level), num_values,
        validate_max=max_level,
    )


def _bit_width(max_value):
    if max_value <= 0:
        return 0
    width = 0
    v = max_value
    while v:
        width += 1
        v >>= 1
    return width


# ---------------------------------------------------------------------
# PLAIN 值与字典
# ---------------------------------------------------------------------

def decode_plain(buf, base_offset, physical_type, num_values, is_utf8):
    """解码 PLAIN 编码的非空值序列。"""
    reader = ByteReader(buf, base_offset)
    values = []
    if physical_type == TYPE_INT64:
        raw = reader.read(8 * num_values)
        values = list(struct.unpack("<%dq" % num_values, raw))
    elif physical_type == TYPE_BYTE_ARRAY:
        for _ in range(num_values):
            length = reader.read_int32()
            if length < 0:
                raise TruncatedFileError(
                    f"BYTE_ARRAY 长度为负 {length}", reader.abs_offset
                )
            data = reader.read(length)
            if is_utf8:
                try:
                    values.append(data.decode("utf-8"))
                except UnicodeDecodeError:
                    raise UnsupportedFormatError(
                        "UTF8 列包含非法 UTF-8 字节", reader.abs_offset
                    )
            else:
                values.append(data)
    else:
        raise UnsupportedFormatError(
            f"PLAIN 不支持的物理类型 {physical_type}", base_offset
        )
    return values, reader.pos


def decode_dictionary_page(buf, base_offset, header, physical_type, is_utf8):
    """解码字典页，返回字典值列表。"""
    if header.encoding not in (ENC_PLAIN, ENC_PLAIN_DICTIONARY):
        raise UnsupportedFormatError(
            f"字典页编码 {header.encoding} 不受支持（仅 PLAIN）", base_offset
        )
    values, consumed = decode_plain(
        buf, base_offset, physical_type, header.num_values, is_utf8
    )
    if consumed != len(buf):
        raise TruncatedFileError(
            f"字典页解码后残留 {len(buf) - consumed} 字节",
            base_offset + consumed,
        )
    if len(values) != header.num_values:
        raise TruncatedFileError(
            f"字典页声明 {header.num_values} 项，实际解码 {len(values)} 项",
            base_offset,
        )
    return values


def decode_data_values(reader, encoding, physical_type, is_utf8,
                       num_non_null, dict_values, payload_start):
    """按数据页编码解码非空值序列。

    返回长度为 num_non_null 的列表。
    """
    if encoding == ENC_PLAIN:
        remaining = reader.buf[reader.pos:]
        values, consumed = decode_plain(
            remaining, reader.abs_offset, physical_type,
            num_non_null, is_utf8,
        )
        reader.pos += consumed
        return values

    if encoding in (ENC_RLE_DICTIONARY, ENC_PLAIN_DICTIONARY):
        if dict_values is None:
            raise UnsupportedFormatError(
                "字典编码数据页之前缺少字典页", payload_start
            )
        bit_width = reader.read_byte()
        if bit_width > 32:
            raise UnsupportedFormatError(
                f"字典索引比特宽度 {bit_width} 非法", reader.abs_offset
            )
        width_start = reader.abs_offset
        # 索引流一直延伸到页尾（V1）或 is_compressed 数据尾（V2，调用方已切好）
        body = reader.read(reader.remaining())
        indices = decode_rle_hybrid(
            body, width_start, bit_width, num_non_null,
        )
        dict_len = len(dict_values)
        for idx in indices:
            if idx < 0 or idx >= dict_len:
                raise DictionaryOutOfBoundsError(
                    f"字典索引 {idx} 越界（字典共 {dict_len} 项）",
                    width_start,
                )
        return [dict_values[i] for i in indices]

    raise UnsupportedFormatError(
        f"数据页编码 {encoding} 不受支持（仅 PLAIN / RLE_DICTIONARY）",
        payload_start,
    )
