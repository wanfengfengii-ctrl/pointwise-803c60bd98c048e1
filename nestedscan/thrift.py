"""Thrift Compact Protocol 的最小解码/编码实现（仅标准库）。

Parquet 页脚与页头均以 Thrift Compact Protocol 序列化。本模块只实现
Parquet 元数据用到的特性：布尔紧凑形式、zigzag 变长整数、二进制、
list/set/map、struct 字段头（字段 id 增量 + 字段类型）。
"""

from .errors import CorruptMetadataError, TruncatedFileError

# Compact protocol 字段类型编号
CT_STOP = 0
CT_TRUE = 1
CT_FALSE = 2
CT_BYTE = 3
CT_I16 = 4
CT_I32 = 5
CT_I64 = 6
CT_DOUBLE = 7
CT_BINARY = 8
CT_LIST = 9
CT_SET = 10
CT_MAP = 11
CT_STRUCT = 12


def zigzag_decode(n):
    return (n >> 1) ^ -(n & 1)


def zigzag_encode(n):
    # 先按 64 位有符号回绕，再左移；负数的算术右移由掩码保证位模式正确
    n &= (1 << 64) - 1
    return ((n << 1) ^ (n >> 63)) & ((1 << 64) - 1)


class CompactReader:
    """从 bytes 读取 compact protocol；记录流内偏移以便报错。"""

    def __init__(self, buf, base_offset=0):
        self.buf = buf
        self.pos = 0
        # base_offset: 该字节流在整个 Parquet 文件中的起始绝对偏移
        self.base_offset = base_offset

    @property
    def abs_offset(self):
        return self.base_offset + self.pos

    def remaining(self):
        return len(self.buf) - self.pos

    def read_byte(self):
        if self.pos >= len(self.buf):
            raise TruncatedFileError("Thrift 结构被截断", self.abs_offset)
        b = self.buf[self.pos]
        self.pos += 1
        return b

    def read_varint(self):
        result = 0
        shift = 0
        start = self.abs_offset
        while True:
            if self.pos >= len(self.buf):
                raise TruncatedFileError("varint 被截断", start)
            b = self.buf[self.pos]
            self.pos += 1
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                return result
            shift += 7
            if shift > 70:
                raise CorruptMetadataError("varint 过长", start)

    def read_zigzag(self):
        return zigzag_decode(self.read_varint())

    def read_binary(self):
        length = self.read_varint()
        start = self.pos
        if length > len(self.buf) - self.pos:
            raise TruncatedFileError(
                "binary 字段声明长度超出文件", self.abs_offset
            )
        data = self.buf[start:start + length]
        self.pos += length
        return data

    def read_string(self):
        return self.read_binary().decode("utf-8")

    def read_double(self):
        import struct
        if self.pos + 8 > len(self.buf):
            raise TruncatedFileError("double 被截断", self.abs_offset)
        vals = struct.unpack_from("<d", self.buf, self.pos)
        self.pos += 8
        return vals[0]

    def read_list_header(self):
        """返回 (元素类型, 元素个数)。"""
        size_type = self.read_byte()
        size = size_type >> 4
        etype = size_type & 0x0F
        if size == 15:
            size = self.read_varint()
        return etype, size

    def read_map_header(self):
        """返回 (key 类型, value 类型, 元素个数)。"""
        size = self.read_varint()
        if size == 0:
            return 0, 0, 0
        kv = self.read_byte()
        return kv >> 4, kv & 0x0F, size

    def read_field_header(self, last_field_id):
        """返回 (字段 id, compact 类型)；STOP 时返回 (0, 0)。"""
        b = self.read_byte()
        if b == CT_STOP:
            return 0, CT_STOP
        delta = b >> 4
        ctype = b & 0x0F
        if delta == 0:
            # 非紧凑形式：字段 id 是绝对 id（zigzag）
            fid = self.read_zigzag()
        else:
            fid = last_field_id + delta
        return fid, ctype

    def skip_value(self, ctype):
        """跳过一个未知字段，保证前向兼容。"""
        if ctype in (CT_TRUE, CT_FALSE):
            return
        if ctype == CT_BYTE:
            self.read_byte()
        elif ctype in (CT_I16, CT_I32, CT_I64):
            self.read_varint()
        elif ctype == CT_DOUBLE:
            self.read_double()
        elif ctype == CT_BINARY:
            self.read_binary()
        elif ctype in (CT_LIST, CT_SET):
            etype, size = self.read_list_header()
            for _ in range(size):
                self.skip_value(etype)
        elif ctype == CT_MAP:
            kt, vt, size = self.read_map_header()
            for _ in range(size):
                self.skip_value(kt)
                self.skip_value(vt)
        elif ctype == CT_STRUCT:
            self.skip_struct()
        else:
            raise CorruptMetadataError(
                f"未知 Thrift compact 类型 {ctype}", self.abs_offset
            )

    def skip_struct(self):
        last = 0
        while True:
            fid, ctype = self.read_field_header(last)
            if ctype == CT_STOP:
                return
            last = fid
            self.skip_value(ctype)


class CompactWriter:
    """Thrift Compact Protocol 编码器（供测试夹具生成 Parquet 文件）。"""

    def __init__(self):
        self.parts = []

    def getvalue(self):
        return b"".join(self.parts)

    def write_varint(self, n):
        out = bytearray()
        while True:
            b = n & 0x7F
            n >>= 7
            if n:
                out.append(b | 0x80)
            else:
                out.append(b)
                self.parts.append(bytes(out))
                return
    def write_zigzag(self, n):
        self.write_varint(zigzag_encode(n))

    def write_binary(self, data):
        self.write_varint(len(data))
        self.parts.append(bytes(data))

    def write_string(self, s):
        self.write_binary(s.encode("utf-8"))

    def write_double(self, value):
        import struct
        self.parts.append(struct.pack("<d", value))

    def write_field_header(self, fid, ctype, last_fid):
        delta = fid - last_fid
        if 1 <= delta <= 15:
            self.parts.append(bytes([(delta << 4) | ctype]))
        else:
            self.parts.append(bytes([ctype]))
            self.write_zigzag(fid)

    def write_list_header(self, etype, size):
        if size < 15:
            self.parts.append(bytes([(size << 4) | etype]))
        else:
            self.parts.append(bytes([0xF0 | etype]))
            self.write_varint(size)

    def write_map_header(self, ktype, vtype, size):
        self.write_varint(size)
        if size:
            self.parts.append(bytes([(ktype << 4) | vtype]))

    def write_struct_stop(self):
        self.parts.append(b"\x00")
