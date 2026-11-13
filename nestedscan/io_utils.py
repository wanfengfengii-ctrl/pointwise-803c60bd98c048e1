"""带绝对文件偏移感知的只读文件封装。

所有越界读取都会抛出 :class:`TruncatedFileError` 并携带文件偏移，
供上层在截断/损坏场景下给出带偏移的错误信息。
"""

from .errors import TruncatedFileError


class FileView:
    def __init__(self, path):
        self.path = path
        # 以二进制方式打开；调用方负责关闭
        self.f = open(path, "rb")
        self.f.seek(0, 2)
        self.size = self.f.tell()
        self.f.seek(0)

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def seek(self, offset):
        self.f.seek(offset)

    def tell(self):
        return self.f.tell()

    def read(self, count):
        """精确读取 count 字节，不足则报截断错误（带当前偏移）。"""
        offset = self.f.tell()
        if count < 0:
            raise TruncatedFileError("读取长度为负", offset)
        data = self.f.read(count)
        if len(data) != count:
            raise TruncatedFileError(
                f"需要 {count} 字节，实际只剩 {len(data)} 字节", offset
            )
        return data

    def read_at(self, offset, count):
        self.f.seek(offset)
        return self.read(count)

    def require_inside(self, offset, length=0):
        """校验 [offset, offset+length) 落在文件范围内。"""
        if offset < 0 or length < 0 or offset + length > self.size:
            raise TruncatedFileError(
                f"文件范围 [0x{offset:x}, 0x{offset + length:x}) 超出"
                f"文件大小 0x{self.size:x}",
                offset,
            )
