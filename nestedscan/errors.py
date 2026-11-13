"""nestedscan 的异常类型。

所有针对损坏/截断文件的异常都携带触发问题时的文件绝对偏移 ``offset``
（无法确定时为 None），满足“截断、非法级值或字典越界须带文件偏移报错”。
"""


class NestedScanError(Exception):
    """nestedscan 基础异常。"""

    def __init__(self, message, offset=None):
        self.offset = offset
        if offset is not None:
            message = f"{message} (文件偏移 0x{offset:x})"
        super().__init__(message)


class TruncatedFileError(NestedScanError):
    """文件在读取过程中提前结束（截断）。"""


class IllegalLevelError(NestedScanError):
    """重复级/定义级超出 schema 允许的最大值（非法级值）。"""


class DictionaryOutOfBoundsError(NestedScanError):
    """字典索引超出字典页承载的条目数（字典越界）。"""


class UnsupportedFormatError(NestedScanError):
    """使用了本实现不支持的 Parquet 特性（压缩、编码、类型等）。"""


class InvalidPathError(NestedScanError):
    """请求的逻辑路径在 schema 中不存在或不是叶路径。"""


class InvalidPredicateError(NestedScanError):
    """谓词语法非法或与列类型不匹配。"""


class CorruptMetadataError(NestedScanError):
    """Parquet 页脚/元数据结构非法或自相矛盾。"""
