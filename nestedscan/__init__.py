"""nestedscan: 纯 Python 标准库实现的嵌套 Parquet 扫描器。

对外入口: :func:`nestedscan.scan`。
"""

from .errors import NestedScanError, UnsupportedFormatError

__all__ = ["scan", "NestedScanError", "UnsupportedFormatError"]

__version__ = "1.0.0"


def scan(path, columns, where=None):
    """扫描嵌套 Parquet 文件，按原行序迭代投影字典。

    参数:
        path: Parquet 文件路径。
        columns: 非空逻辑叶路径列表；每个路径是字段名数组，
                 列表位置用空数组 ``[]`` 表示（表示“当前值/元素”）。
        where: 谓词，取值为 None（不筛选）或以下形式之一::

            ["eq", 路径, 整数或字符串]
            ["and", p, q]
            ["not", p]
            ["any", 列表路径, p]   # p 内的路径相对于列表元素

    生成:
        每行一个 dict，仅含所选叶及其祖先，保留列表位置、空列表与 null。
    """
    from .engine import scan_file

    return scan_file(path, columns, where)
