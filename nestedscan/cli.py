"""nestedscan 命令行入口。

用法::

    python -m nestedscan.cli FILE --column a.b --column 'c.[].d' \
        [--where '["eq", ["x"], 1]']

列路径与谓词均为 JSON：字段名字符串数组，列表元素位置用 ``[]`` 表示。
命中的每行以 JSON Lines 形式输出到 stdout。
"""

import argparse
import json
import sys

from .errors import NestedScanError
from . import scan


def _parse_column(text):
    try:
        path = json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"列路径不是合法 JSON: {exc}")
    # JSON 中用空数组 [] 表示列表元素位置
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="nestedscan",
        description="纯标准库嵌套 Parquet 扫描器",
    )
    parser.add_argument("file", help="Parquet 文件路径")
    parser.add_argument(
        "--column", action="append", required=True,
        help="逻辑叶路径（JSON 数组，可重复指定），[] 表示列表元素位置",
    )
    parser.add_argument(
        "--where", default=None,
        help='谓词 JSON，例如 \'["any", ["items"], ["eq", ["qty"], 3]]\'',
    )
    args = parser.parse_args(argv)

    columns = [_parse_column(c) for c in args.column]
    where = json.loads(args.where) if args.where else None

    try:
        for row in scan(args.file, columns, where):
            print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    except NestedScanError as exc:
        print(f"nestedscan 错误: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
