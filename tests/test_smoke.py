"""端到端冒烟测试：生成 Parquet 并用 nestedscan 扫描。"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nestedscan import scan
from nestedscan.metadata import REP_OPTIONAL, REP_REQUIRED, TYPE_INT64
from parquet_writer import leaf, struct_node, list3, write_parquet


class SmokeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nestedscan_smoke_")

    def _path(self, name):
        return os.path.join(self.tmp, name)

    def test_flat_rows_v1_plain(self):
        schema = [
            leaf("x", TYPE_INT64),
            leaf("y", TYPE_INT64),
        ]
        rows = [{"x": 1, "y": 0}, {"x": 0, "y": 1}, {"x": None, "y": 2}]
        f = self._path("flat_v1.parquet")
        write_parquet(f, "root", schema, rows, page_version="v1")
        out = list(scan(f, [["x"], ["y"]]))
        self.assertEqual(out, rows)

    def test_list_basic(self):
        schema = [
            list3("items", leaf("qty", TYPE_INT64)),
        ]
        rows = [
            {"items": [{"qty": 1}, {"qty": 2}]},
            {"items": []},
            {"items": None},
            {"items": [{"qty": None}]},
        ]
        f = self._path("list.parquet")
        write_parquet(f, "root", schema, rows)
        out = list(scan(f, [["items", []]]))
        self.assertEqual(out, rows)


if __name__ == "__main__":
    unittest.main()
