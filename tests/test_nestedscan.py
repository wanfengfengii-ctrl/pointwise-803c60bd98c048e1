"""nestedscan 完整功能测试。

覆盖：
* 扁平/嵌套 struct / 标准三层 LIST（含多层嵌套、null、空列表、null 元素）
* INT64 / UTF8、PLAIN / RLE_DICTIONARY、DataPage V1 / V2
* 小数据页切断一行或列表时的跨页续读
* 谓词 eq/and/not/any 的 SQL 三值逻辑（空列表=假、null=未知）
* 投影裁剪（只含所选叶及祖先，谓词叶不外泄）
* 错误处理：截断、非法级值、字典越界、压缩拒绝、非法路径
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nestedscan import scan
from nestedscan.errors import (
    DictionaryOutOfBoundsError,
    IllegalLevelError,
    InvalidPathError,
    InvalidPredicateError,
    NestedScanError,
    TruncatedFileError,
    UnsupportedFormatError,
)
from nestedscan.metadata import (
    REP_OPTIONAL,
    REP_REQUIRED,
    TYPE_BYTE_ARRAY,
    TYPE_INT64,
)
from parquet_writer import leaf, list3, struct_node, write_parquet


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nestedscan_")

    def path(self, name="t.parquet"):
        return os.path.join(self.tmp, name)

    def write(self, schema, rows, **kw):
        p = self.path()
        write_parquet(p, "root", schema, rows, **kw)
        return p


class FlatAndStructTest(Base):
    def test_flat_int64_nulls(self):
        schema = [leaf("x", TYPE_INT64), leaf("y", TYPE_INT64)]
        rows = [{"x": 1, "y": 0}, {"x": 0, "y": 1}, {"x": None, "y": 2}]
        for ver in ("v1", "v2"):
            p = self.write(schema, rows, page_version=ver)
            self.assertEqual(list(scan(p, [["x"], ["y"]])), rows)

    def test_utf8_strings(self):
        schema = [leaf("s", TYPE_BYTE_ARRAY, utf8=True)]
        rows = [{"s": "hello"}, {"s": "世界"}, {"s": None}, {"s": ""}]
        p = self.write(schema, rows)
        self.assertEqual([r["s"] for r in scan(p, [["s"]])],
                         ["hello", "世界", None, ""])

    def test_nested_struct_null(self):
        schema = [struct_node("addr", [
            leaf("city", TYPE_BYTE_ARRAY, utf8=True),
            leaf("zip", TYPE_INT64),
        ]), leaf("name", TYPE_BYTE_ARRAY, utf8=True)]
        rows = [
            {"addr": {"city": "NYC", "zip": 1}, "name": "a"},
            {"addr": None, "name": "b"},
        ]
        p = self.write(schema, rows)
        out = list(scan(p, [["addr", "city"], ["addr", "zip"], ["name"]]))
        self.assertEqual(out, rows)

    def test_projection_prunes_unselected(self):
        schema = [struct_node("a", [leaf("x", TYPE_INT64),
                                    leaf("y", TYPE_INT64)]),
                  leaf("z", TYPE_INT64)]
        rows = [{"a": {"x": 1, "y": 2}, "z": 3}]
        p = self.write(schema, rows)
        out = list(scan(p, [["a", "x"]]))
        self.assertEqual(out, [{"a": {"x": 1}}])


class ListTest(Base):
    def setUp(self):
        super().setUp()
        self.schema = [list3("items", leaf("qty", TYPE_INT64))]
        self.rows = [
            {"items": [{"qty": 1}, {"qty": 2}]},
            {"items": []},
            {"items": None},
            {"items": [{"qty": None}]},
        ]

    def test_list_all_cases(self):
        for ver in ("v1", "v2"):
            for use_dict in (False, True):
                p = self.write(self.schema, self.rows,
                               page_version=ver, use_dict=use_dict)
                out = list(scan(p, [["items", []]]))
                self.assertEqual(out, self.rows, f"{ver} dict={use_dict}")

    def test_list_of_structs(self):
        schema = [list3("orders", struct_node("o", [
            leaf("id", TYPE_INT64),
            leaf("code", TYPE_BYTE_ARRAY, utf8=True),
        ]))]
        rows = [
            {"orders": [{"o": {"id": 1, "code": "a"}},
                        {"o": {"id": None, "code": "b"}}]},
            {"orders": []},
            {"orders": None},
        ]
        expected = [
            {"orders": [{"id": 1, "code": "a"},
                        {"id": None, "code": "b"}]},
            {"orders": []},
            {"orders": None},
        ]
        p = self.write(schema, rows)
        out = list(scan(p, [["orders", [], "id"],
                            ["orders", [], "code"]]))
        self.assertEqual(out, expected)

    def test_nested_lists(self):
        schema = [list3("groups", list3("vals", leaf("n", TYPE_INT64)))]
        rows = [
            {"groups": [{"vals": [{"n": 1}, {"n": 2}]}, {"vals": []}]},
            {"groups": []},
            {"groups": None},
            {"groups": [{"vals": None}]},
        ]
        p = self.write(schema, rows)
        out = list(scan(p, [["groups", [], "vals", []]]))
        self.assertEqual(out, rows)
    def test_required_list_wrapper(self):
        schema = [list3("tags", leaf("t", TYPE_BYTE_ARRAY, utf8=True),
                        repetition=REP_REQUIRED)]
        rows = [{"tags": [{"t": "x"}]}, {"tags": []}]
        p = self.write(schema, rows)
        self.assertEqual(list(scan(p, [["tags", []]])), rows)


class CrossPageTest(Base):
    def test_page_splits_row_and_list(self):
        # 每页只放 2 条物理记录，强制列表与行被切断到不同页面
        schema = [list3("items", leaf("qty", TYPE_INT64))]
        rows = [{"items": [{"qty": k} for k in range(5)]},
                {"items": [{"qty": 9}, {"qty": 8}]},
                {"items": []}, {"items": None}]
        for ver in ("v1", "v2"):
            p = self.write(schema, rows, page_value_cap=2, page_version=ver)
            out = list(scan(p, [["items", []]]))
            self.assertEqual(out, rows, ver)

    def test_many_pages_flat(self):
        schema = [leaf("v", TYPE_INT64)]
        rows = [{"v": i} for i in range(50)]
        p = self.write(schema, rows, page_value_cap=3)
        self.assertEqual([r["v"] for r in scan(p, [["v"]])],
                         list(range(50)))

    def test_multi_page_shared_dictionary(self):
        # 字典页被多个小数据页共享：验证字典在跨页时持续有效
        schema = [leaf("cat", TYPE_INT64)]
        rows = [{"cat": i % 4} for i in range(50)]
        for ver in ("v1", "v2"):
            p = self.write(schema, rows, use_dict=True,
                           page_value_cap=5, page_version=ver)
            self.assertEqual([r["cat"] for r in scan(p, [["cat"]])],
                             [i % 4 for i in range(50)])

    def test_multiple_row_groups(self):
        schema = [leaf("v", TYPE_INT64),
                  leaf("s", TYPE_BYTE_ARRAY, utf8=True)]
        rows = [{"v": i, "s": f"t{i}"} for i in range(12)]
        p = self.write(schema, rows, row_group_rows=4)
        out = list(scan(p, [["v"], ["s"]]))
        self.assertEqual(out, rows)
        # 谓词跨全部行组生效
        hit = list(scan(p, [["v"]], ["eq", ["v"], 9]))
        self.assertEqual(hit, [{"v": 9}])

    def test_cross_page_any_predicate_same_element(self):
        # 每页仅 2 条记录，列表被切断到多个页面；any 的 AND 必须仍在
        # 同一元素路径内求值，跨元素的 (x=1,y=1) 不命中。
        schema = [list3("pts", struct_node("p", [
            leaf("x", TYPE_INT64), leaf("y", TYPE_INT64)]))]
        rows = [
            {"pts": [{"p": {"x": 1, "y": 0}}, {"p": {"x": 0, "y": 1}},
                     {"p": {"x": 1, "y": 1}}, {"p": {"x": 1, "y": 1}}]},
            {"pts": []},
        ]
        where = ["any", ["pts"],
                 ["and", ["eq", ["x"], 1], ["eq", ["y"], 1]]]
        for ver in ("v1", "v2"):
            p = self.write(schema, rows, page_value_cap=2,
                           page_version=ver)
            out = list(scan(p, [["pts", [], "x"]], where))
            self.assertEqual(len(out), 1, ver)


class PredicateTest(Base):
    def setUp(self):
        super().setUp()
        # 题目给定的反例数据
        self.schema = [leaf("x", TYPE_INT64), leaf("y", TYPE_INT64)]
        self.rows = [{"x": 1, "y": 0}, {"x": 0, "y": 1}]

    def test_eq_filter(self):
        p = self.write(self.schema, self.rows)
        out = list(scan(p, [["x"]], ["eq", ["x"], 1]))
        self.assertEqual(out, [{"x": 1}])

    def test_and_not(self):
        p = self.write(self.schema, self.rows)
        where = ["and", ["eq", ["x"], 0], ["not", ["eq", ["y"], 0]]]
        out = list(scan(p, [["x"], ["y"]], where))
        self.assertEqual(out, [{"x": 0, "y": 1}])

    def test_any_counterexample_no_match(self):
        # [{x:1,y:0},{x:0,y:1}] 不命中 any(x=1 且 y=1)
        schema = [list3("pts", struct_node("p", [
            leaf("x", TYPE_INT64), leaf("y", TYPE_INT64)]))]
        rows = [{"pts": [{"p": {"x": 1, "y": 0}},
                         {"p": {"x": 0, "y": 1}}]}]
        p = self.write(schema, rows)
        where = ["any", ["pts"],
                 ["and", ["eq", ["x"], 1],
                  ["eq", ["y"], 1]]]
        out = list(scan(p, [["pts", [], "x"]], where))
        self.assertEqual(out, [])

    def test_any_matches_same_element_only(self):
        # 谓词必须在同一元素内成立：跨元素的 x=1、y=1 不应命中
        schema = [list3("pts", struct_node("p", [
            leaf("x", TYPE_INT64), leaf("y", TYPE_INT64)]))]
        rows = [
            {"pts": [{"p": {"x": 1, "y": 0}},
                     {"p": {"x": 0, "y": 1}}]},   # 跨元素 -> 不命中
            {"pts": [{"p": {"x": 1, "y": 1}}]},    # 同元素 -> 命中
        ]
        p = self.write(schema, rows)
        where = ["any", ["pts"],
                 ["and", ["eq", ["x"], 1],
                  ["eq", ["y"], 1]]]
        out = list(scan(p, [["pts", [], "x"]], where))
        self.assertEqual(len(out), 1)

    def test_any_empty_list_is_false(self):
        schema = [list3("items", leaf("qty", TYPE_INT64))]
        rows = [{"items": []}, {"items": [{"qty": 5}]}]
        p = self.write(schema, rows)
        out = list(scan(p, [["items", []]],
                        ["any", ["items"], ["eq", [[]], 5]]))
        self.assertEqual(out, [{"items": [{"qty": 5}]}])

    def test_any_null_list_is_unknown(self):
        schema = [list3("items", leaf("qty", TYPE_INT64))]
        rows = [{"items": None}, {"items": [{"qty": 5}]}]
        p = self.write(schema, rows)
        # null 列表 -> any 未知，not(any) 也是未知，均不命中
        out = list(scan(p, [["items", []]],
                        ["not", ["any", ["items"], ["eq", [[]], 5]]]))
        self.assertEqual(out, [])

    def test_nested_any_through_two_list_levels(self):
        schema = [list3("groups", list3("vals", leaf("n", TYPE_INT64)))]
        rows = [
            {"groups": [{"vals": [{"n": 1}, {"n": 2}]}, {"vals": []}]},
            {"groups": []},
            {"groups": None},
            {"groups": [{"vals": None}]},
        ]
        p = self.write(schema, rows)
        where = ["any", ["groups", [], "vals"], ["eq", [[]], 2]]
        out = list(scan(p, [["groups", [], "vals", []]], where))
        # 仅第一行的内层列表含 n==2；空/null 列表按三值逻辑不命中
        self.assertEqual([r is not None for r in out], [True])
        # 不存在的值不命中任何行
        none_hit = list(scan(p, [["groups", [], "vals", []]],
                             ["any", ["groups", [], "vals"],
                              ["eq", [[]], 99]]))
        self.assertEqual(none_hit, [])

    def test_eq_null_is_unknown_not_match(self):
        schema = [leaf("v", TYPE_INT64)]
        rows = [{"v": None}, {"v": 1}]
        p = self.write(schema, rows)
        self.assertEqual(list(scan(p, [["v"]], ["eq", ["v"], 1])),
                         [{"v": 1}])

    def test_predicate_leaf_not_in_output(self):
        schema = [leaf("x", TYPE_INT64), leaf("y", TYPE_INT64)]
        rows = [{"x": 1, "y": 2}]
        p = self.write(schema, rows)
        # 只投影 x，用 y 做谓词；y 不应出现在结果里
        out = list(scan(p, [["x"]], ["eq", ["y"], 2]))
        self.assertEqual(out, [{"x": 1}])

    def test_utf8_eq(self):
        schema = [leaf("s", TYPE_BYTE_ARRAY, utf8=True)]
        rows = [{"s": "a"}, {"s": "b"}, {"s": None}]
        p = self.write(schema, rows)
        out = list(scan(p, [["s"]], ["eq", ["s"], "b"]))
        self.assertEqual(out, [{"s": "b"}])


class ErrorHandlingTest(Base):
    def test_truncated_file_reports_offset(self):
        schema = [leaf("x", TYPE_INT64)]
        p = self.write(schema, [{"x": 1}, {"x": 2}])
        with open(p, "r+b") as f:
            f.truncate(os.path.getsize(p) - 9)
        with self.assertRaises(NestedScanError) as ctx:
            list(scan(p, [["x"]]))
        self.assertIsNotNone(ctx.exception.offset)

    def test_illegal_level_reports_offset(self):
        # 两层 LIST 的叶 max_def=5（3 位，可表示 0..7）：把 DL 数据字节
        # 置全 1 会解出 7 > 5，干净地触发非法级值而不破坏值计数。
        schema = [list3("groups", list3("vals", leaf("n", TYPE_INT64)))]
        rows = [{"groups": [{"vals": [{"n": 1}, {"n": 2}]}]}]
        p = self.write(schema, rows, corrupt="illegal_level")
        with self.assertRaises(IllegalLevelError) as ctx:
            list(scan(p, [["groups", [], "vals", []]]))
        self.assertIsNotNone(ctx.exception.offset)

    def test_invalid_path_rejected(self):
        schema = [leaf("x", TYPE_INT64)]
        p = self.write(schema, [{"x": 1}])
        with self.assertRaises(InvalidPathError):
            list(scan(p, [["nope"]]))
        with self.assertRaises(InvalidPathError):
            list(scan(p, []))
        with self.assertRaises(InvalidPathError):
            list(scan(p, [["x", "deep"]]))

    def test_predicate_literal_type_mismatch(self):
        schema = [leaf("x", TYPE_INT64)]
        p = self.write(schema, [{"x": 1}])
        with self.assertRaises(InvalidPredicateError):
            list(scan(p, [["x"]], ["eq", ["x"], "str"]))

    def test_unknown_predicate_op(self):
        schema = [leaf("x", TYPE_INT64)]
        p = self.write(schema, [{"x": 1}])
        with self.assertRaises(InvalidPredicateError):
            list(scan(p, [["x"]], ["bogus", ["x"], 1]))

    def test_bad_magic_rejected(self):
        p = self.path()
        with open(p, "wb") as f:
            f.write(b"XXXX" + b"\x00" * 20)
        with self.assertRaises(NestedScanError):
            list(scan(p, [["x"]]))

    def test_compressed_v2_page_rejected(self):
        schema = [leaf("x", TYPE_INT64)]
        rows = [{"x": 1}, {"x": 2}]
        p = self.write(schema, rows, page_version="v2",
                       corrupt="compressed")
        with self.assertRaises(UnsupportedFormatError):
            list(scan(p, [["x"]]))


class DictionaryErrorTest(Base):
    def test_dictionary_index_out_of_bounds(self):
        # 3 个不同值 -> 字典 3 项、索引位宽 2（可出现索引 2）。
        # 把字典页 PLAIN 载荷截断为只剩 2 项，数据页仍引用索引 2，
        # 从而触发字典索引越界。
        schema = [leaf("s", TYPE_BYTE_ARRAY, utf8=True)]
        rows = [{"s": "a"}, {"s": "b"}, {"s": "c"}, {"s": "a"}]
        p = self.write(schema, rows, use_dict=True)
        self._truncate_dictionary(p, keep_entries=2)
        with self.assertRaises(DictionaryOutOfBoundsError) as ctx:
            list(scan(p, [["s"]]))
        self.assertIsNotNone(ctx.exception.offset)

    def _truncate_dictionary(self, path, keep_entries):
        """重建文件：缩短字典页 PLAIN 载荷到 keep_entries 项，
        但保留数据页的原始索引（含 >= keep_entries 的越界索引）。

        通过定位字典页并重写其压缩大小与载荷实现。
        """
        from nestedscan.metadata import (
            parse_page_header,
            PAGE_DICTIONARY,
        )
        with open(path, "rb") as f:
            data = bytearray(f.read())
        off = 4
        ph, hs = parse_page_header(bytes(data[off:off + 64]), off)
        assert ph.type == PAGE_DICTIONARY
        dict_header = ph.dictionary_page_header
        payload_start = off + hs
        # PLAIN BYTE_ARRAY：每项 = 4 字节长度 + 数据
        payload = bytes(data[payload_start:
                            payload_start + ph.compressed_page_size])
        pos = 0
        for _ in range(keep_entries):
            length = int.from_bytes(payload[pos:pos + 4], "little")
            pos += 4 + length
        new_payload = payload[:pos]
        # 重写页头的两个大小字段并替换载荷：直接在测试里重建整个字典页
        from parquet_writer import write_page_header
        new_header = write_page_header({
            "type": PAGE_DICTIONARY,
            "uncompressed": len(new_payload),
            "compressed": len(new_payload),
            "dict": {"num_values": keep_entries,
                     "encoding": dict_header.encoding},
        })
        rebuilt = bytes(data[:off]) + new_header + new_payload \
            + bytes(data[payload_start + ph.compressed_page_size:])
        with open(path, "wb") as f:
            f.write(rebuilt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
