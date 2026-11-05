import json

from mood_steering.eval.resumable import (
    clean_for_append,
    dump_row,
    part_path,
    read_rows,
    write_rows_atomic,
)

ROWS = [
    {"item_id": "a", "output": "x\u2028y\u2029z\u0085w"},
    {"item_id": "b", "output": "café"},
]


def _raw(rows):
    return "".join(dump_row(r) for r in rows).encode("utf-8")


def test_dump_row_keeps_accents_and_ends_with_newline():
    line = dump_row({"o": "é\u2028"})
    assert line == '{"o": "é\u2028"}\n'


def test_read_rows_missing_file_is_empty(tmp_path):
    assert read_rows(tmp_path / "absent.jsonl") == []


def test_read_rows_keeps_unicode_separators_and_drops_truncated_last_line(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_bytes(_raw(ROWS) + b'{"item_id": "c", "outp')
    assert read_rows(path) == ROWS


def test_read_rows_drops_line_cut_inside_a_multibyte_character(tmp_path):
    path = tmp_path / "a.jsonl"
    cut = dump_row({"item_id": "c", "output": "é"}).encode("utf-8")[:-4]  # coupe dans « é »
    path.write_bytes(_raw(ROWS) + cut)
    assert read_rows(path) == ROWS


def test_read_rows_stops_at_first_unreadable_line(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_bytes(_raw(ROWS[:1]) + b"pas du json\n" + _raw(ROWS[1:]))
    assert read_rows(path) == ROWS[:1]


def test_write_rows_atomic_leaves_no_part_and_is_readable(tmp_path):
    path = tmp_path / "sous" / "a.jsonl"
    write_rows_atomic(path, ROWS)
    assert not part_path(path).exists()
    assert path.read_bytes() == _raw(ROWS)
    assert [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x] == ROWS


def test_clean_for_append_rewrites_without_truncated_line(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_bytes(_raw(ROWS) + b'{"item_id": "c"')
    assert clean_for_append(path) == ROWS
    assert path.read_bytes() == _raw(ROWS)  # un ajout ne se colle plus à la ligne coupée
    assert not part_path(path).exists()


def test_clean_for_append_does_not_create_missing_file(tmp_path):
    assert clean_for_append(tmp_path / "absent.jsonl") == []
    assert not (tmp_path / "absent.jsonl").exists()
