"""Tests for opto.trace.io.bindings."""
import pytest
from opto.trace.io.bindings import Binding, apply_updates, make_dict_binding


class TestBinding:
    def test_basic_get_set(self):
        store = {"val": "hello"}
        b = Binding(get=lambda: store["val"], set=lambda v: store.__setitem__("val", v))
        assert b.get() == "hello"
        b.set("world")
        assert store["val"] == "world"

    def test_kind_default(self):
        b = Binding(get=lambda: None, set=lambda v: None)
        assert b.kind == "prompt"

    def test_kind_code(self):
        b = Binding(get=lambda: None, set=lambda v: None, kind="code")
        assert b.kind == "code"


class TestApplyUpdates:
    def test_apply_single(self):
        store = {"prompt": "old"}
        bindings = {"prompt": make_dict_binding(store, "prompt")}
        apply_updates({"prompt": "new"}, bindings)
        assert store["prompt"] == "new"

    def test_apply_multiple(self):
        store = {"a": "1", "b": "2"}
        bindings = {
            "a": make_dict_binding(store, "a"),
            "b": make_dict_binding(store, "b"),
        }
        apply_updates({"a": "X", "b": "Y"}, bindings)
        assert store == {"a": "X", "b": "Y"}

    def test_strict_missing_key_raises(self):
        bindings = {"a": make_dict_binding({}, "a")}
        with pytest.raises(KeyError, match="no binding for key 'z'"):
            apply_updates({"z": "val"}, bindings, strict=True)

    def test_non_strict_missing_key_skips(self):
        store = {"a": "old"}
        bindings = {"a": make_dict_binding(store, "a")}
        apply_updates({"a": "new", "z": "skip"}, bindings, strict=False)
        assert store["a"] == "new"

    def test_empty_updates(self):
        store = {"a": "old"}
        bindings = {"a": make_dict_binding(store, "a")}
        apply_updates({}, bindings)
        assert store["a"] == "old"


class TestMakeDictBinding:
    def test_roundtrip(self):
        store = {"key": "initial"}
        b = make_dict_binding(store, "key")
        assert b.get() == "initial"
        b.set("updated")
        assert b.get() == "updated"
        assert store["key"] == "updated"

    def test_missing_key_returns_none(self):
        store = {}
        b = make_dict_binding(store, "missing")
        assert b.get() is None
