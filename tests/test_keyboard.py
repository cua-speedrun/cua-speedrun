"""Offline payload/segmentation checks, not a replacement for auto-harness."""

import ast
import base64
import ctypes
import inspect
import json
import string

import pytest

from cua_speedrun.envs import _pyautogui_keyboard as guest
from cua_speedrun.envs.keyboard import keyboard_script
from cua_speedrun.envs.modal_native import _keyboard_script


def decode_payload(script):
    tree = ast.parse(script)
    encoded = tree.body[-1].value.args[0].args[0].args[0]
    return json.loads(base64.b64decode(ast.literal_eval(encoded)))


@pytest.mark.parametrize("text", [
    "He said \"it's $HOME\"; `whoami`; $(id) \\ ",
    "!@#$%^&*()_+-=[]{}<>?,./~",
    "naïve café über Straße 東京 Ελληνικά",
    "😀🚀", "مرحبا שלום", "e\u0301 a\u0308", "👨‍👩‍👧‍👦 🇮🇳",
    "line one\n\tline two\r\n", "a = 1 < 2\n" * 10000,
])
def test_text_is_serialized_losslessly_as_data(text):
    action = {"text": text}
    script = keyboard_script(action)
    compile(script, "<guest keyboard>", "exec")
    assert decode_payload(script) == action
    assert script.isascii()
    assert _keyboard_script(action) == script


@pytest.mark.parametrize("name,expected", [
    ("kp_enter", "KP_Enter"), ("kp_add", "KP_Add"),
    ("menu", "Menu"), ("caps_lock", "Caps_Lock"),
    ("capslock", "Caps_Lock"), ("apps", "Menu"),
    ("CTRL", "Control_L"), ("cmd", "Super_L"),
    ("enter", "Return"), ("\n", "Return"), ("\t", "Tab"),
    ("<", "<"), (">", ">"), ("+", "+"), (" ", " "),
    ("XF86AudioPlay", "XF86AudioPlay"),
])
def test_key_vocabulary(name, expected):
    assert guest.key_names(name) == [expected]
    assert guest.key_names([name]) == [expected]


def test_chords_and_both_hold_spellings_survive_transport():
    action = {"keys": "ctrl+shift+pagedown", "key_down": "shift",
              "keys_down": ["alt"], "key_up": "ctrl", "keys_up": ["alt"]}
    assert decode_payload(keyboard_script(action)) == action
    assert guest.key_names(action["keys"]) == ["Control_L", "Shift_L", "Next"]


def test_layout_supported_text_never_requires_spare_keys():
    # Pure segmentation logic: no display, injected event, or pretend VM.
    text = string.printable * 1000
    assert list(guest.text_segments(text, set(text), 0)) == [text]
    assert list(guest.text_segments("", set(), 0)) == []


def test_only_out_of_layout_vocabulary_consumes_capacity():
    text = ("print('<>'); " * 1000) + "東京" * 1000
    assert list(guest.text_segments(text, set(string.printable), 2)) == [text]


@pytest.mark.parametrize("capacity", [1, 2, 5, 19, 40])
def test_unicode_segmentation_does_not_drop_reorder_or_normalize(capacity):
    text = "".join(chr(n) for n in range(0x400, 0x460)) + "e\u0301👨‍👩‍👧‍👦\n" * 10
    mapped = set(string.printable)
    segments = list(guest.text_segments(text, mapped, capacity))
    assert "".join(segments) == text
    assert all(len(set(segment) - mapped) <= capacity for segment in segments)


def test_missing_spare_keys_raise_instead_of_dropping_text():
    with pytest.raises(RuntimeError, match="no unused keycodes"):
        list(guest.text_segments("ASCII then 東京", set(string.printable), 0))


def test_mapping_updates_never_span_occupied_keys():
    rows = {200: (1, 1), 201: (2, 2), 204: (3, 3), 207: (4, 4), 208: (5, 5)}
    assert list(guest.mapping_runs(rows)) == [
        (200, [(1, 1), (2, 2)]), (204, [(3, 3)]), (207, [(4, 4), (5, 5)]),
    ]
    assert list(guest.mapping_runs({})) == []


@pytest.mark.parametrize("char,keysym", [
    ("<", 0x3C), ("é", 0xE9), ("東", 0x01006771),
    ("😀", 0x0101F600), ("\u0301", 0x01000301), ("\u200d", 0x0100200D),
])
def test_unicode_keysym_encoding(char, keysym):
    assert guest.Keyboard.char_symbol(char) == keysym


@pytest.mark.parametrize("char", ["\ud800", "\x00", "\x7f"])
def test_non_key_characters_are_not_silently_ignored(char):
    with pytest.raises(ValueError):
        guest.Keyboard.char_symbol(char)


def test_xkb_state_abi_matches_published_header():
    assert ctypes.sizeof(guest._XkbState) == 18
    assert guest._XkbState.group.offset == 0
    assert guest._XkbState.base_group.offset == 2
    assert guest._XkbState.latched_group.offset == 4
    assert guest._XkbState.mods.offset == 6
    assert guest._XkbState.ptr_buttons.offset == 16


def test_input_dispatch_stays_in_pyautogui():
    source = inspect.getsource(guest)
    assert "self.pg.write(segment, interval=0.006)" in source
    assert "self.pg.hotkey(*names, interval=0.01)" in source
    assert "self.pg.keyDown if down else self.pg.keyUp" in source
    assert "XkbLookupKeySym" in source
    assert "isShiftCharacter" not in source
    assert "xtest.fake_input" not in source
