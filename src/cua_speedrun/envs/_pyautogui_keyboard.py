"""Guest-side PyAutoGUI corrections for X11; imported on the host without X.

The source is sent with each keyboard action. No guest package installation,
clipboard, application-specific input method, or background service is needed.
"""

import ctypes as C
from itertools import combinations
import time


MAPPING_SETTLE_SEC = 0.12


KEYSYMS = {
    "enter": "Return", "return": "Return", "\n": "Return", "\r": "Return",
    "esc": "Escape", "escape": "Escape", "backspace": "BackSpace", "\b": "BackSpace",
    "tab": "Tab", "\t": "Tab", "space": "space",
    "ctrl": "Control_L", "control": "Control_L", "ctrlleft": "Control_L",
    "ctrlright": "Control_R", "shift": "Shift_L", "shiftleft": "Shift_L",
    "shiftright": "Shift_R", "alt": "Alt_L", "altleft": "Alt_L", "altright": "Alt_R",
    "meta": "Super_L", "super": "Super_L", "cmd": "Super_L", "command": "Super_L",
    "win": "Super_L", "winleft": "Super_L", "winright": "Super_R",
    "delete": "Delete", "del": "Delete", "insert": "Insert", "ins": "Insert",
    "pageup": "Prior", "pgup": "Prior", "page_up": "Prior",
    "pagedown": "Next", "pgdn": "Next", "page_down": "Next",
    "kp_enter": "KP_Enter", "kp_add": "KP_Add", "add": "KP_Add",
    "kp_subtract": "KP_Subtract", "subtract": "KP_Subtract",
    "kp_multiply": "KP_Multiply", "multiply": "KP_Multiply",
    "kp_divide": "KP_Divide", "divide": "KP_Divide",
    "menu": "Menu", "apps": "Menu", "caps_lock": "Caps_Lock", "capslock": "Caps_Lock",
    "num_lock": "Num_Lock", "numlock": "Num_Lock", "print": "Print",
    "home": "Home", "end": "End", "left": "Left", "right": "Right",
    "up": "Up", "down": "Down",
    **{f"f{n}": f"F{n}" for n in range(1, 25)},
}


def key_names(value):
    values = value if isinstance(value, (list, tuple)) else (
        [value] if len(str(value)) == 1 else str(value).split("+")
    )
    names = [str(key) if len(str(key)) == 1 else str(key).strip() for key in values]
    return [KEYSYMS.get(key.lower(), key) for key in names]


def text_segments(text, mapped, capacity):
    """Split only when the number of *unmapped* characters exhausts spare keys."""
    start, missing = 0, set()
    for index, char in enumerate(text):
        if char in mapped or char in missing:
            continue
        if not capacity:
            raise RuntimeError("X11 has no unused keycodes for out-of-layout text")
        if len(missing) == capacity:
            yield text[start:index]
            start, missing = index, set()
        missing.add(char)
    if start < len(text):
        yield text[start:]


def mapping_runs(rows):
    """Batch adjacent spare keys without rewriting occupied keys between them."""
    run = []
    for code in sorted(rows):
        if run and code != start + len(run):
            yield start, run
            run = []
        if not run:
            start = code
        run.append(rows[code])
    if run:
        yield start, run


class _XkbState(C.Structure):
    # ABI from X11/extensions/XKBstr.h (not the obsolete struct in the manual).
    _fields_ = [
        ("group", C.c_ubyte), ("locked_group", C.c_ubyte),
        ("base_group", C.c_ushort), ("latched_group", C.c_ushort),
        *[(name, C.c_ubyte) for name in (
            "mods", "base_mods", "latched_mods", "locked_mods", "compat_state",
            "grab_mods", "compat_grab_mods", "lookup_mods", "compat_lookup_mods",
        )],
        ("ptr_buttons", C.c_ushort),
    ]


class Keyboard:
    def __init__(self):
        import pyautogui

        self.pg = pyautogui
        self.backend = pyautogui.platformModule
        self.display = self.backend._display
        self.pg.FAILSAFE = False  # Same setting as the runner's mouse path.
        self.pg.PAUSE = 0
        self.down, self.up = self.backend._keyDown, self.backend._keyUp
        self.x11 = C.CDLL("libX11.so.6")
        self.xkb = C.CDLL("libxkbcommon.so.0")
        self.xkb.xkb_keysym_to_utf32.argtypes = [C.c_uint32]
        self.xkb.xkb_keysym_to_utf32.restype = C.c_uint32
        self.xkb.xkb_keysym_from_name.argtypes = [C.c_char_p, C.c_int]
        self.xkb.xkb_keysym_from_name.restype = C.c_uint32
        signatures = {
            "XOpenDisplay": (C.c_void_p, [C.c_char_p]),
            "XCloseDisplay": (C.c_int, [C.c_void_p]),
            "XKeysymToKeycode": (C.c_ubyte, [C.c_void_p, C.c_ulong]),
            "XkbGetState": (C.c_int, [C.c_void_p, C.c_uint, C.POINTER(_XkbState)]),
            "XkbKeysymToModifiers": (C.c_uint, [C.c_void_p, C.c_ulong]),
            "XkbLookupKeySym": (C.c_int, [C.c_void_p, C.c_ubyte, C.c_uint,
                                         C.POINTER(C.c_uint), C.POINTER(C.c_ulong)]),
        }
        for name, (result, args) in signatures.items():
            function = getattr(self.x11, name)
            function.restype, function.argtypes = result, args
        self.connection = self.x11.XOpenDisplay(None)
        if not self.connection:
            raise RuntimeError("Cannot open the X11 display for keyboard layout lookup")
        self.bindings = {}
        self.modifier_groups = {}
        self.named = {}
        self.temporary = {}
        self.original_rows = {}
        info = self.display.display.info
        self.keycodes = range(info.min_keycode, info.max_keycode + 1)
        # PyAutoGUI's integer-keycode path avoids its US-only Shift heuristic.
        self.backend.keyboardMapping.update({code: code for code in self.keycodes})
        self.backend._keyDown, self.backend._keyUp = self.key_down, self.key_up

    def symbol(self, name):
        encoded = name.encode("utf-8")
        return (self.xkb.xkb_keysym_from_name(encoded, 0)
                or self.xkb.xkb_keysym_from_name(encoded, 1))

    def layout(self):
        state = _XkbState()
        if self.x11.XkbGetState(self.connection, 0x100, C.byref(state)) != 0:
            raise RuntimeError("Cannot read the active XKB keyboard group")
        modifiers = []
        for name in ("Shift_L", "ISO_Level3_Shift", "Mode_switch", "ISO_Level5_Shift"):
            symbol = self.symbol(name)
            code = self.x11.XKeysymToKeycode(self.connection, symbol)
            mask = self.x11.XkbKeysymToModifiers(self.connection, symbol)
            if code and mask and mask not in [entry[0] for entry in modifiers]:
                modifiers.append((mask, code))
        modmap = self.display.get_modifier_mapping()
        self.modifier_groups = {
            code: {key for bit, row in enumerate(modmap) if mask & (1 << bit)
                   for key in row if key}
            for mask, code in modifiers
        }
        self.bindings = {}
        # Prefer fewer modifiers. XKB resolves groups and key types; flattened
        # core keymap columns are NOT reliable Shift/AltGr level indices.
        consumed, symbol = C.c_uint(), C.c_ulong()
        for count in range(len(modifiers) + 1):
            for subset in combinations(modifiers, count):
                mask = 0
                for bitmask, _ in subset:
                    mask |= bitmask
                for code in self.keycodes:
                    if self.x11.XkbLookupKeySym(
                        self.connection, code, (state.group << 13) | mask,
                        C.byref(consumed), C.byref(symbol),
                    ):
                        value = self.xkb.xkb_keysym_to_utf32(symbol.value)
                        if value:
                            self.bindings.setdefault(chr(value), (code, tuple(k for _, k in subset)))

    @staticmethod
    def char_symbol(char):
        value = ord(char)
        if 0xD800 <= value <= 0xDFFF:
            raise ValueError("Text contains an unpaired Unicode surrogate")
        if value < 0x20 or 0x7F <= value < 0xA0:
            raise ValueError(f"Control character requires a named keyboard action: U+{value:04X}")
        return value if value <= 0xFF else 0x01000000 | value

    def resolve(self, key):
        if key in self.named:
            return self.named[key], ()
        if key in self.temporary:
            return self.temporary[key], ()
        if len(key) == 1:
            binding = self.bindings.get(key)
            if binding:
                return binding
        code = self.backend.keyboardMapping.get(key)
        if code and len(key) > 1:
            return code, ()
        raise ValueError(f"Key is not available on the active X11 keyboard: {key!r}")

    def register(self, names):
        for name in names:
            symbol_name = KEYSYMS.get(name.lower(), name)
            if len(symbol_name) == 1:
                continue
            symbol = self.symbol(symbol_name)
            code = self.x11.XKeysymToKeycode(self.connection, symbol) if symbol else 0
            if code:
                self.named[name.lower()] = code

    def key_down(self, key):
        code, modifiers = self.resolve(key)
        added = []
        try:
            if modifiers:
                bitmap = self.display.query_keymap()
                held = {k for k in self.keycodes if bitmap[k // 8] & (1 << (k % 8))}
                for modifier in modifiers:
                    # Do not release a Shift/AltGr already held by the agent,
                    # including the right-hand counterpart of a left modifier.
                    if not held.intersection(self.modifier_groups[modifier]):
                        added.append(modifier)
                        self.down(modifier)
            self.down(code)
        finally:
            for modifier in reversed(added):
                self.up(modifier)

    def key_up(self, key):
        self.up(self.resolve(key)[0])

    def change_rows(self, rows):
        from Xlib.error import CatchError

        # Do not round-trip occupied rows through the flattened core keymap:
        # that can change their XKB types even when the keysyms look identical.
        error = CatchError()
        for first, keysyms in mapping_runs(rows):
            self.display.change_keyboard_mapping(first, keysyms, onerror=error)
        self.display.sync()
        if error.get_error() is not None:
            raise RuntimeError(f"X11 rejected the temporary keyboard mapping: {error.get_error()}")

    def type_text(self, text):
        self.layout()
        self.register(["\n", "\r", "\t", "\b"])
        mapped = set(self.named) | set(self.bindings)
        for char in set(text) - mapped:
            self.char_symbol(char)  # Validate before emitting any text.
        rows = self.display.get_keyboard_mapping(self.keycodes.start, len(self.keycodes))
        modifiers = {key for row in self.display.get_modifier_mapping() for key in row}
        held = self.display.query_keymap()
        spare = {code: row for code, row in zip(self.keycodes, rows)
                 if not any(row) and code not in modifiers
                 and not held[code // 8] & (1 << (code % 8))}
        # No keymap mutation (or map-settle sleep) at all for layout-supported
        # text, irrespective of its length.
        try:
            for segment in text_segments(text, mapped, len(spare)):
                missing = list(dict.fromkeys(char for char in segment if char not in mapped))
                if missing:
                    # Keep bindings for characters shared with the next segment.
                    self.temporary = {char: code for char, code in self.temporary.items()
                                      if char in missing}
                    codes = iter(code for code in spare if code not in self.temporary.values())
                    updates = {}
                    for char in missing:
                        if char in self.temporary:
                            continue
                        code = next(codes)
                        self.temporary[char] = code
                        updates[code] = [self.char_symbol(char)] * len(spare[code])
                    if updates and self.original_rows:
                        time.sleep(0.30)
                    for code in updates:
                        self.original_rows.setdefault(code, spare[code])
                    if updates:
                        self.change_rows(updates)
                        time.sleep(MAPPING_SETTLE_SEC)
                self.pg.write(segment, interval=0.006)
        finally:
            if self.original_rows:
                # XSync is not an acknowledgement from the focused app.
                # Retain the existing Unicode-consumption allowance before
                # reusing or restoring temporary keys, including on errors.
                time.sleep(0.30)
                self.change_rows(self.original_rows)
                self.original_rows.clear()
                self.temporary.clear()

    def chord(self, names):
        self.layout()
        self.register(names)
        for name in names:
            self.resolve(name.lower() if len(name) > 1 else name)
        self.pg.hotkey(*names, interval=0.01)

    def hold(self, names, down):
        self.layout()
        self.register(names)
        for name in names:
            self.resolve(name.lower() if len(name) > 1 else name)
        function = self.pg.keyDown if down else self.pg.keyUp
        for name in names:
            function(name)
            time.sleep(0.01)

    def close(self):
        self.backend._keyDown, self.backend._keyUp = self.down, self.up
        self.x11.XCloseDisplay(self.connection)


def run_keyboard(keyboard):
    driver = Keyboard()
    try:
        if "text" in keyboard:
            driver.type_text(str(keyboard["text"]))
        if "keys" in keyboard:
            driver.chord(key_names(keyboard["keys"]))
        for field in ("key_down", "keys_down"):
            if field in keyboard:
                driver.hold(key_names(keyboard[field]), True)
        for field in ("key_up", "keys_up"):
            if field in keyboard:
                driver.hold(key_names(keyboard[field]), False)
    finally:
        driver.close()
