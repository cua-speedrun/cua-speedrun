"""Guest-side walk of the front window's accessibility tree; imported on the host without AT-SPI.

The source is sent with each request, like the keyboard script. It reads only
the active window of the active application, keeps on-screen nodes that have
a name, text, or an editable or focusable state, lists at most MAX_CHILDREN
children per node, and stops at fixed node and time budgets, so one walk costs
a bounded amount of the measured observation.
"""

import json
import time

MAX_VISITED = 4000
MAX_KEPT = 600
MAX_DEPTH = 60
MAX_SECONDS = 1.0
MAX_TEXT = 200
# A spreadsheet's table reports every cell as a child; listing them all blocks.
MAX_CHILDREN = 200


def walk():
    import gi

    gi.require_version("Atspi", "2.0")
    from gi.repository import Atspi

    started = time.monotonic()
    state = Atspi.StateType
    window, app_name = None, ""
    desktop = Atspi.get_desktop(0)
    for i in range(desktop.get_child_count()):
        app = desktop.get_child_at_index(i)
        if app is None:
            continue
        for j in range(app.get_child_count()):
            candidate = app.get_child_at_index(j)
            if candidate is not None and candidate.get_state_set().contains(state.ACTIVE):
                window, app_name = candidate, app.get_name() or ""
                break
        if window is not None:
            break
    result = {"app": app_name, "window": "", "nodes": [], "truncated": False}
    if window is None:
        result["seconds"] = time.monotonic() - started
        return result
    result["window"] = window.get_name() or ""

    visited = 0
    stack = [(window, 0)]
    while stack:
        if visited >= MAX_VISITED or time.monotonic() - started > MAX_SECONDS:
            result["truncated"] = True
            break
        node, depth = stack.pop()
        visited += 1
        try:
            states = node.get_state_set()
            if not (states.contains(state.SHOWING) and states.contains(state.VISIBLE)):
                continue
        except Exception:
            # A node can vanish mid-walk when the app redraws; skip it.
            continue
        try:
            box = node.get_extents(Atspi.CoordType.SCREEN)
            name = (node.get_name() or "").strip()
            text = ""
            try:
                count = Atspi.Text.get_character_count(node)
                if count:
                    text = Atspi.Text.get_text(node, 0, min(count, MAX_TEXT)).strip()
            except Exception:
                pass
            editable = states.contains(state.EDITABLE)
            focused = states.contains(state.FOCUSED)
            if box.width > 0 and box.height > 0 and (name or text or editable or focused):
                result["nodes"].append({
                    "role": node.get_role_name(),
                    "name": name[:MAX_TEXT],
                    "text": text if text != name else "",
                    "x": box.x, "y": box.y, "w": box.width, "h": box.height,
                    "focused": focused,
                    "editable": editable,
                    "checked": states.contains(state.CHECKED),
                    "selected": states.contains(state.SELECTED),
                    "enabled": states.contains(state.ENABLED),
                })
                if len(result["nodes"]) >= MAX_KEPT:
                    result["truncated"] = True
                    break
        except Exception:
            # Containers without a geometry or text interface still hold children.
            pass
        if depth < MAX_DEPTH:
            children = []
            try:
                for k in range(min(node.get_child_count(), MAX_CHILDREN)):
                    if time.monotonic() - started > MAX_SECONDS:
                        result["truncated"] = True
                        break
                    children.append(node.get_child_at_index(k))
            except Exception:
                continue
            # Reverse so the walk visits children in their screen order.
            stack.extend((child, depth + 1) for child in reversed(children) if child is not None)
    result["seconds"] = time.monotonic() - started
    return result


def main():
    print(json.dumps(walk()))
