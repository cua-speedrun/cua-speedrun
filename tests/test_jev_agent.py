"""Check the Jev agent's request shape and action translation; no environment doubles."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("jev_agent", ROOT / "agents/jev/agent.py")
assert spec is not None and spec.loader is not None
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)

PARSED = {
    "texts": [{"text": "Restore", "box": [0.95, 0.28, 0.98, 0.30]},
              {"text": "Activities", "box": [0.01, 0.0, 0.04, 0.02]}],
    "icons": [{"box": [0.0, 0.05, 0.03, 0.08]}, {"box": [0.5, 0.5, 0.52, 0.52]}],
}


def test_jev_template_is_discoverable_and_valid():
    from cua_speedrun.commands.benchmark import validate_agent
    from cua_speedrun.service.templates_catalog import list_templates, template_dir

    entry = next(item for item in list_templates() if item["name"] == "jev")
    assert entry["gpu"] == "L40S"
    assert entry["required_environment_variables"] == ["OPENROUTER_API_KEY"]
    validate_agent(template_dir("jev"))


def test_typed_text_comes_only_from_the_task():
    task = ('Rename the folder "Other bookmarks" to "Archive 2026", save it as report.pdf, '
            "email jo@example.com, and open https://example.com/a. Set the margin to 2.5%.")
    assert agent.text_options(task) == [
        "Other bookmarks", "Archive 2026", "https://example.com/a", "jo@example.com", "report.pdf", "2.5%",
    ]
    assert agent.text_options("Close the dialog.") == []


def test_parsed_elements_are_numbered_in_reading_order_with_pixel_centers():
    elements = agent.screen_elements(PARSED, 1920, 1080)
    assert [e["label"] for e in elements] == [
        'text "Activities" (top-left, x=48 y=11)',
        'icon near "Activities" (top-left, x=29 y=70)',
        'text "Restore" (top-right, x=1853 y=313)',
        "icon (middle-center, x=979 y=551)",
    ]
    assert [e["id"] for e in elements] == ["e1", "e2", "e3", "e4"]
    assert elements[2]["box"] == [1824, 302, 1882, 324]


def test_tree_elements_keep_enabled_on_screen_controls_with_their_state():
    node = {"text": "", "focused": False, "editable": False, "checked": False,
            "selected": False, "enabled": True, "w": 40, "h": 20}
    tree = {"app": "gedit", "window": "notes.txt", "nodes": [
        {**node, "role": "frame", "name": "notes.txt", "x": 0, "y": 0, "w": 1920, "h": 1080},
        {**node, "role": "push button", "name": "Save", "x": 1800, "y": 40},
        {**node, "role": "push button", "name": "Save", "x": 1800, "y": 40},
        {**node, "role": "text", "name": "", "text": "hello", "x": 100, "y": 200,
         "focused": True, "editable": True},
        {**node, "role": "push button", "name": "Undo", "x": 60, "y": 40, "enabled": False},
        {**node, "role": "push button", "name": "Off screen", "x": 2400, "y": 40},
    ]}
    assert [e["label"] for e in agent.tree_elements(tree, 1920, 1080)] == [
        'push button "Save" (top-right, x=1820 y=50)',
        'text with text "hello" [focused, editable] (top-left, x=120 y=210)',
    ]


def test_request_offers_only_actions_it_can_carry_out():
    elements = agent.screen_elements(PARSED, 1920, 1080)
    request = agent.build_request("Close the dialog.", elements, [], [], [])
    assert request["model"] == agent.MODEL
    assert set(request["questions"]) == {"action", "key", "element"}
    assert not {"type", "type_into"} & set(request["questions"]["action"]["criteria"])
    assert all(len(q["criteria"]) <= 255 for q in request["questions"].values())
    blank = agent.build_request("Close the dialog.", [], ["x"], ["clicked e1"], [])
    assert not {"click", "double_click", "right_click", "type_into"} & set(blank["questions"]["action"]["criteria"])
    assert blank["state"]["previous_actions"] == ["clicked e1"]


def test_answers_become_environment_actions():
    elements = agent.screen_elements(PARSED, 1920, 1080)
    answers = {"element": {"choice": "e2"}, "text": {"choice": "t2"}, "key": {"choice": "save"}}
    texts = ["Other bookmarks", "Archive 2026"]
    assert agent.step_actions("click", answers, elements, texts, 1920, 1080)[0] == [
        {"mouse": {"left_click": [29, 70]}}]
    assert agent.step_actions("type", answers, elements, texts, 1920, 1080)[0] == [
        {"keyboard": {"text": "Archive 2026"}}]
    assert agent.step_actions("type_into", answers, elements, texts, 1920, 1080)[0] == [
        {"mouse": {"left_click": [29, 70]}}, {"keyboard": {"text": "Archive 2026"}}]
    assert agent.step_actions("key", answers, elements, texts, 1920, 1080)[0] == [
        {"keyboard": {"keys": ["ctrl", "s"]}}]
    assert agent.step_actions("scroll_down", answers, elements, texts, 1920, 1080)[0] == [
        {"mouse": {"move": [29, 70], "scroll": 5}}]
    assert agent.step_actions("scroll_up", {}, [], texts, 1920, 1080)[0] == [
        {"mouse": {"move": [960, 540], "scroll": -5}}]


def test_a_tree_shifted_from_the_screen_is_not_trusted():
    node = {"text": "", "w": 200, "h": 20, "focused": False, "editable": False, "checked": False,
            "selected": False, "enabled": True, "role": "label"}
    tree = {"nodes": [{**node, "name": "English (USA)", "x": 900, "y": 1000},
                      {**node, "name": "100%", "x": 1700, "y": 1000},
                      {**node, "name": "Get involved", "x": 1500, "y": 200}]}

    def screen(dy):
        # Text drawn dy pixels below where the tree puts it, except the last.
        return {"texts": [{"text": "English (USA)", "box": [1000 / 1920, (1005 + dy) / 1080, 1050 / 1920, (1015 + dy) / 1080]},
                          {"text": "100%", "box": [1790 / 1920, (1005 + dy) / 1080, 1810 / 1920, (1015 + dy) / 1080]},
                          {"text": "Get involved", "box": [1550 / 1920, 205 / 1080, 1600 / 1920, 215 / 1080]}],
                "icons": []}

    assert agent.tree_shift(tree, screen(0), 1920, 1080) == {"matched": 3, "shifted": 0, "shift": 0}
    assert agent.tree_shift(tree, screen(27), 1920, 1080) == {"matched": 3, "shifted": 2, "shift": 27}


def test_a_guide_reaches_jev_with_the_text_it_says_to_type(monkeypatch):
    task = "Save the image as res.png on the Desktop."
    steps = ['Click the "File" menu.', 'Click "Export As...".',
             'Type "res.png" into the "Name" field.', 'Enter "~/Desktop" as the folder.', 'Click "Export".']
    monkeypatch.setattr(agent, "GUIDED", True)
    monkeypatch.setattr(agent, "GUIDES", {"Save the image as   res.png on the Desktop.": steps})

    guide = agent.load_guide(task)
    assert guide == steps
    assert agent.guide_texts(guide) == ["res.png", "~/Desktop"]
    request = agent.build_request(task, [], ["res.png"], [], [], guide)
    assert request["state"]["guide"] == steps
    assert "guide" in request["questions"]["action"]["instructions"]
    assert "guide" not in agent.build_request(task, [], [], [], [])["state"]
    with pytest.raises(ValueError):
        agent.load_guide("Some other task.")


def test_a_guide_adds_the_shortcuts_it_says_to_press():
    guide = ["Press Ctrl+S.", "Press Ctrl+Shift+S.", "Press Escape.", "Press Ctrl+Shift+T to reopen it.",
             "Press the button."]
    keys = agent.guide_keys(guide)
    assert keys == {"guide_ctrl_shift_s": (["ctrl", "shift", "s"], "Ctrl+Shift+S, as the guide says."),
                    "guide_ctrl_shift_t": (["ctrl", "shift", "t"], "Ctrl+Shift+T, as the guide says.")}
    all_keys = {**agent.KEYS, **keys}
    request = agent.build_request("Reopen the tab.", [], [], [], [], guide, all_keys)
    assert "guide_ctrl_shift_t" in request["questions"]["key"]["criteria"]
    actions, line = agent.step_actions("key", {"key": {"choice": "guide_ctrl_shift_t"}}, [], [], 100, 100, all_keys)
    assert actions == [{"keyboard": {"keys": ["ctrl", "shift", "t"]}}]
