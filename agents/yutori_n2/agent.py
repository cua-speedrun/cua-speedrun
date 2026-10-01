"""Yutori's n2 SDK loop over the environment observe/step/done interface.

Contract: python agent.py <env_url> <task_description>
The SDK owns batching, normalized coordinates, reasoning history, WebP
encoding, and default context compaction. This adapter owns only input
primitives. Shell and file tools are disabled.
YUTORI_REASONING_EFFORT unset preserves the API default (medium).
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import AsyncExitStack
import io
import json
import os
import sys
import time
from typing import Any

from cua_speedrun.client import Computer


MODEL = "n2"
TOOL_SET = "computer_use_tools-20260830"
DISABLED_TOOLS = ["bash", "read", "write", "edit"]


def reasoning_effort() -> str | None:
    value = os.environ.get("YUTORI_REASONING_EFFORT", "").strip().lower()
    if not value:
        return None
    if value not in {"none", "low", "medium", "xhigh"}:
        raise ValueError("YUTORI_REASONING_EFFORT must be none, low, medium, or xhigh")
    return value


def modified_actions(action: dict[str, Any], modifier: list[str] | None) -> list[dict[str, Any]]:
    keys = list(modifier or [])
    return ([{"keyboard": {"keys_down": keys}}] if keys else []) + [action] + (
        [{"keyboard": {"keys_up": list(reversed(keys))}}] if keys else []
    )


def wheel_notches(model_action: dict[str, Any]) -> int:
    """Keep the model's wheel units, bypassing the SDK's pixel conversion."""
    direction, amount = model_action["direction"], model_action["amount"]
    if direction not in {"up", "down"}:
        raise ValueError("The environment interface supports vertical scrolling only")
    return int(amount) * (1 if direction == "down" else -1)


def environment_key(key: str) -> str:
    return {"cmd": "super", "printscreen": "print"}.get(key, key)


class SpeedrunComputer:
    def __init__(self, computer: Computer):
        self.computer = computer
        self.dimensions: tuple[int, int] | None = None
        self.left_button_held = False
        self.pending_actions: list[dict[str, Any]] = []
        self.uncertain_keys: list[str] = []
        self.uncertain_left_button = False

    async def _modified_step(self, action: dict[str, Any], modifier: list[str] | None) -> None:
        keys = [environment_key(key) for key in modifier or []]
        await self._step(modified_actions(action, keys))

    async def _step(self, actions: list[dict[str, Any]]) -> None:
        """Queue one SDK primitive for the current model computer_batch."""
        self.pending_actions.extend(actions)

    async def _run_step(self, actions: list[dict[str, Any]]) -> None:
        result = await asyncio.to_thread(self.computer.step, actions)
        if result.get("ok") is False:
            raise RuntimeError(str(result.get("error") or result))
        if (result.get("info") or {}).get("step_outcome") == "unknown":
            raise RuntimeError("Connection lost after input; outcome unknown; actions were not replayed")

    async def _flush(self) -> None:
        """Send one model computer_batch as one measured gateway step."""
        recovery: list[dict[str, Any]] = []
        if self.uncertain_keys:
            recovery.append({"keyboard": {"keys_up": list(reversed(self.uncertain_keys))}})
        if self.uncertain_left_button:
            recovery.append({"mouse": {"buttons": {"left_up": True}}})
        if not self.pending_actions and not recovery:
            return
        actions, self.pending_actions = recovery + self.pending_actions, []
        cleanup_keys = [
            key
            for action in actions
            for key in (action.get("keyboard") or {}).get("keys_down", [])
        ]
        may_hold_left = self.uncertain_left_button or any(
            (action.get("mouse") or {}).get("buttons", {}).get("left_down")
            for action in actions
        )
        try:
            await self._run_step(actions)
        except BaseException:
            # The gateway may have stopped part-way through a batch. Release
            # input state without replaying any model action.
            self.uncertain_keys = list(dict.fromkeys(self.uncertain_keys + cleanup_keys))
            self.uncertain_left_button = may_hold_left
            cleanup: list[dict[str, Any]] = []
            if self.uncertain_keys:
                cleanup.append({"keyboard": {"keys_up": list(reversed(self.uncertain_keys))}})
            if self.uncertain_left_button:
                cleanup.append({"mouse": {"buttons": {"left_up": True}}})
            if cleanup:
                try:
                    await self._run_step(cleanup)
                except BaseException:
                    pass
                else:
                    self.uncertain_keys.clear()
                    self.uncertain_left_button = False
                    self.left_button_held = False
            raise
        else:
            self.uncertain_keys.clear()
            self.uncertain_left_button = False

    async def screenshot(self) -> str:
        from PIL import Image

        # N2 calls release_held_mouse_button at the end of computer_batch,
        # which normally flushes before its screenshot delay. This fallback
        # also handles a standalone screenshot or future non-batch GUI tool.
        await self._flush()
        observation = await asyncio.to_thread(self.computer.observe)
        with Image.open(io.BytesIO(observation["png"])) as image:
            self.dimensions = image.size
        return "data:image/png;base64," + base64.b64encode(observation["png"]).decode("ascii")

    async def get_dimensions(self) -> tuple[int, int]:
        if self.dimensions is None:
            await self.screenshot()
        assert self.dimensions is not None
        return self.dimensions

    async def click(self, x: int, y: int, button: str = "left", modifier: list[str] | None = None) -> None:
        if button not in {"left", "right", "middle"}:
            raise ValueError(f"Unsupported mouse button: {button}")
        await self._modified_step({"mouse": {f"{button}_click": [x, y]}}, modifier)

    async def double_click(self, x: int, y: int, modifier: list[str] | None = None) -> None:
        await self._modified_step({"mouse": {"double_click": [x, y]}}, modifier)

    async def triple_click(self, x: int, y: int, modifier: list[str] | None = None) -> None:
        await self._modified_step({"mouse": {"triple_click": [x, y]}}, modifier)

    async def move(self, x: int, y: int) -> None:
        await self._step([{"mouse": {"move": [x, y]}}])

    async def drag(self, path: list[dict[str, int]]) -> None:
        if len(path) < 2:
            raise ValueError("Drag needs a start and end point")
        await self._step([{"mouse": {"left_click_drag": [
            [path[0]["x"], path[0]["y"]], [path[-1]["x"], path[-1]["y"]],
        ]}}])

    async def scroll(
        self, x: int, y: int, scroll_x: int, scroll_y: int,
        modifier: list[str] | None = None, model_action: dict[str, Any] | None = None,
    ) -> None:
        if model_action is None or scroll_x:
            raise ValueError("Scroll requires the original vertical model action")
        action = {"mouse": {"move": [x, y], "scroll": wheel_notches(model_action)}}
        await self._modified_step(action, modifier)

    async def type(self, text: str) -> None:
        await self._step([{"keyboard": {"text": text}}])

    async def keypress(self, keys: list[str]) -> None:
        await self._step([{"keyboard": {"keys": [environment_key(key) for key in keys]}}])

    async def key_down(self, key: str) -> None:
        await self._step([{"keyboard": {"keys_down": [environment_key(key)]}}])

    async def key_up(self, key: str) -> None:
        await self._step([{"keyboard": {"keys_up": [environment_key(key)]}}])

    async def hold_key(self, key: str, ms: int) -> None:
        await self.key_down(key)
        try:
            await self.wait(ms)
        finally:
            await self.key_up(key)

    async def wait(self, ms: int) -> None:
        await self._step([{"action": "wait", "time": ms / 1000}])

    async def left_mouse_down(self, x: int | None = None, y: int | None = None) -> None:
        self.left_button_held = True
        await self._mouse_button(True, x, y)

    async def left_mouse_up(self, x: int | None = None, y: int | None = None) -> None:
        await self._mouse_button(False, x, y)
        self.left_button_held = False

    async def _mouse_button(self, down: bool, x: int | None, y: int | None) -> None:
        if (x is None) != (y is None):
            raise ValueError("Mouse coordinates need both x and y")
        mouse: dict[str, Any] = {"buttons": {"left_down" if down else "left_up": True}}
        if x is not None:
            mouse["move"] = [x, y]
        await self._step([{"mouse": mouse}])

    async def release_held_mouse_button(self) -> None:
        if self.left_button_held:
            await self.left_mouse_up()
        # The SDK invokes this once after all members of computer_batch and
        # before its post-action settle delay and screenshot.
        try:
            await self._flush()
        except Exception as exc:
            # SDK 0.9.29 has already marked the queued members complete at this
            # hook, while the gateway can report only a batch-level uncertain
            # outcome. Make the uncertainty override those member statuses.
            raise RuntimeError(
                "computer_batch did not complete cleanly; ignore per-member "
                f"completion statuses because outcomes are uncertain: {exc}"
            ) from exc

    async def discard_pending_and_release(self) -> None:
        """Do not execute an incomplete batch while unwinding an exception."""
        self.pending_actions.clear()
        cleanup: list[dict[str, Any]] = []
        if self.uncertain_keys:
            cleanup.append({"keyboard": {"keys_up": list(reversed(self.uncertain_keys))}})
        if self.left_button_held or self.uncertain_left_button:
            cleanup.append({"mouse": {"buttons": {"left_up": True}}})
        if cleanup:
            try:
                await self._run_step(cleanup)
            finally:
                self.uncertain_keys.clear()
                self.uncertain_left_button = False
                self.left_button_held = False


class LoggedCompletions:
    """Record each successful actor, compaction, and cap-summary response."""
    def __init__(self, completions: Any):
        self.completions = completions

    async def create(self, **kwargs: Any) -> Any:
        # Enforce the environment boundary on every call, including compaction.
        kwargs["disable_tools"] = list(DISABLED_TOOLS)
        started = time.monotonic()
        response = await self.completions.create(**kwargs)
        print(json.dumps({
            "type": "model_response", "model_sec": time.monotonic() - started,
            "reasoning_effort": kwargs.get("reasoning_effort", "server_default"),
            "response": response.model_dump(exclude_none=True),
        }, ensure_ascii=False), flush=True)
        return response


async def run(env_url: str, instruction: str) -> None:
    from yutori import AsyncYutoriClient
    from yutori.navigator import N2ComputerAgent, format_stop_and_summarize

    effort = reasoning_effort()
    if not os.environ.get("YUTORI_API_KEY", "").strip():
        raise ValueError("YUTORI_API_KEY is required")
    computer = Computer(env_url, timeout_sec=float(os.environ.get("YUTORI_ENV_TIMEOUT_SEC", "600")))
    adapter = SpeedrunComputer(computer)
    print(json.dumps({
        "type": "configuration", "model": MODEL, "sdk_version": "0.9.29",
        "tool_set": TOOL_SET, "disable_tools": DISABLED_TOOLS,
        "reasoning_effort": effort, "default_reasoning_effort": "medium",
        "compactor": "sdk_default", "interface": "observe/step/done",
    }), flush=True)
    client: Any = None

    async def finalize() -> None:
        # End measurement before closing API transports, and guarantee both
        # environment and client cleanup on model, SDK, or adapter failures.
        try:
            await adapter.discard_pending_and_release()
        finally:
            try:
                await asyncio.to_thread(computer.done)
            finally:
                if client is not None:
                    await client.close()

    async with AsyncExitStack() as stack:
        stack.push_async_callback(finalize)
        client = AsyncYutoriClient(api_key=os.environ["YUTORI_API_KEY"])
        completions = LoggedCompletions(client.chat.completions)
        async with N2ComputerAgent(
            computer=adapter, completions=completions,
            model=MODEL, tool_set=TOOL_SET, reasoning_effort=effort,
            supports_click_modifiers=True, supports_scroll_modifiers=True,
            completion_kwargs={"disable_tools": list(DISABLED_TOOLS)},
            max_steps=int(os.environ.get("YUTORI_MAX_STEPS", "500")),
        ) as agent:
            async for step in agent.run(instruction):
                print(json.dumps({"type": "step", "usage": step.get("usage")}), flush=True)
            if agent.stopped_by == "max_steps":
                # The cookbook takes one final summary call and executes no
                # tools from that response. It uses the exact next request.
                nudge = {"role": "user", "content": [
                    {"type": "text", "text": format_stop_and_summarize(instruction)},
                ]}
                await completions.create(**agent.completion_request([nudge]))
            print(json.dumps({"type": "run_end", "stopped_by": agent.stopped_by}), flush=True)


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1], sys.argv[2]))
