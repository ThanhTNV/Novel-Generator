# -*- coding: utf-8 -*-
"""
Live end-to-end check for the MCP server. Requires the app to be running
(``python main.py``) and must be run with the 3.12 venv:

    mcp_server/.venv/Scripts/python.exe mcp_server/smoke.py

It spawns server.py over stdio exactly the way an MCP host would, so it
catches what the mocked unit tests cannot: a broken shebang/venv, a server
that logs to stdout and corrupts the JSON-RPC channel, an output schema the
real payload violates, or a state endpoint the app does not serve yet.

Exit 0 on success; non-zero with a one-line reason otherwise.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent
SERVER = HERE / "server.py"
URL = os.environ.get("NOVELGEN_URL", "http://127.0.0.1:8000")
# Generous: the app's first get_story_state can trigger a history-index load.
OVERALL_TIMEOUT_SECONDS = 180


class SmokeFailure(Exception):
    pass


def _payload(result):
    """
    The tool's dict, from structuredContent when the SDK validated it, else
    from the text block. Either way an ``error`` key is a failed call.
    """
    if result.isError:
        text = " ".join(getattr(c, "text", "") for c in result.content)
        raise SmokeFailure("tool call failed at the protocol level: %s" % text)
    if result.structuredContent is not None:
        data = result.structuredContent
    else:
        try:
            data = json.loads(result.content[0].text)
        except (IndexError, AttributeError, ValueError) as exc:
            raise SmokeFailure("tool returned no JSON payload: %s" % exc)
    if isinstance(data, dict) and "error" in data:
        raise SmokeFailure("tool returned an error: %s" % data["error"])
    return data


async def run() -> int:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER)],
        # stdio_client merges this over its minimal default environment; the
        # host will not forward NOVELGEN_URL unless told to, so neither do we
        # rely on inheritance here.
        env={"NOVELGEN_URL": URL, "PYTHONIOENCODING": "utf-8"},
        cwd=str(HERE),
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            print("server:", init.serverInfo.name, init.serverInfo.version)
            if init.serverInfo.name != "novel-generator":
                raise SmokeFailure("unexpected server name %r" % init.serverInfo.name)

            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print("tools:", ", ".join(names))
            if "get_story_state" not in names or "list_novels" not in names:
                raise SmokeFailure("core tools missing from list_tools")

            novels = _payload(await session.call_tool("list_novels", {})).get("novels", [])
            print("novels:", len(novels))
            if not novels:
                print("no novels in the workspace; get_story_state skipped")
                return 0

            slug = novels[0]["slug"]
            state = _payload(await session.call_tool("get_story_state", {"novel": slug}))
            print("state[%s] keys: %s" % (slug, ", ".join(sorted(state))))
            for key in ("novel", "chapters", "latest", "latest_scene_date", "outline",
                        "divergences", "memory", "rules", "skills"):
                if key not in state:
                    raise SmokeFailure("state is missing %r" % key)
            latest = state.get("latest")
            print("latest.filename:", latest["filename"] if latest else None)
            print("latest_scene_date:", state.get("latest_scene_date"))
            return 0


def main() -> int:
    # Vietnamese titles on a cp1252 console would otherwise crash the print,
    # and a crashed smoke script looks like a broken server.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if not SERVER.is_file():
        print("SMOKE FAILED: %s not found" % SERVER)
        return 2
    try:
        code = asyncio.run(asyncio.wait_for(run(), OVERALL_TIMEOUT_SECONDS))
    except SmokeFailure as exc:
        print("SMOKE FAILED: %s" % exc)
        return 1
    except asyncio.TimeoutError:
        print("SMOKE FAILED: no answer within %ds (is the app running at %s?)"
              % (OVERALL_TIMEOUT_SECONDS, URL))
        return 3
    except Exception as exc:  # noqa: BLE001 - report, never traceback
        print("SMOKE FAILED: %s: %s" % (type(exc).__name__, exc))
        return 4
    if code == 0:
        print("SMOKE OK")
    return code


if __name__ == "__main__":
    sys.exit(main())
