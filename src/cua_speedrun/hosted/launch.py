"""Hosted launch through the public benchmark command and its existing UI."""

import argparse
import json
import sys
import time

from cua_speedrun.commands.output import diagnostics, emit
from cua_speedrun.hosted import TERMINAL
from cua_speedrun.hosted.client import HostedEvaluations


def run_hosted_benchmark(args):
    with diagnostics(args.json):
        service = HostedEvaluations.open(args.home)
        queued = service.launch(args)
    if args.json:
        emit(queued)
    else:
        print(f"Hosted evaluation {queued['run_id']}")
    if args.background:
        return 0
    if not args.json:
        from cua_speedrun.commands.dashboard_client import register_dashboard_client_commands, run_status, _follow_local
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            try:
                return _follow_local(service, queued['run_id'], 2)
            except KeyboardInterrupt:
                print(f"Detached; evaluation {queued['run_id']} is still running.")
                return 130
        parser = argparse.ArgumentParser()
        register_dashboard_client_commands(parser.add_subparsers(dest="command"))
        argv = ["status", queued["run_id"]]
        if args.home:
            argv += ["--home", args.home]
        return run_status(parser.parse_args(argv))
    try:
        previous = None
        while True:
            payload = service.status(queued["run_id"])
            serialized = json.dumps(payload, sort_keys=True)
            if serialized != previous:
                emit({"type": "status", **payload})
                previous = serialized
            if payload["stage"] in TERMINAL:
                return 0 if payload["stage"] == "card_ready" else 1
            time.sleep(2)
    except KeyboardInterrupt:
        emit({"type": "detached", "run_id": queued["run_id"]})
        return 130
