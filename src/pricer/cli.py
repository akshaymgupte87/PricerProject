"""Lazy command dispatch keeps help independent of ML startup."""
from __future__ import annotations
import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pricer", description="Product pricing and deal intelligence")
    parser.add_argument("command", choices=["api", "dashboard", "demo", "scan", "mcp", "build-index", "prepare-index", "train", "dpo", "gateway-eval"])
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments[:1])
    remaining = arguments[1:]
    from pricer.config import load_environment
    load_environment()
    if args.command in {"build-index", "prepare-index", "train", "dpo", "gateway-eval"}:
        from importlib import import_module
        name = {
            "build-index": "build_bm25",
            "prepare-index": "prepare_index",
            "train": "train_neural",
            "dpo": "train_dpo",
            "gateway-eval": "gateway_eval",
        }[args.command]
        previous = sys.argv
        try:
            sys.argv = [f"pricer {args.command}", *remaining]
            import_module(f"pricer.jobs.{name}").main()
        finally:
            sys.argv = previous
    elif args.command == "api":
        api_parser = argparse.ArgumentParser(prog="pricer api")
        api_parser.add_argument("--host", default="127.0.0.1")
        api_parser.add_argument("--port", type=int, default=8000)
        options = api_parser.parse_args(remaining)
        import uvicorn
        uvicorn.run("pricer.interfaces.api:app", host=options.host, port=options.port)
    else:
        if remaining:
            parser.error("unrecognized arguments: " + " ".join(remaining))
        if args.command in {"dashboard", "demo"}:
            if args.command == "demo":
                from pricer.interfaces.demo import main
                main()
                return
            from pricer.interfaces.dashboard import App
            App().run()
        elif args.command == "scan":
            from pricer.application.framework import DealAgentFramework
            DealAgentFramework().run()
        elif args.command == "mcp":
            from pricer.interfaces.mcp_server import create_server
            create_server().run(transport="stdio")
