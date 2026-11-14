from __future__ import annotations


def run(host: str, port: int) -> int:
    try:
        import uvicorn

        from par.service.app import create_app
    except ImportError as exc:
        print(f"par serve needs the service extra: pip install 'par[service]' ({exc})")
        return 1

    # workers=1 is required, not a default: task and robot-connection state
    # live in this process's memory (see par.service.tasks.registry).
    uvicorn.run(create_app(), host=host, port=port, workers=1)
    return 0
