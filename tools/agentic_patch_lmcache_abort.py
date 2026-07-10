#!/usr/bin/env python3
"""Patch LMCache abort cleanup so scheduler-side aborts are non-fatal."""

from __future__ import annotations

import importlib.util
from pathlib import Path


OLD = '''        # Cleanup if request was aborted
        if request.status == RequestStatus.FINISHED_ABORTED:
            # Notify storage backends of aborted requests
            assert self.lmcache_engine is not None
            sm = self.lmcache_engine.storage_manager
            if sm is not None:
                sm.cancel_request(request.request_id)

            if self.async_loading:
'''

NEW = '''        # Cleanup if request was aborted
        if request.status == RequestStatus.FINISHED_ABORTED:
            # Notify storage backends of aborted requests when this connector
            # owns an LMCache engine. Scheduler-side connectors may not own one,
            # and abort cleanup should not make EngineCore fatal.
            lmcache_engine = self.lmcache_engine
            if lmcache_engine is not None:
                sm = lmcache_engine.storage_manager
                if sm is not None:
                    sm.cancel_request(request.request_id)
            else:
                logger.debug(
                    "Skipping LMCache abort cleanup for request %s because "
                    "lmcache_engine is not initialized on this connector.",
                    request.request_id,
                )

            if self.async_loading:
'''


def main() -> None:
    spec = importlib.util.find_spec("lmcache.integration.vllm.vllm_v1_adapter")
    if spec is None or spec.origin is None:
        print("LMCache vLLM adapter not installed; skipping abort cleanup patch.")
        return

    path = Path(spec.origin)
    text = path.read_text()
    if NEW in text:
        print(f"LMCache abort cleanup patch already applied: {path}")
        return
    if OLD not in text:
        raise RuntimeError(
            "Could not find expected LMCache abort cleanup block in "
            f"{path}; refusing to apply a fuzzy patch."
        )

    path.write_text(text.replace(OLD, NEW, 1))
    print(f"Applied LMCache abort cleanup patch: {path}")


if __name__ == "__main__":
    main()
