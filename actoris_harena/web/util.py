"""Small helpers shared by the console's request handlers."""

from __future__ import annotations

import asyncio
from functools import partial

from aiohttp import web  # type: ignore[import]


# The page and its scripts are edited between runs of this console, and the
# browser is not told how long they are good for -- aiohttp's static handler
# sends a validator (ETag, Last-Modified) but no ``Cache-Control``, which lets a
# browser guess a freshness lifetime and serve a script from cache without
# asking. MEASURED consequence: a reload revalidated the page but not its
# scripts, leaving a NEW page running an OLD script, whose first act was to read
# a form field that no longer existed. Asking for revalidation every time costs
# one conditional request on loopback and answers 304 when nothing changed.
@web.middleware
async def revalidate_assets(request: web.Request, handler):
    response = await handler(request)
    path = request.path
    if path == "/" or path.startswith("/static/"):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


async def in_executor(app: web.Application, fn, *args):
    """Run a blocking call on the app's worker pool, off the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(app["executor"], partial(fn, *args))


def preinit_tqdm_lock() -> None:
    """Create tqdm's class lock up front, to stop concurrent loads racing on it.

    LeRobot's dataset loading reaches HuggingFace ``datasets``, whose
    ``thread_map`` wraps work in tqdm's ``ensure_lock``. That helper deletes
    ``tqdm._lock`` again if it was absent when it started, so two loads running
    together in a thread pool both find it absent, both try to delete it, and the
    second raises ``AttributeError: type object 'tqdm' has no attribute '_lock'``
    -- which surfaced as a 500 on an otherwise fine dataset. Creating the lock
    once means it is never absent, so it is never deleted.
    """
    try:
        from tqdm import tqdm

        tqdm.set_lock(tqdm.get_lock())
    except Exception:
        pass
