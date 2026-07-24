"""Path handling that does not assume an operating-system filesystem.

Every path a :class:`~brukerapi.dataset.Dataset` or :class:`~brukerapi.folders.Folder`
is built from passes through this module, so any object implementing the pathlib
read protocol can stand in for a real path. In particular a
:class:`zipfile.Path` works, which lets a dataset be read straight out of a
``.zip``/``.PvDatasets`` archive without extracting it first.

The protocol used here is deliberately small: ``iterdir``, ``open``, ``exists``,
``is_dir``, ``is_file``, ``parent``, ``name``, ``stem``, ``suffix`` and ``/``.
Everything else an archive member cannot do -- ``stat``, ``with_suffix``,
``os.listdir``, ``np.memmap`` -- has a helper here that falls back to something
it can.
"""

import os
from pathlib import Path

import numpy as np


def as_path(path):
    """Return `path` as a :class:`pathlib.Path`, passing path-likes through.

    ``str`` and :class:`os.PathLike` are coerced as before; anything else is
    assumed to implement the read protocol above and is returned unchanged
    (``Path(zipfile.Path(...))`` raises).
    """
    if isinstance(path, (str, os.PathLike)):
        return Path(path)
    return path


def traverse(base, relative):
    """Resolve a ``'../../method'``-style relative path against `base`.

    :class:`zipfile.Path` joins textually and never collapses ``..``, so the
    walk is done one segment at a time through ``parent`` rather than by
    joining a string.
    """
    for part in str(relative).replace(os.sep, "/").split("/"):
        if part in ("", "."):
            continue
        base = base.parent if part == ".." else base / part
    return base


def with_suffix(path, suffix):
    """`path` with its final suffix replaced -- ``Path.with_suffix`` for any path-like."""
    return path.parent / f"{path.stem}{suffix}"


def listdir(path):
    """Names of the entries in the directory `path` -- ``os.listdir`` for any path-like."""
    return [entry.name for entry in path.iterdir()]


def file_size(path):
    """Size in bytes of the file `path` points at."""
    stat = getattr(path, "stat", None)
    if stat is not None:
        return stat().st_size
    with path.open("rb") as binary:
        binary.seek(0, os.SEEK_END)
        return binary.tell()


def read_array(path, dtype, shape, order="F"):
    """Read the binary file `path` into an array of `shape`.

    A filesystem path is memory-mapped as before. An archive member cannot be
    (a memory map cannot address a compressed member), so it degrades to a full
    read.
    """
    if isinstance(path, (str, os.PathLike)):
        return np.array(np.memmap(path, dtype=dtype, shape=shape, order=order)[:])
    with path.open("rb") as binary:
        buffer = binary.read()
    return np.frombuffer(buffer, dtype=dtype).reshape(shape, order=order)
