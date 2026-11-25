# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Recognise a datarecord store by its layout and pick one of its revisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import UUID

if TYPE_CHECKING:
    import os

    from datarecord.store.store import Revision, Store

Kind = Literal["layer", "record", "store", "csv", "file"]


def _layer_id(path: Path) -> UUID | None:
    """Return the id naming the layer folder `<store>/layers/<id>`, else `None`."""
    if path.parent.name != "layers":
        return None
    try:
        return UUID(path.name)
    except ValueError:
        return None


def classify(path: str | os.PathLike) -> Kind:
    """Tell what `path` holds, a layer folder, record, store, CSV folder or file.

    The layer check comes first, as a root layer also holds a `manifest.json`.
    """
    path = Path(path)
    if not path.is_dir():
        return "file"
    if _layer_id(path) is not None:
        if not (path / "manifest.json").is_file():
            msg = f"layer '{path}' is not committed (a pending layer has no manifest)"
            raise ValueError(msg)
        return "layer"
    if (path / "manifest.json").is_file():
        return "record"
    if (path / "layers").is_dir():
        return "store"
    return "csv"


def _store_at(path: Path) -> Store:
    from datarecord import Workspace  # noqa: PLC0415

    return Workspace.process().store(str(path))


def _describe(revisions: list[Revision]) -> str:
    return "\n".join(
        f"  {r.id}  {json.dumps(r.meta, sort_keys=True)}"
        for r in sorted(revisions, key=lambda r: str(r.id))
    )


def open_revision(path: str | os.PathLike, revision: str | None = None) -> Revision:
    """Return the revision a store or layer folder at `path` addresses.

    Parameters
    ----------
    path
        A store, or one of its layer folders.
    revision
        Full id or unique prefix of a revision of a store. Only for a store.
        Without it a store opens its single leaf.

    Raises
    ------
    ValueError
        If `path` is no store or layer folder, `revision` is given for
        anything but a store, the store holds no revision, has several
        leaves with no `revision`, or `revision` is unknown or ambiguous.

    """
    path = Path(path)
    kind = classify(path)
    if kind == "layer":
        if revision is not None:
            msg = f"revision= is for a store, '{path}' is a layer folder"
            raise ValueError(msg)
        return _store_at(path.parent.parent).revision(UUID(path.name))
    if kind != "store":
        msg = f"'{path}' is not a datarecord store or layer folder"
        raise ValueError(msg)

    revisions = _store_at(path).revisions()
    if not revisions:
        msg = f"store '{path}' holds no revision"
        raise ValueError(msg)

    if revision is None:
        parents = {r.parent for r in revisions if r.parent is not None}
        leaves = [r for r in revisions if r.id not in parents]
        if len(leaves) == 1:
            return leaves[0]
        msg = (
            f"store '{path}' has {len(leaves)} leaf revisions, pick one with "
            f"revision=:\n{_describe(leaves)}"
        )
        raise ValueError(msg)

    prefix = revision.lower()
    matches = [
        r
        for r in revisions
        if str(r.id).startswith(prefix) or r.id.hex.startswith(prefix)
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        msg = f"no revision of '{path}' starts with '{revision}', it holds:\n{_describe(revisions)}"
    else:
        msg = f"revision '{revision}' is ambiguous in '{path}', it matches:\n{_describe(matches)}"
    raise ValueError(msg)
