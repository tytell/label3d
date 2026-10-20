"""Writing the project's Parquet working files without risking the old ones."""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

COMPRESSION = "zstd"


def write_table(table: pa.Table, path: Path, backup: bool = False, **kwargs) -> None:
    """Write table to path via path.tmp, so a failed write leaves the old file as it was.

    With backup, the previous file is kept as path.bak. kwargs go to
    pyarrow.parquet.write_table.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        pq.write_table(table, tmp, compression=COMPRESSION, **kwargs)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if backup and path.exists():
        os.replace(path, path.with_name(path.name + ".bak"))
    os.replace(tmp, path)
