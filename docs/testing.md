# label3d — Testing

## Framework

Tests use **pytest** (`tests/` directory, run with `uv run pytest`). A session-scoped `QApplication` fixture in `tests/conftest.py` is required before any `QObject`, `Signal`, or `QTimer`. All test modules using `PointsStore` must accept the `qapp` fixture.

## Test modules

| Module | What it covers |
|---|---|
| `tests/test_skeleton.py` | `Skeleton` load/save, node navigation, special-char node names |
| `tests/test_setup_project.py` | File classification, trial matching, execute, project.toml |
| `tests/test_points.py` | `PointsStore` construction, full CRUD API, cache, dirty tracking, signals |
| `tests/test_points_paths.py` | OS-level path robustness: spaces, unicode, separators, missing paths |
| `tests/test_adapters.py` | Adapter registry, `SleapAdapter` load/save, basename matching edge cases |
| `tests/test_project_loading.py` | `Project.load()` integration: skeleton/points wiring, missing files, error messages |

## Key testing conventions

- **`tmp_path` fixture** for all file I/O — tests never write outside their temp directory
- **Path robustness**: spaces in directory names, spaces in file names, unicode characters, Windows-style `\` vs POSIX `/` separators in video filenames
- **Special characters** in bodypart/node/trial names round-tripped through JSON and CSV
- **Error paths**: `FileNotFoundError` expected for missing files; error messages must contain path/filename for diagnostics
- **sleap_io dependency**: `SleapAdapter` tests use `pytest.importorskip("sleap_io")`
- **Cache invalidation**: frame cache state inspected via `_cache_key` to verify mutations invalidate it
- **Signal emission**: `changed` signal callbacks counted via list-append lambda

## Running tests

```
uv run pytest               # all tests
uv run pytest -v            # verbose
uv run pytest tests/test_skeleton.py   # single module
uv run pytest -k "spaces"  # tests whose name contains "spaces"
```

See [docs/synthetic_data.md](synthetic_data.md) for the synthetic data generator.
