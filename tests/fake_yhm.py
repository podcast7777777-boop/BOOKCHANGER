"""In-memory stand-in for the YHM REST surface that bookchanger.py uses.

Kept separate from conftest.py so the zero-install fallback runner in
tests/test_bookchanger.py can reuse it without importing pytest.

Stores "files" by their ivr2:/... path exactly as the real REST helpers
address them (get_yhm_path), records deletions and writes for assertions,
and never touches the network.
"""
from __future__ import annotations


class FakeYHM:
    """In-memory replacement for GetTextFile/UploadTextFile/FileAction."""

    def __init__(self):
        self.files: dict[str, str] = {}
        self.deleted: list[str] = []
        self.write_log: list[tuple[str, str]] = []

    # -- REST-helper replacements (same signatures as bookchanger's) -------
    async def read_text_file(self, path: str, token: str):
        return self.files.get(path)

    async def write_text_file(self, path: str, contents: str, token: str) -> bool:
        self.files[path] = contents
        self.write_log.append((path, contents))
        return True

    async def delete_files(self, paths, token: str) -> bool:
        for p in list(paths):
            self.deleted.append(p)
            self.files.pop(p, None)
        return True

    # -- Test conveniences --------------------------------------------------
    def put(self, path: str, contents: str) -> None:
        """Pre-seed a file, e.g. the phonebook ini or a temp TTS."""
        self.files[path] = contents

    def wrote(self, suffix: str):
        """Latest write whose path ends with `suffix`, or None."""
        hits = [(p, c) for p, c in self.write_log if p.endswith(suffix)]
        return hits[-1] if hits else None
