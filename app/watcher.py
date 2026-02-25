from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from app.scanner import SUPPORTED_EXTS


class FolderWatcher:
    def __init__(self, root: Path, debounce_seconds: int, on_dirty):
        self.root = root
        self.debounce_seconds = debounce_seconds
        self.on_dirty = on_dirty
        self.last_change_time: str | None = None
        self.dirty_flag = False
        self._observer = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> bool:
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except Exception:
            return False

        watcher = self

        class Handler(FileSystemEventHandler):
            def _is_supported(self, raw_path: str | None) -> bool:
                if not raw_path:
                    return False
                path = Path(raw_path)
                return path.suffix.lower() in SUPPORTED_EXTS

            def on_created(self, event):
                if event.is_directory:
                    return
                if not self._is_supported(getattr(event, "src_path", None)):
                    return
                watcher.mark_dirty()

            def on_modified(self, event):
                if event.is_directory:
                    return
                if not self._is_supported(getattr(event, "src_path", None)):
                    return
                watcher.mark_dirty()

            def on_deleted(self, event):
                if event.is_directory:
                    return
                if not self._is_supported(getattr(event, "src_path", None)):
                    return
                watcher.mark_dirty()

            def on_moved(self, event):
                if event.is_directory:
                    return
                src_ok = self._is_supported(getattr(event, "src_path", None))
                dst_ok = self._is_supported(getattr(event, "dest_path", None))
                if not src_ok and not dst_ok:
                    return
                watcher.mark_dirty()

        self._observer = Observer()
        self._observer.schedule(Handler(), str(self.root), recursive=True)
        self._observer.start()
        self._thread = threading.Thread(target=self._debounce_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=2)

    def mark_dirty(self) -> None:
        self.dirty_flag = True
        self.last_change_time = datetime.now(timezone.utc).isoformat()

    def _debounce_loop(self) -> None:
        while not self._stop.is_set():
            if self.dirty_flag and self.last_change_time:
                then = datetime.fromisoformat(self.last_change_time)
                if (
                    datetime.now(timezone.utc) - then
                ).total_seconds() >= self.debounce_seconds:
                    self.dirty_flag = False
                    self.on_dirty()
            time.sleep(1)
