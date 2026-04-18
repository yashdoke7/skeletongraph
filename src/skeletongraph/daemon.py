"""
Background watchdog daemon.

Watches the project root for file modifications and automatically
triggers incremental updates to the SkeletonGraph index. Keeps the
IDE's context ultra-fresh without manual rebuild steps.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from threading import Timer

from ..build import update_index, discover_files

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

logger = logging.getLogger(__name__)


class SkeletonGraphEventHandler(FileSystemEventHandler):
    """Debounced event handler targeting source file changes."""
    def __init__(self, project_root: Path, debounce_seconds: float = 1.0):
        self.project_root = project_root
        self.debounce_seconds = debounce_seconds
        self._timer: Timer | None = None
        
        # Load the known list of valid extensions based on the build module
        # If it changes, update here.
        self.valid_extensions = {".py", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx",
                                ".java", ".go", ".rs", ".cpp", ".cs", ".rb", ".php"}
                                
        self._pending_files = set()

    def _is_valid_file(self, path: Path) -> bool:
        if path.suffix.lower() not in self.valid_extensions:
            return False
            
        rel_parts = path.relative_to(self.project_root).parts
        
        # Skip hidden dirs or common ignore folders
        if any(p.startswith(".") for p in rel_parts):
            return False
        if any(p in {"node_modules", "dist", "build", "venv", "__pycache__"} for p in rel_parts):
            return False
            
        return True

    def on_any_event(self, event):
        if event.is_directory:
            return

        if event.event_type in ("modified", "created", "deleted"):
            path = Path(event.src_path)
            if self._is_valid_file(path):
                self._pending_files.add(path)
                self._schedule_update()

    def _schedule_update(self):
        if self._timer is not None:
            self._timer.cancel()
        self._timer = Timer(self.debounce_seconds, self._run_update)
        self._timer.daemon = True
        self._timer.start()

    def _run_update(self):
        if not self._pending_files:
            return
            
        logger.info(f"Triggering incremental update for {len(self._pending_files)} changed files...")
        self._pending_files.clear()
        
        start = time.perf_counter()
        try:
            store = update_index(self.project_root)
            elapsed = time.perf_counter() - start
            logger.info(f"Updated index in {elapsed:.3f}s (Functions: {store.meta.total_functions})")
        except Exception as e:
            logger.error(f"Error updating index: {e}")


def start_daemon(project_root: Path, debounce_seconds: float = 1.5) -> None:
    """Start the watchdog observer."""
    if not HAS_WATCHDOG:
        print("Watchdog is not installed. Run: pip install watchdog")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [SkeletonGraph] %(message)s",
        datefmt="%H:%M:%S"
    )

    event_handler = SkeletonGraphEventHandler(project_root, debounce_seconds)
    observer = Observer()
    observer.schedule(event_handler, str(project_root), recursive=True)
    
    logger.info(f"Watching {project_root} for source changes...")
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Daemon stopped.")
        
    observer.join()
