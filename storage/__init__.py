"""Concrete storage implementations satisfying the domain interfaces."""

from storage.filesystem_object_store import FilesystemObjectStore
from storage.sqlite_data_store import SQLiteDataStore

__all__ = ["FilesystemObjectStore", "SQLiteDataStore"]
