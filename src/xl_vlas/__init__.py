from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("xl-vlas")
except PackageNotFoundError:
    __version__ = "unknown"
