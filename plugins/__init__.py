"""
Plugin system for 标书智能助手.

To add a new plugin:
  1. Create a .py file in the appropriate subdirectory
     (providers/, embeddings/, loaders/, or nodes/)
  2. Implement the corresponding abstract base class from plugins.base
  3. Call registry.register_*(YourClass()) at module level
  4. The file is auto-discovered on next startup — no other changes needed.
"""
import importlib
import pkgutil

from plugins.registry import PluginRegistry, registry  # noqa: F401 — re-exported


def _load_all_plugins() -> None:
    import plugins.providers
    import plugins.embeddings
    import plugins.loaders
    import plugins.nodes

    for pkg in (plugins.providers, plugins.embeddings, plugins.loaders, plugins.nodes):
        for _, module_name, _ in pkgutil.iter_modules(pkg.__path__):
            importlib.import_module(f"{pkg.__name__}.{module_name}")


_load_all_plugins()

__all__ = ["PluginRegistry", "registry"]
