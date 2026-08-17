"""
Common subpackage – Utilities and data structures.
"""

# Lazy for the same reason as the top-level package: importing
# torchdisorder.common.validation or .glass_quality must not require the training
# stack, so that the melt-quench script can run its checks inside a bare MLIP
# conda env. See the note in torchdisorder/__init__.py.
_LAZY_EXPORTS = {
    'TargetRDFData': 'torchdisorder.common.target_rdf',
    'standard_nl': 'torchdisorder.common.neighbors',
    'MODELS_PROJECT_ROOT': 'torchdisorder.common.utils',
}

__all__ = [
    'TargetRDFData',
    'standard_nl',
    'MODELS_PROJECT_ROOT',
]


def __getattr__(name):
    try:
        module_path = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
