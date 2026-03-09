from . import health as _health
from . import fall as _fall

__all__ = []
for _module in (_health, _fall):
    _module_names = getattr(_module, "__all__", None)
    if _module_names is None:
        _module_names = [
            n
            for n, obj in vars(_module).items()
            if (not n.startswith("_")) and (getattr(obj, "__module__", None) == _module.__name__)
        ]
    for _name in _module_names:
        globals()[_name] = getattr(_module, _name)
    __all__.extend(_module_names)
