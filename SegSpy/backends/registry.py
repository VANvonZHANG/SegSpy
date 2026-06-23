"""Registry for segmentation backends."""
from SegSpy.backends.base import SegmentationBackend


class SegmentationRegistry:
    """Class-level registry for managing available segmentation backends.

    Backends are registered by their ``name`` class attribute and can be
    retrieved by name or auto-discovered via :meth:`find_compatible`.
    """

    _backends: dict[str, type[SegmentationBackend]] = {}

    @classmethod
    def register(cls, backend_class: type[SegmentationBackend]):
        """Register a segmentation backend class.

        Raises:
            TypeError: if ``backend_class`` is not a ``SegmentationBackend`` subclass.
            ValueError: if its ``name`` is the default ``'abstract'`` (i.e. unset),
                or if a *different* class is already registered under the same name.
        """
        if not issubclass(backend_class, SegmentationBackend):
            raise TypeError(f"{backend_class} is not a SegmentationBackend subclass")
        if backend_class.name == "abstract":
            raise ValueError(
                f"{backend_class.__name__} must define a 'name' class attribute"
            )
        if (
            backend_class.name in cls._backends
            and cls._backends[backend_class.name] is not backend_class
        ):
            raise ValueError(f"Backend '{backend_class.name}' is already registered")
        cls._backends[backend_class.name] = backend_class

    @classmethod
    def get(cls, name: str) -> SegmentationBackend:
        """Instantiate and return a registered backend by name.

        Raises:
            ValueError: if no backend with the given name is registered.
        """
        if name not in cls._backends:
            available = ", ".join(cls.list_backends())
            raise ValueError(
                f"Unknown segmentation backend: '{name}'. Available: [{available}]"
            )
        return cls._backends[name]()

    @classmethod
    def find_compatible(cls, microscope_type: str, particle_type: str) -> str:
        """Return the name of the first backend whose ``supports()`` is True."""
        for name, backend_cls in cls._backends.items():
            if backend_cls.supports(microscope_type, particle_type):
                return name
        raise ValueError(
            f"No segmentation backend supports {microscope_type}/{particle_type}"
        )

    @classmethod
    def list_backends(cls) -> list[str]:
        """Return the names of all registered backends (in registration order)."""
        return list(cls._backends.keys())

    @classmethod
    def clear(cls):
        """Remove all registered backends. Useful for test isolation."""
        cls._backends.clear()
