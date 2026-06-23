"""SegSpy — headless, backend-pluggable EM image segmentation library for HyperSpy.

Public API symbols are re-exported here as modules are implemented
(see :mod:`SegSpy.core`, :mod:`SegSpy.io`, ...). During incremental
construction this module only exposes ``__version__``; downstream imports
should target the submodules directly or import from the top level once the
full API is in place.
"""

__version__ = "0.1.0"
