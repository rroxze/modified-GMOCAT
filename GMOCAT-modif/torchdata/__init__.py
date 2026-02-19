"""Local shim package to expose `torchdata.datapipes` while delegating
other submodules to the installed `torchdata` package.

This package ensures imports like `from torchdata.datapipes.iter import
IterDataPipe` work by providing a small `datapipes` subpackage that
re-exports the PyTorch datapipe classes, while still allowing other
installed `torchdata` submodules to be found.
"""
import os
import sys

# Ensure the package __path__ includes the real installed torchdata
# package directory (so other submodules are discoverable).
local_dir = os.path.dirname(__file__)
real_path = None
for p in sys.path:
    candidate = os.path.join(p, 'torchdata')
    # skip our local package
    try:
        if os.path.isdir(candidate) and os.path.abspath(candidate) != os.path.abspath(local_dir):
            real_path = candidate
            break
    except Exception:
        continue

__path__ = [local_dir]
if real_path:
    __path__.append(real_path)

__all__ = []
