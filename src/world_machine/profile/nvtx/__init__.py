try:
    from nvtx import *
    from .annotate_se import annotate_se
except ModuleNotFoundError:
    from .mock import *

from .colors import *