from contextlib import ContextDecorator

from torch.autograd.profiler import record_function

from .nvtx import annotate_se


class profile_range(ContextDecorator):
    def __init__(self, message=None, color=None, domain=None, category=None):
        self.message = message
        self.color = color
        self.domain = domain
        self.category = category

        self._nvtx_range = annotate_se(self.message,
                                       self.color,
                                       self.domain,
                                       self.category)

        self._torch_range = record_function(self.message,
                                            f"{self.category}@{self.domain}({self.color})")

    def __enter__(self):
        self._nvtx_range.__enter__()
        self._torch_range.__enter__()

    def __exit__(self, exception_type, exception, traceback):
        self._nvtx_range.__exit__(exception_type, exception, traceback)
        self._torch_range.__exit__(exception_type, exception, traceback)
