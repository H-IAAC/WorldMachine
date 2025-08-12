from contextlib import ContextDecorator

from torch.autograd.profiler import record_function

from .nvtx import annotate_se


class profile_range(ContextDecorator):
    def __init__(self, message=None, color=None, domain=None, category=None):
        self.message = message
        self.color = color
        self.domain = domain
        self.category = category

        torch_message = message
        if category is not None:
            torch_message = f"{category}:{torch_message}"
        if domain is not None:
            torch_message = f"{torch_message}@{domain}"

        self._nvtx_range = annotate_se(self.message,
                                       self.color,
                                       self.domain,
                                       self.category)

        self._torch_range = record_function(torch_message)

    def __enter__(self):
        self._nvtx_range.__enter__()
        self._torch_range.__enter__()

    def __exit__(self, exception_type, exception, traceback):
        self._nvtx_range.__exit__(exception_type, exception, traceback)
        self._torch_range.__exit__(exception_type, exception, traceback)
