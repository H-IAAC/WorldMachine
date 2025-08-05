import nvtx


class annotate_se:
    '''
    Like annotate, but using start/end range instead of push/pop range
    '''

    def __init__(self, message=None, color=None, domain=None, category=None):
        self.message = message
        self.color = color
        self.domain = domain
        self.category = category

    def __enter__(self):
        self.profile_range = nvtx.start_range(
            self.message, self.color, self.domain, self.category)

    def __exit__(self, exc_type, exc_value, traceback):
        nvtx.end_range(self.profile_range)
