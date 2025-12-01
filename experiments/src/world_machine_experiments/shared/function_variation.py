import inspect
from functools import wraps
from typing import Any

from docstring_parser import compose, parse
from hamilton.function_modifiers import LiteralDependency, UpstreamDependency


class function_variation():
    def __init__(self, param_map, new_name: str | None = None):
        upstream_dependencies: dict[str, str] = {}
        literal_dependencies: dict[str, Any] = {}

        for param_name in param_map:
            target = param_map[param_name]

            if isinstance(target, UpstreamDependency):
                upstream_dependencies[param_name] = target.source
            elif isinstance(target, LiteralDependency):
                literal_dependencies[param_name] = target.value
            else:
                raise ValueError(
                    f"Parameter map must be a value or source. Parameter {param_name} is {type(target)}")

        self.upstream_dependencies = upstream_dependencies
        self.literal_dependencies = literal_dependencies

        self.new_name = new_name

    def _change_parameters(self, func):
        name = func.__name__
        if self.new_name is not None:
            name = self.new_name

        @wraps(func)
        def wrapper(*args, **kwargs):

            for literal_dep_name in self.literal_dependencies:
                if literal_dep_name in kwargs:
                    raise TypeError(
                        f"{name}() got an unexpected keyword argument '{literal_dep_name}'")

            for old_param, new_param in self.upstream_dependencies.items():
                if new_param in kwargs:
                    kwargs[old_param] = kwargs.pop(new_param)

            kwargs.update(self.literal_dependencies)

            try:
                return func(*args, **kwargs)
            except Exception as e:
                e.add_note(f"While calling function variation: {name}")
                raise

        return wrapper

    def _change_name(self, func):
        if self.new_name is not None:
            func.__name__ = self.new_name

            qualname = func.__qualname__.split(".")
            qualname[-1] = self.new_name

            func.__qualname__ = ".".join(qualname)

    def _update_signature(self, func):
        signature = inspect.signature(func)
        params = list(signature.parameters.values())

        new_params = []
        for param in params:
            if param.name in self.upstream_dependencies:
                new_name = self.upstream_dependencies[param.name]
                new_param = param.replace(name=new_name)

                new_params.append(new_param)
            elif param.name not in self.literal_dependencies:
                new_params.append(param)

        new_signature = signature.replace(parameters=new_params)
        func.__signature__ = new_signature

    def _update_docstring(self, func):
        docstring: str | None = func.__doc__

        if docstring is None:
            return

        # Rename parameters
        for old_name in self.upstream_dependencies:
            new_name = self.upstream_dependencies[old_name]

            docstring = docstring.replace(old_name, new_name)

        # Remove literal parameters
        docstring_obj = parse(docstring)
        remove_param = [
            param for param in docstring_obj.params if param.arg_name in self.literal_dependencies]
        docstring_obj.meta = [
            meta for meta in docstring_obj.meta if meta not in remove_param]
        docstring = compose(docstring_obj)

        func.__doc__ = docstring

    def _update_module(self, func):

        # 0 = _update_module|1 = __call__ NOSONAR
        if inspect.getmodule(inspect.stack()[2][0]) is not None:
            func.__module__ = inspect.getmodule(inspect.stack()[2][0]).__name__

    def _update_annotations(self, func):

        func.__annotations__ = func.__annotations__.copy()

        for old_name in self.upstream_dependencies:
            new_name = self.upstream_dependencies[old_name]
            func.__annotations__[new_name] = func.__annotations__[old_name]
            del func.__annotations__[old_name]

        for name in self.literal_dependencies:
            del func.__annotations__[name]

    def __call__(self, func):
        func = self._change_parameters(func)
        self._change_name(func)
        self._update_signature(func)
        self._update_docstring(func)
        self._update_module(func)
        self._update_annotations(func)

        return func
