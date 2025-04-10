import abc
import json
from typing import Protocol, TypeVar

import numpy as np
import pydantic_core
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema


class AddMul(Protocol):
    def __add__(self, other: "AddMul") -> "AddMul":
        ...

    def __mul__(self, factor: float) -> "AddMul":
        ...


T = TypeVar("T", bound=AddMul)


class ParameterScheduler(abc.ABC):
    def __init__(self, n_epoch: int):
        super().__init__()

        self._n_epoch = n_epoch

    @abc.abstractmethod
    def __call__(self, epoch_index: int) -> T:
        ...


class LinearScheduler(ParameterScheduler):
    def __init__(self, initial_value: T, final_value: T, n_epoch: int):
        super().__init__(n_epoch)

        self._initial_value = initial_value
        self._final_value = final_value

    def __call__(self, epoch_index: int) -> T:
        t = epoch_index/(self._n_epoch-1)

        result = (t-1)*self._initial_value
        result += t*self._final_value

        return result

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler) -> CoreSchema:
        def serialize(value: "LinearScheduler") -> str:
            return json.dumps({"type": value.__class__.__name__, "initial_value": value._initial_value, "final_value": value._final_value, "n_epoch": value._n_epoch})

        def validate(value: str) -> "LinearScheduler":
            return value

        schema = core_schema.union_schema([
            core_schema.is_instance_schema(cls),
        ])

        return pydantic_core.core_schema.no_info_after_validator_function(
            validate,
            schema,
            serialization=pydantic_core.core_schema.plain_serializer_function_ser_schema(
                serialize, when_used="json"
            ),
        )


class UniformScheduler(ParameterScheduler):

    def __init__(self, low_value: T, high_value: T, n_epoch: int):
        super().__init__(n_epoch)

        self._low_value = low_value
        self._high_value = high_value

    def __call__(self, epoch_index: int) -> T:

        # TODO use generator
        result = np.random.uniform()

        result *= (self._high_value-self._low_value)
        result += self._low_value

        return result

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler) -> CoreSchema:
        def serialize(value: "UniformScheduler") -> str:
            return json.dumps({"type": value.__class__.__name__, "low_value": value._low_value, "high_value": value._high_value, "n_epoch": value._n_epoch})

        def validate(value: str) -> "UniformScheduler":
            return value

        schema = core_schema.union_schema([
            core_schema.is_instance_schema(cls),
        ])

        return pydantic_core.core_schema.no_info_after_validator_function(
            validate,
            schema,
            serialization=pydantic_core.core_schema.plain_serializer_function_ser_schema(
                serialize, when_used="json"
            ),
        )


class ConstantScheduler(ParameterScheduler):

    def __init__(self, value: T, n_epoch: int):
        super().__init__(n_epoch)

        self._value = value

    def __call__(self, epoch_index: int) -> T:
        return self._value

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler) -> CoreSchema:
        def serialize(value: "ConstantScheduler") -> str:
            return json.dumps({"type": value.__class__.__name__, "value": value._value, "n_epoch": value._n_epoch})

        def validate(value: str) -> "ConstantScheduler":
            return value

        schema = core_schema.union_schema([
            core_schema.is_instance_schema(cls),
        ])

        return pydantic_core.core_schema.no_info_after_validator_function(
            validate,
            schema,
            serialization=pydantic_core.core_schema.plain_serializer_function_ser_schema(
                serialize, when_used="json"
            ),
        )


class ChoiceScheduler(ParameterScheduler):
    def __init__(self, values: list[T], n_epoch: int):
        super().__init__(n_epoch)

        self._values = values

    def __call__(self, epoch_index: int) -> T:

        # TODO Use generator
        result = np.random.choice(self._values)

        return result

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler) -> CoreSchema:
        def serialize(value: "ChoiceScheduler") -> str:
            return json.dumps({"type": value.__class__.__name__, "values": value._values, "n_epoch": value._n_epoch})

        def validate(value: str) -> "ChoiceScheduler":
            return value

        schema = core_schema.union_schema([
            core_schema.is_instance_schema(cls),
        ])

        return pydantic_core.core_schema.no_info_after_validator_function(
            validate,
            schema,
            serialization=pydantic_core.core_schema.plain_serializer_function_ser_schema(
                serialize, when_used="json"
            ),
        )
