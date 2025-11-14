from typing import Any

from .upgrade_none_to_1 import upgrade_none_to_1


def upgrade(state: dict[str, Any]) -> None:
    if "_compatibility_version" not in state:
        upgrade_none_to_1(state)
