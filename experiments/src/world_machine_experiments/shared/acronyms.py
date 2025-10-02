import re

acronyms = {"MSE", "SDTW"}


def format_name(name: str) -> str:
    name = name.replace("_", " ").title()

    for acro in acronyms:
        expr = re.compile(re.escape(acro), re.IGNORECASE)
        name = expr.sub(acro, name)

    return name
