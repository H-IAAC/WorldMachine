import re

acronyms = {"MSE", "SDTW", "LR"}


def camel_case_split(identifier):
    matches = re.finditer(
        '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def format_name(name: str | list[str]) -> str | list[str]:
    if not isinstance(name, str):
        return [format_name(n) for n in name]

    name = name.replace("_", " ")
    name = " ".join(camel_case_split(name))
    name = name.title()

    for acro in acronyms:
        expr = re.compile(re.escape(acro), re.IGNORECASE)
        name = expr.sub(acro, name)

    return name
