import re

acronyms = {"MSE", "SDTW"}


def camel_case_split(identifier):
    matches = re.finditer(
        '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def format_name(name: str) -> str:
    name = name.replace("_", " ")
    name = " ".join(camel_case_split(name))
    name = name.title()

    for acro in acronyms:
        expr = re.compile(re.escape(acro), re.IGNORECASE)
        name = expr.sub(acro, name)

    return name
