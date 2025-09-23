acronyms = {"MSE", "SDTW"}


def format_name(name: str) -> str:
    name_format = name.replace("_", " ").title()

    for acro in acronyms:
        name_format = name_format.replace(acro.capitalize(), acro)

    return name_format
