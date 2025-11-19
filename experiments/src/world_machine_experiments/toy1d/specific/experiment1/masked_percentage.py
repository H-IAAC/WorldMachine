
import json
import os

import pandas as pd
from hamilton.function_modifiers import datasaver

from ._shared import *


def masked_percentage(variations_df: pd.DataFrame) -> float:
    return variations_df["mask"].sum()/len(variations_df)


@datasaver()
def save_masked_percentage(masked_percentage: float, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, "masked_percentage.json")

    masked_percentage_dict = {"masked_percentage": masked_percentage}

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(masked_percentage_dict, file)

    return {"path": file_path}
