from typing import TypeVar, Sequence

T = TypeVar("T")

def split_data(x_all:Sequence[T]) -> dict[str, Sequence[T]]:
    size_all = len(x_all)

    cut1 = int(0.6*size_all)
    cut2 = int(0.8*size_all)

    x_train = x_all[0:cut1]

    x_val = x_all[cut1:cut2]

    x_test = x_all[cut2:]

    data = {"train":x_train, 
            "val":x_val,
            "test":x_test}

    return data


def split_data_dict(x_all:dict[str, Sequence[T]]) -> dict[str, dict[str, Sequence[T]]]:
    data_split = {}
    for name in ["train", "val", "test"]:
        data_split[name] = {}
        for dimension in x_all:
            data_split[name][dimension] = None
    
    for dimension in x_all:
        splitted = split_data(x_all[dimension])

        data_split["train"][dimension] = splitted["train"]
        data_split["val"][dimension] = splitted["val"]
        data_split["test"][dimension] = splitted["test"]

    return data_split