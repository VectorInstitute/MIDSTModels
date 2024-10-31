import enum
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Union, cast

import tomli
import tomli_w

RawConfig = Dict[str, Any]


class TaskType(enum.Enum):
    BINCLASS = "binclass"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"

    def __str__(self) -> str:
        return self.value


def raise_unknown(unknown_what: str, unknown_value: Any):
    raise ValueError(f"Unknown {unknown_what}: {unknown_value}")


def _replace(data, condition, value):
    def do(x):
        if isinstance(x, dict):
            return {k: do(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [do(y) for y in x]
        else:
            return value if condition(x) else x

    return do(data)


_CONFIG_NONE = "__none__"


def unpack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x == _CONFIG_NONE, None))
    return config


def pack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x is None, _CONFIG_NONE))
    return config


def load_config(path: Union[Path, str]) -> Any:
    with open(path, "rb") as f:
        return unpack_config(tomli.load(f))


def dump_config(config: Any, path: Union[Path, str]) -> None:
    with open(path, "wb") as f:
        tomli_w.dump(pack_config(config), f)
    # check that there are no bugs in all these "pack/unpack" things
    assert config == load_config(path)


def load_json(path: Union[Path, str], **kwargs) -> Any:
    return json.loads(Path(path).read_text(), **kwargs)


def dump_json(x: Any, path: Union[Path, str], **kwargs) -> None:
    kwargs.setdefault("indent", 4)
    Path(path).write_text(json.dumps(x, **kwargs) + "\n")


def load_pickle(path: Union[Path, str], **kwargs) -> Any:
    return pickle.loads(Path(path).read_bytes(), **kwargs)


def dump_pickle(x: Any, path: Union[Path, str], **kwargs) -> None:
    Path(path).write_bytes(pickle.dumps(x, **kwargs))


def get_categories(X_train_cat):
    return (
        None
        if X_train_cat is None
        else [len(set(X_train_cat[:, i])) for i in range(X_train_cat.shape[1])]
    )


def visualize_default(data_df):
    # Define the mappings
    sex_mapping = {1: "male", 2: "female"}
    education_mapping = {
        0: "unknown_0",
        1: "graduate school",
        2: "university",
        3: "high school",
        4: "others",
        5: "unknown_1",
        6: "unknown_2",
    }
    marriage_mapping = {0: "unknown", 1: "married", 2: "single", 3: "others"}
    pay_mapping = {
        -2: "unknown",
        -1: "pay duly",
        0: "payment delay for one month",
        1: "payment delay for two months",
        2: "payment delay for three months",
        3: "payment delay for four months",
        4: "payment delay for five months",
        5: "payment delay for six months",
        6: "payment delay for seven months",
        7: "payment delay for eight months",
        8: "payment delay for nine months and above",
    }

    # Apply the mappings
    data_df["SEX"] = data_df["SEX"].map(sex_mapping)
    data_df["EDUCATION"] = data_df["EDUCATION"].map(education_mapping)
    data_df["MARRIAGE"] = data_df["MARRIAGE"].map(marriage_mapping)
    data_df["PAY_0"] = data_df["PAY_0"].map(pay_mapping)
    data_df["PAY_2"] = data_df["PAY_2"].map(pay_mapping)
    data_df["PAY_3"] = data_df["PAY_3"].map(pay_mapping)
    data_df["PAY_4"] = data_df["PAY_4"].map(pay_mapping)
    data_df["PAY_5"] = data_df["PAY_5"].map(pay_mapping)
    data_df["PAY_6"] = data_df["PAY_6"].map(pay_mapping)

    return data_df
