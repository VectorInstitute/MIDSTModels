import hashlib
import os
from collections import Counter
from copy import deepcopy
from dataclasses import astuple, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from category_encoders import LeaveOneOutEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from .metrics import calculate_metrics as calculate_metrics_
from .util import TaskType, dump_pickle, get_categories, load_json, load_pickle

ArrayDict = Dict[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]


CAT_MISSING_VALUE = "nan"
CAT_RARE_VALUE = "__rare__"
Normalization = Literal["standard", "quantile", "minmax"]
NumNanPolicy = Literal["drop-rows", "mean"]
CatNanPolicy = Literal["most_frequent"]
CatEncoding = Literal["one-hot", "counter"]
YPolicy = Literal["default"]


def get_category_sizes(X: Union[torch.Tensor, np.ndarray]) -> List[int]:
    XT = X.T.cpu().tolist() if isinstance(X, torch.Tensor) else X.T.tolist()
    return [len(set(x)) for x in XT]


@dataclass(frozen=False)
class Dataset:
    X_num: Optional[ArrayDict]
    X_cat: Optional[ArrayDict]
    y: ArrayDict
    y_info: Dict[str, Any]
    task_type: TaskType
    n_classes: Optional[int]

    @classmethod
    def from_dir(cls, dir_: Union[Path, str]) -> "Dataset":
        dir_ = Path(dir_)
        splits = [k for k in ["train", "test"] if dir_.joinpath(f"y_{k}.npy").exists()]

        def load(item) -> ArrayDict:
            return {
                x: cast(
                    np.ndarray, np.load(dir_ / f"{item}_{x}.npy", allow_pickle=True)
                )  # type: ignore[code]
                for x in splits
            }

        if Path(dir_ / "info.json").exists():
            info = load_json(dir_ / "info.json")
        else:
            info = None
        return Dataset(
            load("X_num") if dir_.joinpath("X_num_train.npy").exists() else None,
            load("X_cat") if dir_.joinpath("X_cat_train.npy").exists() else None,
            load("y"),
            {},
            TaskType(info["task_type"]),
            info.get("n_classes"),
        )

    @property
    def is_binclass(self) -> bool:
        return self.task_type == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == TaskType.MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    @property
    def n_num_features(self) -> int:
        return 0 if self.X_num is None else self.X_num["train"].shape[1]

    @property
    def n_cat_features(self) -> int:
        return 0 if self.X_cat is None else self.X_cat["train"].shape[1]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def size(self, part: Optional[str]) -> int:
        return sum(map(len, self.y.values())) if part is None else len(self.y[part])

    @property
    def nn_output_dim(self) -> int:
        if self.is_multiclass:
            assert self.n_classes is not None
            return self.n_classes
        else:
            return 1

    def get_category_sizes(self, part: str) -> List[int]:
        return [] if self.X_cat is None else get_category_sizes(self.X_cat[part])

    def calculate_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        prediction_type: Optional[str],
    ) -> Dict[str, Any]:
        metrics = {
            x: calculate_metrics_(
                self.y[x], predictions[x], self.task_type, prediction_type, self.y_info
            )
            for x in predictions
        }
        if self.task_type == TaskType.REGRESSION:
            score_key = "rmse"
            score_sign = -1
        else:
            score_key = "accuracy"
            score_sign = 1
        for part_metrics in metrics.values():
            part_metrics["score"] = score_sign * part_metrics[score_key]
        return metrics


class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat):
        self.X_num = X_num
        self.X_cat = X_cat

    def __getitem__(self, index):
        this_num = self.X_num[index]
        this_cat = self.X_cat[index]

        sample = (this_num, this_cat)

        return sample

    def __len__(self):
        return self.X_num.shape[0]


def preprocess(
    dataset_path,
    ref_dataset_path,
    transforms,
    task_type="binclass",
    inverse=False,
    concat=True,
):
    T = Transformations(**transforms)
    ref_dataset = make_dataset(
        data_path=ref_dataset_path,
        T=T,
        task_type=task_type,
        change_val=False,
        concat=concat,
    )

    dataset = make_dataset(
        data_path=dataset_path,
        T=T,
        task_type=task_type,
        change_val=False,
        concat=concat,
    )

    if transforms["cat_encoding"] is None:
        X_num = dataset.X_num
        X_cat = dataset.X_cat

        X_train_num, X_test_num = X_num["train"], X_num["test"]
        X_train_cat, X_test_cat = X_cat["train"], X_cat["test"]

        ref_X_train_cat = ref_dataset.X_cat["train"]
        categories = get_categories(ref_X_train_cat)
        d_numerical = X_train_num.shape[1]

        X_num = (X_train_num, X_test_num)
        X_cat = (X_train_cat, X_test_cat)

        if inverse:
            num_inverse = dataset.num_transform.inverse_transform
            # cat_inverse = None
            cat_inverse = ref_dataset.cat_transform.inverse_transform
            return X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse
        else:
            return X_num, X_cat, categories, d_numerical
    else:
        return dataset


def change_val(dataset: Dataset, val_size: float = 0.2):
    # should be done before transformations

    y = np.concatenate([dataset.y["train"], dataset.y["val"]], axis=0)

    ixs = np.arange(y.shape[0])
    if dataset.is_regression:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=777)
    else:
        train_ixs, val_ixs = train_test_split(
            ixs, test_size=val_size, random_state=777, stratify=y
        )

    dataset.y["train"] = y[train_ixs]
    dataset.y["val"] = y[val_ixs]

    if dataset.X_num is not None:
        X_num = np.concatenate([dataset.X_num["train"], dataset.X_num["val"]], axis=0)
        dataset.X_num["train"] = X_num[train_ixs]
        dataset.X_num["val"] = X_num[val_ixs]

    if dataset.X_cat is not None:
        X_cat = np.concatenate([dataset.X_cat["train"], dataset.X_cat["val"]], axis=0)
        dataset.X_cat["train"] = X_cat[train_ixs]
        dataset.X_cat["val"] = X_cat[val_ixs]

    return dataset


def num_process_nans(dataset: Dataset, policy: Optional[NumNanPolicy]) -> Dataset:
    assert dataset.X_num is not None
    nan_masks = {k: np.isnan(v) for k, v in dataset.X_num.items()}
    if not any(x.any() for x in nan_masks.values()):  # type: ignore[code]
        # assert policy is None
        print("No NaNs in numerical features, skipping")
        return dataset

    assert policy is not None
    if policy == "drop-rows":
        valid_masks = {k: ~v.any(1) for k, v in nan_masks.items()}
        assert valid_masks[
            "test"
        ].all(), "Cannot drop test rows, since this will affect the final metrics."
        new_data = {}
        for data_name in ["X_num", "X_cat", "y"]:
            data_dict = getattr(dataset, data_name)
            if data_dict is not None:
                new_data[data_name] = {
                    k: v[valid_masks[k]] for k, v in data_dict.items()
                }
        dataset = replace(dataset, **new_data)
    elif policy == "mean":
        new_values = np.nanmean(dataset.X_num["train"], axis=0)
        X_num = deepcopy(dataset.X_num)
        for k, v in X_num.items():
            num_nan_indices = np.where(nan_masks[k])
            v[num_nan_indices] = np.take(new_values, num_nan_indices[1])
        dataset = replace(dataset, X_num=X_num)
    else:
        assert raise_unknown("policy", policy)
    return dataset


# Inspired by: https://github.com/yandex-research/rtdl/blob/a4c93a32b334ef55d2a0559a4407c8306ffeeaee/lib/data.py#L20
def normalize(
    X: ArrayDict,
    normalization: Normalization,
    seed: Optional[int],
    return_normalizer: bool = False,
) -> ArrayDict:
    X_train = X["train"]
    if normalization == "standard":
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == "minmax":
        normalizer = sklearn.preprocessing.MinMaxScaler()
    elif normalization == "quantile":
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(X["train"].shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
        # noise = 1e-3
        # if noise > 0:
        #     assert seed is not None
        #     stds = np.std(X_train, axis=0, keepdims=True)
        #     noise_std = noise / np.maximum(stds, noise)  # type: ignore[code]
        #     X_train = X_train + noise_std * np.random.default_rng(seed).standard_normal(
        #         X_train.shape
        #     )
    else:
        raise_unknown("normalization", normalization)

    normalizer.fit(X_train)
    if return_normalizer:
        return {k: normalizer.transform(v) for k, v in X.items()}, normalizer
    return {k: normalizer.transform(v) for k, v in X.items()}


def cat_process_nans(X: ArrayDict, policy: Optional[CatNanPolicy]) -> ArrayDict:
    assert X is not None
    nan_masks = {k: v == CAT_MISSING_VALUE for k, v in X.items()}
    for x in nan_masks.values():
        print(x)
        print(x.any())
    if any(x.any() for x in nan_masks.values()):  # type: ignore[code]
        if policy is None:
            X_new = X
        elif policy == "most_frequent":
            imputer = SimpleImputer(missing_values=CAT_MISSING_VALUE, strategy=policy)  # type: ignore[code]
            imputer.fit(X["train"])
            X_new = {k: cast(np.ndarray, imputer.transform(v)) for k, v in X.items()}
        else:
            raise_unknown("categorical NaN policy", policy)
    else:
        assert policy is None
        X_new = X
    return X_new


def cat_drop_rare(X: ArrayDict, min_frequency: float) -> ArrayDict:
    assert 0.0 < min_frequency < 1.0
    min_count = round(len(X["train"]) * min_frequency)
    X_new = {x: [] for x in X}
    for column_idx in range(X["train"].shape[1]):
        counter = Counter(X["train"][:, column_idx].tolist())
        popular_categories = {k for k, v in counter.items() if v >= min_count}
        for part in X_new:
            X_new[part].append(
                [
                    (x if x in popular_categories else CAT_RARE_VALUE)
                    for x in X[part][:, column_idx].tolist()
                ]
            )
    return {k: np.array(v).T for k, v in X_new.items()}


def cat_encode(
    X: ArrayDict,
    encoding: Optional[CatEncoding] = None,
    y_train: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    return_encoder: bool = False,
) -> Tuple[ArrayDict, bool, Optional[Any]]:  # (X, is_converted_to_numerical)
    if encoding != "counter":
        y_train = None

    # Step 1. Map strings to 0-based ranges

    if encoding is None:
        unknown_value = np.iinfo("int64").max - 3
        oe = sklearn.preprocessing.OrdinalEncoder(
            handle_unknown="use_encoded_value",  # type: ignore[code]
            unknown_value=unknown_value,  # type: ignore[code]
            dtype="int64",  # type: ignore[code]
        ).fit(X["train"])
        encoder = make_pipeline(oe)
        encoder.fit(X["train"])
        X = {k: encoder.transform(v) for k, v in X.items()}
        max_values = X["train"].max(axis=0)
        for part in X.keys():
            if part == "train":
                continue
            for column_idx in range(X[part].shape[1]):
                X[part][X[part][:, column_idx] == unknown_value, column_idx] = (
                    max_values[column_idx] + 1
                )
        if return_encoder:
            return (X, False, encoder)
        return (X, False)

    # Step 2. Encode.

    elif encoding == "one-hot":
        ohe = sklearn.preprocessing.OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
            dtype=np.float32,  # type: ignore[code]
        )
        encoder = make_pipeline(ohe)

        # encoder.steps.append(('ohe', ohe))
        encoder.fit(X["train"])
        X = {k: encoder.transform(v) for k, v in X.items()}

    elif encoding == "counter":
        assert y_train is not None
        assert seed is not None
        loe = LeaveOneOutEncoder(sigma=0.1, random_state=seed, return_df=False)
        encoder.steps.append(("loe", loe))
        encoder.fit(X["train"], y_train)
        X = {k: encoder.transform(v).astype("float32") for k, v in X.items()}  # type: ignore[code]
        if not isinstance(X["train"], pd.DataFrame):
            X = {k: v.values for k, v in X.items()}  # type: ignore[code]
    else:
        raise_unknown("encoding", encoding)

    if return_encoder:
        return X, True, encoder  # type: ignore[code]
    return (X, True)


def build_target(
    y: ArrayDict, policy: Optional[YPolicy], task_type: TaskType
) -> Tuple[ArrayDict, Dict[str, Any]]:
    info: Dict[str, Any] = {"policy": policy}
    if policy is None:
        pass
    elif policy == "default":
        if task_type == TaskType.REGRESSION:
            mean, std = float(y["train"].mean()), float(y["train"].std())
            y = {k: (v - mean) / std for k, v in y.items()}
            info["mean"] = mean
            info["std"] = std
    else:
        raise_unknown("policy", policy)
    return y, info


@dataclass(frozen=True)
class Transformations:
    seed: int = 0
    normalization: Optional[Normalization] = None
    num_nan_policy: Optional[NumNanPolicy] = None
    cat_nan_policy: Optional[CatNanPolicy] = None
    cat_min_frequency: Optional[float] = None
    cat_encoding: Optional[CatEncoding] = None
    y_policy: Optional[YPolicy] = "default"


def transform_dataset(
    dataset: Dataset,
    transformations: Transformations,
    cache_dir: Optional[Path],
    return_transforms: bool = False,
) -> Dataset:
    # WARNING: the order of transformations matters. Moreover, the current
    # implementation is not ideal in that sense.
    if cache_dir is not None:
        transformations_md5 = hashlib.md5(
            str(transformations).encode("utf-8")
        ).hexdigest()
        transformations_str = "__".join(map(str, astuple(transformations)))
        cache_path = (
            cache_dir / f"cache__{transformations_str}__{transformations_md5}.pickle"
        )
        if cache_path.exists():
            cache_transformations, value = load_pickle(cache_path)
            if transformations == cache_transformations:
                print(
                    f"Using cached features: {cache_dir.name + '/' + cache_path.name}"
                )
                return value
            else:
                raise RuntimeError(f"Hash collision for {cache_path}")
    else:
        cache_path = None

    if dataset.X_num is not None:
        dataset = num_process_nans(dataset, transformations.num_nan_policy)

    num_transform = None
    cat_transform = None
    X_num = dataset.X_num

    if X_num is not None and transformations.normalization is not None:
        X_num, num_transform = normalize(
            X_num,
            transformations.normalization,
            transformations.seed,
            return_normalizer=True,
        )
        num_transform = num_transform

    if dataset.X_cat is None:
        assert transformations.cat_nan_policy is None
        assert transformations.cat_min_frequency is None
        # assert transformations.cat_encoding is None
        X_cat = None
    else:
        # X_cat = cat_process_nans(dataset.X_cat, transformations.cat_nan_policy)
        X_cat = dataset.X_cat
        if transformations.cat_min_frequency is not None:
            X_cat = cat_drop_rare(X_cat, transformations.cat_min_frequency)
        X_cat, is_num, cat_transform = cat_encode(
            X_cat,
            transformations.cat_encoding,
            dataset.y["train"],
            transformations.seed,
            return_encoder=True,
        )

        if is_num:
            X_num = (
                X_cat
                if X_num is None
                else {x: np.hstack([X_num[x], X_cat[x]]) for x in X_num}
            )
            X_cat = None

    y, y_info = build_target(dataset.y, transformations.y_policy, dataset.task_type)

    dataset = replace(dataset, X_num=X_num, X_cat=X_cat, y=y, y_info=y_info)
    dataset.num_transform = num_transform
    dataset.cat_transform = cat_transform

    if cache_path is not None:
        dump_pickle((transformations, dataset), cache_path)
    # if return_transforms:
    # return dataset, num_transform, cat_transform
    return dataset


###############
## DataLoader##
###############


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def prepare_fast_dataloader(D: Dataset, split: str, batch_size: int):
    assert D.X_num is not None
    assert D.X_cat is not None
    X = torch.from_numpy(
        np.concatenate([D.X_num[split], D.X_cat[split]], axis=1)
    ).float()
    dataloader = FastTensorDataLoader(
        X, batch_size=batch_size, shuffle=(split == "train")
    )
    while True:
        yield from dataloader


def read_pure_data(path, split="train"):
    y = np.load(os.path.join(path, f"y_{split}.npy"), allow_pickle=True)
    X_num = None
    X_cat = None
    if os.path.exists(os.path.join(path, f"X_num_{split}.npy")):
        X_num = np.load(os.path.join(path, f"X_num_{split}.npy"), allow_pickle=True)
    if os.path.exists(os.path.join(path, f"X_cat_{split}.npy")):
        X_cat = np.load(os.path.join(path, f"X_cat_{split}.npy"), allow_pickle=True)

    return X_num, X_cat, y


def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)


def make_dataset(
    data_path: str,
    T: Transformations,
    task_type,
    change_val: bool,
    concat=True,
):
    # classification
    if task_type == "binclass" or task_type == "multiclass":
        X_cat = (
            {} if os.path.exists(os.path.join(data_path, "X_cat_train.npy")) else None
        )
        X_num = (
            {} if os.path.exists(os.path.join(data_path, "X_num_train.npy")) else None
        )
        y = {} if os.path.exists(os.path.join(data_path, "y_train.npy")) else None

        for split in ["train", "test"]:
            X_num_t, X_cat_t, y_t = read_pure_data(data_path, split)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                if concat:
                    X_cat_t = concat_y_to_X(X_cat_t, y_t)
                X_cat[split] = X_cat_t
            if y is not None:
                y[split] = y_t
    else:
        # regression
        X_cat = (
            {} if os.path.exists(os.path.join(data_path, "X_cat_train.npy")) else None
        )
        X_num = (
            {} if os.path.exists(os.path.join(data_path, "X_num_train.npy")) else None
        )
        y = {} if os.path.exists(os.path.join(data_path, "y_train.npy")) else None

        for split in ["train", "test"]:
            X_num_t, X_cat_t, y_t = read_pure_data(data_path, split)

            if X_num is not None:
                if concat:
                    X_num_t = concat_y_to_X(X_num_t, y_t)
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            if y is not None:
                y[split] = y_t

    info = load_json(os.path.join(data_path, "info.json"))

    D = Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=TaskType(info["task_type"]),
        n_classes=info.get("n_classes"),
    )

    if change_val:
        D = change_val(D)

    # def categorical_to_idx(feature):
    #     unique_categories = np.unique(feature)
    #     idx_mapping = {category: index for index, category in enumerate(unique_categories)}
    #     idx_feature = np.array([idx_mapping[category] for category in feature])
    #     return idx_feature

    # for split in ['train', 'val', 'test']:
    # D.y[split] = categorical_to_idx(D.y[split].squeeze(1))

    return transform_dataset(D, T, None)
