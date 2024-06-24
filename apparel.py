from torch.utils.data import Dataset
import pandas as pd
from typing import List, Tuple
from PIL import Image
import torchvision.transforms as transforms
import os
import time


class ApparelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_path: str, preprocessing: transforms.Compose) -> None:
        assert os.path.isdir(images_path)
        self.df = df
        self.images_path = images_path
        self.preprocessing = preprocessing

    @classmethod
    def load(
        cls,
        csv_path: str,
        images_path: str,
        preprocessing: transforms.Compose,
        sample: float = 1,
        labels: List[int] = None
    ) -> "ApparelDataset":
        assert os.path.isfile(csv_path)
        assert sample >= 0 and sample <= 1
        df = pd.read_csv(csv_path)

        if sample < 1:
            sample_size = int(len(df) * sample)
            df = df.sample(n=sample_size).reset_index(drop=True)

        if labels is not None:
            df = df[df['label'].isin(labels)]

        return cls(df, images_path, preprocessing)

    def __len__(self) -> int:
        return len(self.df)

    def _get_id_by_position(self, position: int) -> int:
        assert position < len(self)
        return self.df.iloc[position, self.df.columns.get_loc('id')]

    def _get_image_name_by_position(self, position: int) -> str:
        return f"{self._get_id_by_position(position)}.png"

    def _get_image_path_by_position(self, position: int) -> str:
        image_path: str = os.path.join(self.images_path, self._get_image_name_by_position(position))
        assert os.path.isfile(image_path), image_path
        return image_path

    def _get_image_by_position(self, position: int) -> Image:
        image_path = self._get_image_path_by_position(position)
        image = Image.open(image_path).convert("RGB")
        return self.preprocessing(image)

    def _get_label_by_position(self, position: int) -> int:
        return self.df.iloc[position, self.df.columns.get_loc('label')] if 'label' in self.df.columns else None

    def __getitem__(self, position: int):
        image = self._get_image_by_position(position)
        label = self._get_label_by_position(position)
        if label is None:
            return image
        return image, label


class ApparelStackedDataset(Dataset):
    def __init__(self, *datasets: List[ApparelDataset]) -> None:
        self.datasets = list(datasets)

    def __len__(self) -> int:
        return sum([len(dataset) for dataset in self.datasets])

    @property
    def df(self) -> pd.DataFrame:
        return pd.concat([dataset.df for dataset in self.datasets], ignore_index=True)

    def _map_position(self, position: int) -> Tuple[int]:
        offset = 0
        for dataset in self.datasets:
            if position < len(dataset):
                return offset, position
            position -= len(dataset)
            offset += 1
            assert position >= 0
        raise IndexError("Stacked dataset index out of range!")

    def __getitem__(self, position: int):
        offset, relative_position = self._map_position(position)
        return self.datasets[offset][relative_position]


class ApparelProfiler:
    def __init__(self, label: str) -> None:
        assert label
        self.label = label
        self.measurements = []

    def measure(self, label: str) -> None:
        assert label
        self.measurements.append({
            "label": label,
            "time": time.time()
        })

    def stop(self) -> None:
        self.measure("End")
        print(self)

    @property
    def elapsed_time(self) -> float:
        assert self.measurements
        return self.measurements[-1]['time'] - self.measurements[0]['time']

    @property
    def summary(self) -> dict:
        assert self.elapsed_time
        results = {}
        for i in range(len(self.measurements) - 1):
            elapsed = self.measurements[i + 1]['time'] - self.measurements[i]['time']
            results[self.measurements[i]['label']] = elapsed / self.elapsed_time
        return results
    
    def __str__(self) -> str:
        proportions = ', '.join([f"{label}: {int(100 * p)}%" for label, p in self.summary.items()])
        return f"{self.label}, Elapsed: {self.elapsed_time}, {proportions}"