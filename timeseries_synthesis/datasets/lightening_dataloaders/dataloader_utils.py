from torch.utils.data import DataLoader, Dataset
import lightning as L


class BaseDataModule(L.LightningDataModule):
    def __init__(self, config, train_dataset: Dataset, test_dataset: Dataset):
        super().__init__()

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.kwargs = (
            {
                "num_workers": config.num_workers,
                "pin_memory": True,
                "multiprocessing_context": "fork",
            }
            if config.device == "cuda"
            else {"num_workers": config.num_workers}
        )
        self.kwargs["shuffle"] = True

        self.config = config

        print(
            "Loaded Data: Train: {}, Test: {}".format(
                len(train_dataset), len(test_dataset)
            )
        ) 

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.config.dataset.batch_size, **self.kwargs
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.config.dataset.batch_size, **self.kwargs
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.config.dataset.batch_size, **self.kwargs
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.config.dataset.batch_size, **self.kwargs
        )


class TimeWeaverBaseDataModule(L.LightningDataModule):
    def __init__(
        self,
        config,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.kwargs = (
            {
                "num_workers": config.num_workers,
                "pin_memory": True,
                "multiprocessing_context": "fork",
            }
            if config.device == "cuda"
            else {"num_workers": config.num_workers}
        )

        self.config = config

        print(
            "Loaded Data: Train: {}, Val: {}, Test: {}".format(
                len(train_dataset), len(val_dataset), len(test_dataset)
            )
        )

    def train_dataloader(self):
        self.kwargs["shuffle"] = False
        return DataLoader(
            self.train_dataset, batch_size=self.config.dataset.batch_size, **self.kwargs
        )

    def val_dataloader(self):
        self.kwargs["shuffle"] = False
        return DataLoader(
            self.val_dataset, batch_size=self.config.dataset.batch_size, **self.kwargs
        )

    def test_dataloader(self):
        self.kwargs["shuffle"] = False
        return DataLoader(
            self.test_dataset, batch_size=self.config.dataset.batch_size, **self.kwargs
        )
