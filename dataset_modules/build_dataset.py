import lightning as L
from torch.utils.data import DataLoader
from dataset_modules.classification import Classification_Dataset

class BuildDataloaderModule(L.LightningDataModule):
    def __init__(self, config:dict, batch_size:int=32):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.dataset = Classification_Dataset

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.dataset_train = self.dataset(self.config, subset='train')
            self.dataset_val = self.dataset(self.config, subset='val')
            
        self.dataset_test = self.dataset(self.config, subset='test')

    # Recommendation from https://lightning.ai/docs/pytorch/stable/advanced/speed.html
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=8, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=8, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=8)

