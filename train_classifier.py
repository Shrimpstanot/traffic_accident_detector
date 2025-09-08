# In train_classifier.py

import os
from pathlib import Path
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningDataModule
from pytorch_lightning.tuner import Tuner

# --- Only the transforms we actually use ---
from torchvision.transforms import (
    ColorJitter, Compose, RandomHorizontalFlip, Lambda
)
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths

# --- Key Hyperparameters ---
BATCH_SIZE = 8
NUM_WORKERS = 4
FRAMES_PER_CLIP = 16 # Number of frames per clip is our constant
FPS = 30
CLIP_DURATION = FRAMES_PER_CLIP / FPS 
LEARNING_RATE = 1e-3
MAX_EPOCHS = 10

# --- UPDATE THE DATA_PATH ---
DATA_PATH = "/workspace/datasets/TU-DAT_preprocessed" # We now use the new pre-processed directory

# --- UPDATE THE TRANSFORMS ---
# The train transform now ONLY does fast, on-the-fly augmentations
train_transform = Compose([
    # THE FIX IS HERE: Permute for torchvision's image-based transforms
    Lambda(lambda x: x.permute(1, 0, 2, 3)), # T, C, H, W
    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    Lambda(lambda x: x.permute(1, 0, 2, 3)), # C, T, H, W
    
    RandomHorizontalFlip(p=0.5),
])
# The val transform remains None, which is correct
val_transform = None


class PreprocessedClipDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super().__init__()
        self.data_path = Path(data_path)
        self.transform = transform
        
        # --- THE FIX IS HERE: We manually find all .pt files and get their labels ---
        self._clip_paths = []
        self._labels = []
        
        # Get the class names from the subdirectory names (e.g., "Positive", "Negative")
        class_names = sorted([d.name for d in self.data_path.iterdir() if d.is_dir()])
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.data_path / class_name
            for clip_path in class_dir.glob("*.pt"):
                self._clip_paths.append(clip_path)
                self._labels.append(class_idx)

    def __len__(self):
        return len(self._clip_paths)

    def __getitem__(self, idx):
        # 1. Get the path to the pre-processed .pt file
        clip_path = self._clip_paths[idx]
        
        # 2. Load the tensor from disk (very fast)
        clip_tensor = torch.load(clip_path)

        # Apply any on-the-fly transforms
        if self.transform:
            clip_tensor = self.transform(clip_tensor)

        return {
            "video": clip_tensor,
            "label": self._labels[idx]
        }

class TU_DAT_DataModule(LightningDataModule):
    def __init__(self, data_path, batch_size, num_workers):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = PreprocessedClipDataset(
            data_path=os.path.join(self.data_path, "train"),
            transform=train_transform
        )
        self.val_dataset = PreprocessedClipDataset(
            data_path=os.path.join(self.data_path, "val"),
            transform=val_transform
        )
    
    # The DataLoader definitions are now simple again
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

class VideoClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate

        # Load the pre-trained X3D_N model
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
        
        # Replace the final classification layer for our 2 classes
        num_in_features = self.model.blocks[5].proj.in_features
        self.model.blocks[5].proj = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(num_in_features, 2)
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        videos, labels = batch["video"], batch["label"]
        batch_size = videos.shape[0]
        preds = self(videos)
        loss = self.loss_fn(preds, labels)
        self.log(
            'train_loss', 
            loss, 
            on_step=False,      # Don't log it every batch
            on_epoch=True,      # Log the average at the end of the epoch
            prog_bar=True,      # Show it in the progress bar
            batch_size=batch_size
        )
        return loss

    def validation_step(self, batch, batch_idx):
        videos, labels = batch["video"], batch["label"]
        batch_size = videos.shape[0]
        preds = self(videos)
        loss = self.loss_fn(preds, labels)
        
        # Calculate accuracy
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log('val_loss', loss, batch_size=batch_size)
        self.log('val_acc', acc, batch_size=batch_size)
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
    
    
if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    
    data_module = TU_DAT_DataModule(DATA_PATH, BATCH_SIZE, NUM_WORKERS)
    classifier_model = VideoClassifier(learning_rate=LEARNING_RATE)

    # --- NEW: LEARNING RATE FINDER (Corrected for v2.5.4) ---
    print("Finding optimal learning rate...")
    
    
    # trainer_for_lr_finder = pl.Trainer(accelerator="cuda", devices=1)
    # tuner = Tuner(trainer_for_lr_finder) # Create the Tuner
    
    # Run the LR finder using the tuner
    # lr_finder_results = tuner.lr_find(classifier_model, datamodule=data_module) # Pass datamodule here
    
    # Get the suggested learning rate and print it
    # new_lr = lr_finder_results.suggestion()
    # print(f"Optimal learning rate found: {new_lr}")

    # Optional: Plot the results
    # fig = lr_finder_results.plot(suggest=True)
    # fig.savefig('lr_finder_plot.png')
    
    # --- NOW, TRAIN FOR REAL WITH THE NEW LR ---
    print("Starting main training run...")
    # Re-initialize the model with the new, optimal learning rate
    classifier_model = VideoClassifier(learning_rate=LEARNING_RATE)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints/',
        filename='x3d_m-classifier',
        save_top_k=1,
        mode='max',
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback],
        accelerator="cuda",
        devices=1,
    )

    trainer.fit(classifier_model, data_module)
    print("Training complete.")