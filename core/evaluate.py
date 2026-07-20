# -*- coding: utf-8 -*-

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from PIL import Image

from torchvision import models

from torch.utils.data import (
    Dataset,
    DataLoader
)

from sklearn.metrics import (
    f1_score,
    classification_report
)

import albumentations as A

from albumentations.pytorch import ToTensorV2


# CONFIG

IMG_SIZE = 384

BATCH_SIZE = 8

NUM_WORKERS = 8

CHECKPOINT_PATH = "./outputs/best_swin_hpc.pth"

TEST_CSV = (
    "/home/rishabh.r/train/dataset/"
    "CODEBRIM_classification_balanced_dataset/"
    "classification_dataset_balanced/outputs/"
    "test_dataframe.csv"
)


LABEL_COLUMNS = [

    "Background",
    "Crack",
    "Spallation",
    "Efflorescence",
    "ExposedBars",
    "CorrosionStain"

]


IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


# DEVICE

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nDevice: {device}")

if torch.cuda.is_available():

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )


# TRANSFORMS

test_transform = A.Compose([

    A.Resize(
        IMG_SIZE,
        IMG_SIZE
    ),

    A.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),

    ToTensorV2()

])


# DATASET

class CODEBRIMDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = np.array(

            Image.open(
                row["image_path"]
            ).convert("RGB")

        )

        if self.transform:

            image = self.transform(
                image=image
            )["image"]

        labels = torch.tensor(

            row[LABEL_COLUMNS]
            .values
            .astype(np.float32),

            dtype=torch.float32

        )

        return image, labels


# MODEL

def build_model():

    model = models.swin_b(
        weights=None
    )

    in_features = model.head.in_features

    model.head = nn.Sequential(

        nn.Dropout(0.4),

        nn.Linear(
            in_features,
            len(LABEL_COLUMNS)
        )

    )

    return model


# LOAD DATA

print("\nLoading test dataframe...")

test_df = pd.read_csv(TEST_CSV)

test_loader = DataLoader(

    CODEBRIMDataset(
        test_df,
        test_transform
    ),

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=True,

    persistent_workers=True

)


# LOAD MODEL

print("\nBuilding Swin-B...")

model = build_model()

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print("\nCheckpoint loaded successfully.")


# EVALUATION

all_labels = []

all_preds = []

all_probs = []

print("\nRunning evaluation...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        preds = (
            probs > 0.5
        ).float()

        all_labels.append(
            labels.cpu()
        )

        all_preds.append(
            preds.cpu()
        )

        all_probs.append(
            probs.cpu()
        )


all_labels = torch.cat(
    all_labels
).numpy()

all_preds = torch.cat(
    all_preds
).numpy()


# RESULTS

macro_f1 = f1_score(

    all_labels,
    all_preds,

    average="macro",

    zero_division=0

)

print("\n================================================")
print("FINAL EVALUATION RESULTS")
print("================================================")

print(
    f"\nMacro F1 Score: "
    f"{macro_f1:.4f}"
)

print("\n")

print(

    classification_report(

        all_labels,
        all_preds,

        target_names=LABEL_COLUMNS,

        zero_division=0

    )

)

print("================================================")
