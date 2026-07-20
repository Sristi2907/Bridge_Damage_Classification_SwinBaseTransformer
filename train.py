# -*- coding: utf-8 -*-

"""
============================================================
CODEBRIM PREMIUM HPC TRAINING PIPELINE
FINAL STABLE VERSION
============================================================
"""

import os
import time
import random
import warnings
import argparse
import faulthandler

warnings.filterwarnings("ignore")
faulthandler.enable()

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from PIL import Image

from torch.utils.data import (
    Dataset,
    DataLoader
)

from torch.optim import AdamW

from torchvision import models

from sklearn.metrics import (
    f1_score
)

import albumentations as A

from albumentations.pytorch import ToTensorV2


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--batch_size",
    type=int,
    default=8
)

parser.add_argument(
    "--num_workers",
    type=int,
    default=8
)

args = parser.parse_args()


# ============================================================
# CONFIG
# ============================================================

CFG = {

    "seed": 42,

    "img_size": 384,

    "batch_size": args.batch_size,

    "num_workers": args.num_workers,

    "epochs": 12,

    "lr": 1e-4,

    "weight_decay": 1e-4,

    "early_stop_patience": 3,

    "output_dir":
        "./outputs",

    "train_csv":
        "/home/rishabh.r/train/dataset/"
        "CODEBRIM_classification_balanced_dataset/"
        "classification_dataset_balanced/outputs/"
        "train_dataframe.csv",

    "val_csv":
        "/home/rishabh.r/train/dataset/"
        "CODEBRIM_classification_balanced_dataset/"
        "classification_dataset_balanced/outputs/"
        "val_dataframe.csv"

}


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


# ============================================================
# SEED
# ============================================================

def seed_everything(seed=42):

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


seed_everything(CFG["seed"])


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.set_float32_matmul_precision("high")

print(f"\nDevice: {device}")

if torch.cuda.is_available():

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"GPU Memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
    )


# ============================================================
# AUGMENTATIONS
# ============================================================

def get_train_transforms():

    return A.Compose([

        A.RandomResizedCrop(
            size=(CFG["img_size"], CFG["img_size"]),
            scale=(0.7, 1.0),
            ratio=(0.75, 1.33),
            p=1.0
        ),

        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.2),

        A.RandomRotate90(p=0.3),

        A.CLAHE(
            clip_limit=4.0,
            tile_grid_size=(8, 8),
            p=0.5
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.6
        ),

        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.4
        ),

        A.GaussNoise(
            std_range=(0.04, 0.1),
            p=0.1
        ),

        A.Blur(
            blur_limit=3,
            p=0.2
        ),

        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(8, 24),
            hole_width_range=(8, 24),
            fill=0,
            p=0.1
        ),

        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),

        ToTensorV2()

    ])


def get_val_transforms():

    return A.Compose([

        A.Resize(
            CFG["img_size"],
            CFG["img_size"]
        ),

        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),

        ToTensorV2()

    ])


# ============================================================
# DATASET
# ============================================================

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

        image = None

        for attempt in range(4):

            try:

                img_obj = Image.open(
                    row["image_path"]
                ).convert("RGB")

                img_obj.load()

                image = np.array(img_obj)

                break

            except Exception:

                if attempt == 3:

                    image = np.zeros(
                        (
                            CFG["img_size"],
                            CFG["img_size"],
                            3
                        ),
                        dtype=np.uint8
                    )

                else:

                    time.sleep(
                        0.05 * (attempt + 1)
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


# ============================================================
# ASYMMETRIC LOSS
# ============================================================

class AsymmetricLoss(nn.Module):

    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8
    ):

        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):

        xs_pos = torch.sigmoid(x)

        xs_neg = 1.0 - xs_pos

        if self.clip and self.clip > 0:

            xs_neg = (
                xs_neg + self.clip
            ).clamp(max=1)

        los_pos = y * torch.log(
            xs_pos.clamp(min=self.eps)
        )

        los_neg = (1 - y) * torch.log(
            xs_neg.clamp(min=self.eps)
        )

        loss = los_pos + los_neg

        asymmetric_w = (

            y * (1 - xs_pos) ** self.gamma_pos +

            (1 - y) * xs_pos ** self.gamma_neg

        )

        loss *= asymmetric_w

        return -loss.mean()


# ============================================================
# MODEL
# ============================================================

def build_model():

    model = models.swin_b(
        weights=models.Swin_B_Weights.IMAGENET1K_V1
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


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(

    model,
    loader,
    criterion,
    optimizer,
    scaler

):

    model.train()

    running_loss = 0.0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

    return running_loss / len(loader)


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    threshold=0.5
):

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_preds = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        with torch.cuda.amp.autocast():

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        probs = torch.sigmoid(outputs)

        preds = (
            probs > threshold
        ).float()

        running_loss += loss.item()

        all_labels.append(
            labels.cpu()
        )

        all_preds.append(
            preds.cpu()
        )

    all_labels = torch.cat(
        all_labels
    ).numpy()

    all_preds = torch.cat(
        all_preds
    ).numpy()

    val_loss = running_loss / len(loader)

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    return val_loss, macro_f1


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    os.makedirs(
        CFG["output_dir"],
        exist_ok=True
    )

    train_df = pd.read_csv(
        CFG["train_csv"]
    )

    val_df = pd.read_csv(
        CFG["val_csv"]
    )

    train_loader = DataLoader(

        CODEBRIMDataset(
            train_df,
            get_train_transforms()
        ),

        batch_size=CFG["batch_size"],

        shuffle=True,

        num_workers=CFG["num_workers"],

        pin_memory=True,

        persistent_workers=True,

        prefetch_factor=4

    )

    val_loader = DataLoader(

        CODEBRIMDataset(
            val_df,
            get_val_transforms()
        ),

        batch_size=CFG["batch_size"],

        shuffle=False,

        num_workers=CFG["num_workers"],

        pin_memory=True,

        persistent_workers=True,

        prefetch_factor=4

    )

    model = build_model()

    model = model.to(device)

    criterion = AsymmetricLoss()

    optimizer = AdamW(

        model.parameters(),

        lr=CFG["lr"],

        weight_decay=CFG["weight_decay"]

    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=12

    )

    scaler = torch.cuda.amp.GradScaler()

    best_f1 = 0.0

    patience_counter = 0

    # ========================================================
    # TRAIN LOOP
    # ========================================================

    for epoch in range(12):

        start = time.time()

        train_loss = train_one_epoch(

            model,
            train_loader,
            criterion,
            optimizer,
            scaler

        )

        val_loss, macro_f1 = validate(

            model,
            val_loader,
            criterion

        )

        scheduler.step()

        print(

            f"\nEpoch {epoch+1} | "

            f"Train Loss: {train_loss:.4f} | "

            f"Val Loss: {val_loss:.4f} | "

            f"Macro F1: {macro_f1:.4f} | "

            f"Time: {time.time()-start:.0f}s"

        )

        # ====================================================
        # SAVE BEST
        # ====================================================

        if macro_f1 > best_f1:

            best_f1 = macro_f1

            patience_counter = 0

            torch.save({

                "model_state_dict":
                    model.state_dict(),

                "macro_f1":
                    best_f1,

                "config":
                    CFG

            },

            os.path.join(
                CFG["output_dir"],
                "best_swin_hpc.pth"
            ))

            print(
                f"✓ New Best Macro-F1: "
                f"{best_f1:.4f}"
            )

        else:

            patience_counter += 1

            print(
                f"No improvement "
                f"({patience_counter}/"
                f"{CFG['early_stop_patience']})"
            )

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if patience_counter >= CFG["early_stop_patience"]:

            print(
                "\nEARLY STOPPING TRIGGERED"
            )

            break

        torch.cuda.empty_cache()

    print("\n================================================")
    print(f"FINAL BEST MACRO-F1: {best_f1:.4f}")
    print("================================================")
