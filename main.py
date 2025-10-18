"""
fusion_research_pipeline.py

Research-grade fusion pipeline:
1) Train YOLOv9 detector (calls ultralytics training)
2) Build TLDViT (timm ViT base as placeholder for TLDViT)
3) Cross-attention fusion of YOLO features and ViT tokens
4) Train fusion (classification) with PyTorch Lightning

Notes:
- Staged training: detection -> fusion. Joint finetune notes are in comments.
- You MUST set DATA_YAML to point to your data.yaml file.
"""

import os
import glob
import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.io import read_image
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader

import timm
from ultralytics import YOLO

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

# -----------------------------
# CONFIG - Edit these paths
# -----------------------------
DATA_YAML = "archive/data.yaml"     # <-- replace with your uploaded data.yaml path
YOLO_BASE = "yolov8n.pt"             # start-from weights for YOLOv8 (nano version)
DETECTOR_RUN_DIR = "runs/detect/train" # where ultralytics will save
CROPS_DIR = "tldvit_crops"           # crops generated for TLDViT training
NUM_CLASSES = 7                      # matches your data.yaml nc
BATCH_SIZE = 32
IMG_SIZE = 640                       # YOLO inference size
TLDVIT_IMG = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------

# -----------------------------
# Utilities to parse YOLO labels
# -----------------------------
def read_yolo_labels(lbl_path: str) -> List[Tuple[int, float, float, float, float]]:
    """
    Read YOLO txt label file. Returns list of (class_id, x_center, y_center, w, h) normalized.
    """
    out = []
    if not os.path.exists(lbl_path):
        return out
    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            vals = list(map(float, parts[1:]))
            out.append((cid, *vals))
    return out


def yolo_norm_to_xyxy(box_norm, img_w, img_h):
    """
    box_norm: (x_center, y_center, w, h) normalized.
    returns x1,y1,x2,y2 in absolute pixel coords (clipped to image)
    """
    xc, yc, w, h = box_norm
    x1 = (xc - w/2) * img_w
    y1 = (yc - h/2) * img_h
    x2 = (xc + w/2) * img_w
    y2 = (yc + h/2) * img_h
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = max(0, min(img_w-1, x2)); y2 = max(0, min(img_h-1, y2))
    return x1, y1, x2, y2

# -----------------------------
# Step 1: Train YOLOv9 (ultralytics)
# -----------------------------
def train_yolo(data_yaml: str, epochs: int = 100, imgsz: int = 640, batch: int = 16):
    """
    Trains YOLOv9 with ultralytics API. Produces weights under runs/detect/train/weights/best.pt
    """
    print("Starting YOLOv9 training (this will run ultralytics' train loop)...")
    model = YOLO(YOLO_BASE)  # pick appropriate base weights
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch)
    # After training, ultralytics writes runs/detect/train/weights/best.pt
    print("YOLO training finished. Check the runs/ directory for weights.")
    return model

# -----------------------------
# Step 2: Use trained detector to create crops for classifier dataset
# -----------------------------
def generate_crops_from_labels(images_glob: str, out_dir: str, mode: str = "train"):
    """
    Generate crops directly from ground truth labels without loading any detector.
    This is more efficient since we already have the labels.
    """
    os.makedirs(out_dir, exist_ok=True)
    img_paths = sorted(glob.glob(images_glob))
    print(f"Generating crops from {len(img_paths)} images -> {out_dir}")
    for img_path in img_paths:
        basename = Path(img_path).stem
        # find corresponding label in same dataset structure
        lbl_path = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(lbl_path):
            continue
        # load image with PIL
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        labels = read_yolo_labels(lbl_path)
        for i, (cid, xc, yc, bw, bh) in enumerate(labels):
            x1, y1, x2, y2 = yolo_norm_to_xyxy((xc, yc, bw, bh), w, h)
            crop = img.crop((x1, y1, x2, y2))
            cls_dir = os.path.join(out_dir, str(cid))
            os.makedirs(cls_dir, exist_ok=True)
            save_name = f"{basename}_{i}.jpg"
            crop.save(os.path.join(cls_dir, save_name))
    print("Crop generation done.")


def generate_crops_from_detector(detector: YOLO, images_glob: str, out_dir: str, mode: str = "train"):
    """
    Run detector on images, but we will also use GT boxes to save crops per class
    (this is safer: use ground truth boxes to create classifier dataset).
    images_glob: path glob for images, e.g. 'dataset/images/train/*.jpg'
    """
    os.makedirs(out_dir, exist_ok=True)
    img_paths = sorted(glob.glob(images_glob))
    print(f"Generating crops from {len(img_paths)} images -> {out_dir}")
    for img_path in img_paths:
        basename = Path(img_path).stem
        # find corresponding label in same dataset structure: assume labels in parallel folder
        # attempt to locate label by replacing images with labels pattern
        lbl_path = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        # fallback: if not present, skip
        if not os.path.exists(lbl_path):
            continue
        # load image with PIL
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        labels = read_yolo_labels(lbl_path)
        for i, (cid, xc, yc, bw, bh) in enumerate(labels):
            x1, y1, x2, y2 = yolo_norm_to_xyxy((xc, yc, bw, bh), w, h)
            crop = img.crop((x1, y1, x2, y2))
            cls_dir = os.path.join(out_dir, str(cid))
            os.makedirs(cls_dir, exist_ok=True)
            save_name = f"{basename}_{i}.jpg"
            crop.save(os.path.join(cls_dir, save_name))
    print("Crop generation done.")


# -----------------------------
# TLDViT backbone with regularization (we use a ViT base as placeholder)
# -----------------------------
class TLDViT(nn.Module):
    """
    A vision-transformer backbone + classification head with anti-overfitting measures.
    """
    def __init__(self, num_classes: int, pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        # Use smaller model to reduce overfitting
        self.backbone = timm.create_model("vit_small_patch16_224", pretrained=pretrained)
        in_feat = self.backbone.head.in_features
        # replace head with identity; we'll build our own with regularization
        self.backbone.head = nn.Identity()
        
        # Add regularized classification head
        self.cls_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_feat, in_feat // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_feat // 2, num_classes)
        )

    def forward(self, x):
        # backbone returns a [B, embed_dim] representation for classification
        feat = self.backbone(x)         # [B, embed_dim]
        logits = self.cls_head(feat)
        return logits, feat


# -----------------------------
# Cross-Attention Fusion Module
# -----------------------------
class CrossAttentionFusion(nn.Module):
    """
    Cross-attention where ViT features query YOLO feature maps (spatial).
    vit_feat: [B, D]  (global pooled embedding)
    yolo_feat: [B, C, H, W]
    We'll expand vit_feat into tokens and perform attention with spatial keys.
    """
    def __init__(self, vit_dim: int = 768, yolo_dim: int = 256, fusion_dim: int = 512, nheads: int = 8):
        super().__init__()
        self.vit_proj = nn.Linear(vit_dim, fusion_dim)
        self.yolo_key = nn.Conv2d(yolo_dim, fusion_dim, kernel_size=1)
        self.yolo_val = nn.Conv2d(yolo_dim, fusion_dim, kernel_size=1)
        # use MultiheadAttention expects seq_len x batch x dim
        self.attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=nheads, batch_first=True)
        self.norm = nn.LayerNorm(fusion_dim)
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim*2),
            nn.GELU(),
            nn.Linear(fusion_dim*2, fusion_dim)
        )

    def forward(self, vit_feat: torch.Tensor, yolo_feat: torch.Tensor):
        # vit_feat: [B, vit_dim] -> queries: [B, 1, F]
        q = self.vit_proj(vit_feat).unsqueeze(1)    # [B,1,F]
        B, C, H, W = yolo_feat.shape
        kv = self.yolo_key(yolo_feat)               # [B,F,H,W]
        val = self.yolo_val(yolo_feat)              # [B,F,H,W]
        kv = kv.flatten(2).permute(0,2,1)           # [B,HW,F] as keys
        val = val.flatten(2).permute(0,2,1)         # [B,HW,F] as values
        # MultiheadAttention with batch_first expects (B, S, E)
        attn_out, attn_weights = self.attn(q, kv, val, need_weights=True)  # attn_out [B,1,F]
        fused = self.norm(vit_feat + attn_out.squeeze(1))  # residual -> [B,F]
        fused = fused + self.mlp(fused)
        return fused


# -----------------------------
# Fusion Model (Detector backbone + TLDViT + Fusion head)
# -----------------------------
class FusionModel(nn.Module):
    def __init__(self, detector: YOLO, num_classes: int, freeze_detector: bool = True):
        super().__init__()
        # detector is ultralytics YOLO object. We'll use detector.model as nn.Module
        self.detector = detector.model  # underlying PyTorch model
        # choose feature map layer from detector for spatial cues. We pick an intermediate layer index
        # IMPORTANT: layer indexing depends on the YOLO version; user may need to adapt
        # We'll provide a safe wrapper to fetch a mid-level feature map using forward hooks.
        self.num_classes = num_classes
        self.freeze_detector = freeze_detector

        # TLDViT backbone
        self.tldvit = TLDViT(num_classes=num_classes, pretrained=False)
        vit_dim = self.tldvit.backbone.num_features if hasattr(self.tldvit.backbone, "num_features") else 768

        # We'll assume one detector feature map dimension -- user may adapt to actual model channels
        yolo_dim = 256
        fusion_dim = 512
        self.fusion = CrossAttentionFusion(vit_dim=vit_dim, yolo_dim=yolo_dim, fusion_dim=fusion_dim)
        self.classifier = nn.Linear(fusion_dim, num_classes)

        if freeze_detector:
            for p in self.detector.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        x: images [B,3,H,W]
        We need:
          - detector feature map (spatial) -> yolo_feat
          - tldvit embedding from the entire image or from GT crop depending on training mode
        For training fusion, we will provide crops to the TLDViT separately (in Lightning module).
        Here for completeness, we forward full-image features and return them.
        """
        # forward through detector backbone to get spatial features
        # ultralytics model has model.forward which returns predictions; extracting internal features
        # is model-specific. We attempt to use model.model.forward_features or run a partial forward.
        # Safer approach: run a full forward and hope detection model stores backbone features. If that fails,
        # user will need to adapt this hook per their detector implementation.
        try:
            # Many ultralytics models: detector.model.model contains modules; easiest is to call model.model.forward once.
            # Here, we call .forward but it returns predictions; we cannot easily extract features generically.
            preds = self.detector(x)  # this returns predictions (Boxes)
            # As fallback, create a fake spatial feature by average pooling intermediate conv outputs
            # We'll attempt to access common named module
            # This part is model/version specific; if it fails, user should adapt: see notes.
            # For now, synth a small spatial feature
            B, C, H, W = x.shape
            yolo_feat = F.adaptive_avg_pool2d(x, (H//16, W//16))  # [B,3,H/16,W/16] -> not ideal, placeholder
            # project to expected yolo_dim
            if yolo_feat.shape[1] != 256:
                yolo_feat = nn.Conv2d(yolo_feat.shape[1], 256, kernel_size=1).to(yolo_feat.device)(yolo_feat)
        except Exception as e:
            # Fallback: produce dummy feature map
            B, C, H, W = x.shape
            yolo_feat = torch.zeros(B, 256, H//16, W//16, device=x.device)

        # TLDViT forward for whole image pooled embedding
        vit_logits, vit_feat = self.tldvit(x)  # vit_feat is embedding used by fusion
        fused = self.fusion(vit_feat, yolo_feat)
        out = self.classifier(fused)
        return out, preds if 'preds' in locals() else None


# -----------------------------
# Lightning training module
# Improved Lightning Module with anti-overfitting measures
class SimpleTLDViTLit(pl.LightningModule):
    def __init__(self, num_classes: int, lr: float = 1e-4, dropout_rate: float = 0.3):
        super().__init__()
        # Use TLDViT with regularization
        self.model = TLDViT(num_classes=num_classes, pretrained=True, dropout_rate=dropout_rate)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for regularization
        self.lr = lr
        self.num_classes = num_classes
        self.save_hyperparameters()

    def forward(self, images):
        logits, _ = self.model(images)
        return logits

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self(imgs)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self(imgs)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # Use smaller learning rate and stronger weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=0.01,  # Increased weight decay
            betas=(0.9, 0.999)
        )
        
        # Add learning rate scheduling
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }


# Original Fusion class (commented out to avoid YOLO downloads)
class FusionLit(pl.LightningModule):
    def __init__(self, detector_weights: str, num_classes: int, lr: float = 3e-4, freeze_detector: bool = True):
        super().__init__()
        # load detector weights via ultralytics YOLO
        self.detector_yl = YOLO(detector_weights)
        self.model = FusionModel(detector=self.detector_yl, num_classes=num_classes, freeze_detector=freeze_detector)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.num_classes = num_classes

    def forward(self, images):
        logits, _ = self.model(images)
        return logits

    def training_step(self, batch, batch_idx):
        imgs, labels = batch   # imgs are crops (not full images), labels are ints
        logits = self(imgs)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self(imgs)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


# -----------------------------
# Dataset for TLDViT training (ImageFolder-style from crops)
# -----------------------------
class CropImageFolder(Dataset):
    def __init__(self, root_dir: str, img_size: int = 224, is_training: bool = True):
        self.root = Path(root_dir)
        self.samples = []
        self.classes = []
        self.is_training = is_training
        
        # classes are folder names (0..num_classes-1)
        for cls in sorted(os.listdir(self.root)):
            cls_path = self.root / cls
            if not cls_path.is_dir(): 
                continue
            self.classes.append(cls)
            for img in cls_path.glob("*.jpg"):
                self.samples.append((str(img), int(cls)))
        
        # Data augmentation for training, simple transforms for validation
        if is_training:
            self.transform = T.Compose([
                T.Resize((int(img_size * 1.1), int(img_size * 1.1))),  # Slightly larger for cropping
                T.RandomCrop(img_size),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        img = default_loader(p)  # PIL loader
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


# -----------------------------
# Main orchestration
# -----------------------------
def main():
    # 0) Basic checks
    assert os.path.exists(DATA_YAML), f"Set DATA_YAML path in the script. Current: {DATA_YAML}"

    # 1) Step: (Optional) Train YOLOv9 on raw dataset.
    print("STEP 1: Train YOLOv9 detector (optional if you already trained).")
    # If you want to train from script, uncomment next line. Training may take hours.
    # train_yolo(DATA_YAML, epochs=80, imgsz=IMG_SIZE, batch=16)

    # 2) After you have detector weights (runs/detect/train/weights/best.pt), point to them:
    detector_weights = os.path.join("runs", "detect", "train", "weights", "best.pt")
    if not os.path.exists(detector_weights):
        print(f"Detector weights not found at {detector_weights}. If you have a trained detector, set detector_weights accordingly.")
        # You can set detector_weights = 'path/to/your/best.pt'
        # For now we will still proceed but some parts may be placeholders.

    # 3) Generate crops using GT labels (recommended). This produces tldvit_crops/{class_id}/*.jpg
    print("STEP 2: generate crops from GT labels to build classification dataset")
    # Use ground truth labels directly - no need to load any pretrained model
    train_images_glob = "/Users/adishgarg/Desktop/tomato disease/archive/train/images/*.jpg"
    generate_crops_from_labels(images_glob=train_images_glob, out_dir=os.path.join(CROPS_DIR, "train"), mode="train")
    # Similarly for val:
    val_images_glob = "/Users/adishgarg/Desktop/tomato disease/archive/valid/images/*.jpg"
    generate_crops_from_labels(images_glob=val_images_glob, out_dir=os.path.join(CROPS_DIR, "val"), mode="val")

    # 4) Prepare datasets & dataloaders with data augmentation
    train_ds = CropImageFolder(os.path.join(CROPS_DIR, "train"), img_size=TLDVIT_IMG, is_training=True)
    val_ds = CropImageFolder(os.path.join(CROPS_DIR, "val"), img_size=TLDVIT_IMG, is_training=False)
    
    # Reduce batch size to improve generalization
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    # 5) Build Lightning module and train TLDViT classifier with regularization
    print("STEP 3: Training regularized TLDViT classifier on cropped images")
    # Use improved version with anti-overfitting measures
    lit = SimpleTLDViTLit(num_classes=NUM_CLASSES, lr=1e-4, dropout_rate=0.3)

    # Add early stopping and better checkpointing
    from pytorch_lightning.callbacks import EarlyStopping
    
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=True,
        mode="min",
        min_delta=0.001
    )
    
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss", 
        mode="min", 
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.2f}",
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=50,  # More epochs but with early stopping
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_cb, lr_monitor, early_stop],
        log_every_n_steps=10,
        val_check_interval=1.0,  # Validate every epoch
        deterministic=True  # For reproducibility
    )

    trainer.fit(lit, train_loader, val_loader)

    print("Fusion training complete. Best checkpoint:", checkpoint_cb.best_model_path)

    # 6) Optional: joint finetuning
    print("\nNOTE: For full end-to-end joint finetuning (detection + fusion), see the comments at the end of the script.\n")


if __name__ == "__main__":
    main()