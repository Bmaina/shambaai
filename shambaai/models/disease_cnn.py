"""
ShambaAI CropDiseaseNet — On-Device CNN for crop disease classification.

Design constraints:
  - Must run on $50 Android phones (TensorFlow Lite export path)
  - Offline inference — zero network dependency
  - < 3 second inference time on ARM Cortex-A55
  - < 15MB model size after INT8 quantisation

Architecture: MobileNetV3-Small backbone (ImageNet pretrained)
  - Depthwise separable convolutions → 5× fewer parameters than ResNet-50
  - Hard-swish activation → efficient on mobile hardware
  - Squeeze-and-Excitation blocks → channel attention for disease focus

40 disease classes covering East African staple crops:
  Maize, Potato, Tomato, Cassava, Bean, Coffee, Tea, Sorghum

Export path: PyTorch → ONNX → TFLite (INT8 quantised)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights
from typing import Optional


# ── Disease Classes ───────────────────────────────────────────────────────────

DISEASE_CLASSES = [
    # Maize (most critical — Fall Armyworm is #1 threat)
    "maize_fall_armyworm",
    "maize_northern_leaf_blight",
    "maize_gray_leaf_spot",
    "maize_common_rust",
    "maize_lethal_necrosis",
    "maize_streak_virus",
    "maize_healthy",

    # Potato & Tomato (blight is catastrophic in highlands)
    "potato_late_blight",
    "potato_early_blight",
    "potato_healthy",
    "tomato_late_blight",
    "tomato_early_blight",
    "tomato_leaf_mold",
    "tomato_bacterial_spot",
    "tomato_healthy",

    # Cassava (mosaic is endemic across East Africa)
    "cassava_mosaic_disease",
    "cassava_brown_streak",
    "cassava_bacterial_blight",
    "cassava_healthy",

    # Bean
    "bean_angular_leaf_spot",
    "bean_rust",
    "bean_healthy",

    # Coffee (major export crop)
    "coffee_berry_disease",
    "coffee_leaf_rust",
    "coffee_wilt",
    "coffee_healthy",

    # Tea (Kenya's #1 export)
    "tea_blister_blight",
    "tea_red_spider_mite",
    "tea_healthy",

    # Sorghum
    "sorghum_anthracnose",
    "sorghum_leaf_blight",
    "sorghum_healthy",

    # Wheat
    "wheat_stem_rust",
    "wheat_stripe_rust",
    "wheat_septoria",
    "wheat_healthy",

    # General pest / cross-crop
    "aphid_infestation",
    "thrips_damage",
    "nutrient_deficiency_nitrogen",
    "nutrient_deficiency_iron",
    "drought_stress",
]

NUM_CLASSES = len(DISEASE_CLASSES)  # 40

# Map disease → recommended treatment action
TREATMENT_MAP = {
    "maize_fall_armyworm": {
        "action": "URGENT — spray within 48 hours",
        "chemical": "Emamectin Benzoate 1.9% EC (e.g. Escort)",
        "dose": "200ml per 20L water",
        "timing": "Early morning or evening",
        "cost_kes": 350,
        "swahili": "Haraka! Nyunyiza dawa leo. Tumia Escort ml 200 kwa maji lita 20.",
    },
    "maize_northern_leaf_blight": {
        "action": "Spray within 5–7 days",
        "chemical": "Mancozeb 80% WP (e.g. Dithane M-45)",
        "dose": "50g per 20L water",
        "timing": "Morning",
        "cost_kes": 180,
        "swahili": "Nyunyiza Dithane gramu 50 kwa maji lita 20 ndani ya siku 7.",
    },
    "potato_late_blight": {
        "action": "URGENT — remove infected plants, spray immediately",
        "chemical": "Cymoxanil + Mancozeb (e.g. Curzate)",
        "dose": "40g per 20L water",
        "timing": "Morning, repeat every 7 days",
        "cost_kes": 420,
        "swahili": "Haraka sana! Ondoa mimea iliyoathiriwa. Nyunyiza Curzate gramu 40.",
    },
    "cassava_mosaic_disease": {
        "action": "Remove and destroy infected stems",
        "chemical": "No chemical — use virus-free cuttings for replanting",
        "dose": "N/A",
        "timing": "N/A",
        "cost_kes": 0,
        "swahili": "Hakuna dawa. Ng'oa mimea iliyoathiriwa. Panda vipandikizi visivyo na ugonjwa.",
    },
    "coffee_berry_disease": {
        "action": "Spray preventively at cherry formation",
        "chemical": "Copper Oxychloride 50% WP (e.g. Copcide)",
        "dose": "60g per 20L water",
        "timing": "Every 14 days during rainy season",
        "cost_kes": 290,
        "swahili": "Nyunyiza Copcide gramu 60 kwa maji lita 20 kila wiki 2.",
    },
    "drought_stress": {
        "action": "Irrigation urgently required",
        "chemical": "Apply mulch + foliar fertiliser (DAP or CAN)",
        "dose": "Irrigation 25–35mm",
        "timing": "Early morning",
        "cost_kes": 150,
        "swahili": "Mpe maji mmea haraka. Weka matandazo. Nyunyiza mbolea ya majani.",
    },
}

# Fill remaining classes with a generic template
for cls in DISEASE_CLASSES:
    if cls not in TREATMENT_MAP:
        crop = cls.split("_")[0].title()
        TREATMENT_MAP[cls] = {
            "action": "Monitor closely. Consult local agronomist if worsening.",
            "chemical": "Consult agro-dealer",
            "dose": "Per label instructions",
            "timing": "Morning",
            "cost_kes": 200,
            "swahili": f"Fuatilia kwa makini. Wasiliana na muuzaji dawa wa karibu.",
        }


# ── Model Architecture ────────────────────────────────────────────────────────

class CropDiseaseNet(nn.Module):
    """
    Lightweight crop disease classifier for on-device (TFLite) deployment.

    Backbone: MobileNetV3-Small (pretrained on ImageNet)
    Head:     Dropout + Linear(576 → NUM_CLASSES)

    Key design choices:
      - MobileNetV3 uses Hard-Swish activation — optimised for ARM CPUs
      - Depthwise separable convolutions: 8–9× fewer multiply-adds vs ResNet
      - SE blocks provide implicit disease-region attention
      - No BatchNorm issues at inference (BN runs in eval mode correctly)
    """
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        try:
            backbone = models.mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            )
        except Exception:
            # Offline fallback — random init (fine for architecture demo)
            backbone = models.mobilenet_v3_small(weights=None)
        # Replace classifier head for our classes
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) RGB float32, normalised [0,1]
        Returns:
            logits: (B, num_classes)
        """
        return self.backbone(x)

    @torch.inference_mode()
    def predict(
        self,
        x: torch.Tensor,
        top_k: int = 3,
        confidence_threshold: float = 0.15,
    ) -> list[dict]:
        """
        Single-image inference with confidence filtering.

        Returns list of top-k predictions:
          [{"class": str, "confidence": float, "treatment": dict}]
        """
        self.eval()
        logits = self(x)
        probs  = F.softmax(logits, dim=-1)

        results = []
        topk_vals, topk_idxs = probs[0].topk(top_k)
        for prob, idx in zip(topk_vals.tolist(), topk_idxs.tolist()):
            if prob < confidence_threshold:
                continue
            cls_name = DISEASE_CLASSES[idx]
            results.append({
                "class":       cls_name,
                "confidence":  round(prob, 4),
                "treatment":   TREATMENT_MAP.get(cls_name, {}),
                "is_healthy":  "healthy" in cls_name,
            })
        return results


# ── Training Utilities ────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced disease datasets.
    Rare diseases (e.g. Maize Lethal Necrosis) are down-weighted
    in standard cross-entropy — focal loss forces the model to learn them.

    α: class weights (for known imbalance ratios)
    γ: focusing parameter (2.0 standard)
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        p     = log_p.exp()
        log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t     = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss    = -((1 - p_t) ** self.gamma) * log_p_t
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss    = alpha_t * loss
        return loss.mean()


def get_transforms(split: str = "train"):
    """
    Data augmentation tuned for field photo conditions in East Africa:
    - Random horizontal flip (fields have no canonical orientation)
    - Colour jitter (varying light conditions: overcast vs bright sun)
    - Random perspective (handheld phone angles)
    - Gaussian blur (out-of-focus leaf photos)
    """
    from torchvision import transforms
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])


# ── ONNX / TFLite Export ──────────────────────────────────────────────────────

def export_onnx(model: CropDiseaseNet, output_path: str = "shambaai_disease.onnx"):
    """Export to ONNX for TFLite conversion pipeline."""
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}},
        opset_version=12,
        verbose=False,
    )
    print(f"[CropDiseaseNet] ONNX exported → {output_path}")
    print("Next: onnx → tf saved_model → tflite (INT8 quantised)")
    print("  pip install onnx-tf tensorflow")
    print("  onnx-tf convert -i shambaai_disease.onnx -o saved_model/")
    print("  tflite_convert --saved_model_dir=saved_model/ \\")
    print("    --output_file=shambaai_disease.tflite \\")
    print("    --optimizations=DEFAULT (INT8 quantisation)")
