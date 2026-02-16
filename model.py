import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import cv2

# ================= CONFIG =================
MODEL_PATH = r"best_model_v2.pth"
BACKBONE_SIZE = "base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

W = int(((960 * 0.75) // 14) * 14)
H = int(((540 * 0.75) // 14) * 14)
N_CLASSES = 11

CLASS_COLORS = np.array([
    [20,  20,  20 ],   # 0  Background  — near-black
    [0,   100, 0  ],   # 1  Trees       — dark green
    [50,  205, 50 ],   # 2  LushBush    — lime green
    [218, 195, 80 ],   # 3  DryGrass    — straw yellow
    [107, 84,  30 ],   # 4  DryBush     — olive brown
    [139, 90,  43 ],   # 5  GndClut     — earthy brown
    [200, 130, 50 ],   # 6  Class600    — amber/ochre
    [160, 110, 60 ],   # 7  Logs        — wood brown
    [140, 140, 145],   # 8  Rocks       — slate grey
    [120, 72,  30 ],   # 9  Landscape   — warm brown (terrain/ground)
    [170, 220, 255],   # 10 Sky         — pale sky blue
], dtype=np.uint8)

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# ================= MODEL CLASSES =================
class MultiLayerFeatureExtractor(nn.Module):
    def __init__(self, backbone, layer_indices):
        super().__init__()
        self.backbone = backbone
        self.layer_indices = layer_indices
        self._features = {}
        for idx in self.layer_indices:
            self.backbone.blocks[idx].register_forward_hook(self._make_hook(idx))

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            self._features[layer_idx] = output
        return hook_fn

    @torch.no_grad()
    def forward(self, x):
        self._features = {}
        _ = self.backbone.forward_features(x)
        B = x.shape[0]
        h_patches = x.shape[2] // 14
        w_patches = x.shape[3] // 14
        result = []
        for idx in self.layer_indices:
            feat = self._features[idx]
            feat = feat[:, 1:, :]
            embed_dim = feat.shape[-1]
            feat = feat.reshape(B, h_patches, w_patches, embed_dim)
            feat = feat.permute(0, 3, 1, 2)
            result.append(feat)
        return result


class UPerNetHead(nn.Module):
    def __init__(self, embed_dim, num_classes, num_layers=4):
        super().__init__()
        hidden_dim = 256

        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, hidden_dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, hidden_dim),
                nn.GELU()
            ) for _ in range(num_layers)
        ])

        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, hidden_dim),
                nn.GELU()
            ) for _ in range(num_layers)
        ])

        self.ppm_scales = [1, 2, 3, 6]
        ppm_channels = hidden_dim // len(self.ppm_scales)

        self.ppm_convs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(hidden_dim, ppm_channels, kernel_size=1, bias=False),
                nn.GroupNorm(8, ppm_channels),
                nn.GELU()
            ) for scale in self.ppm_scales
        ])

        self.ppm_bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dim + ppm_channels * len(self.ppm_scales),
                      hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, hidden_dim),
            nn.GELU()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * num_layers, 512, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, multi_layer_features):
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, multi_layer_features)]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:],
                mode='bilinear', align_corners=False
            )

        fpn_outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        deepest = fpn_outs[-1]
        h_, w_ = deepest.shape[2:]

        ppm_outs = []
        for ppm_conv in self.ppm_convs:
            pooled = ppm_conv(deepest)
            upsampled = F.interpolate(pooled, size=(h_, w_), mode='bilinear', align_corners=False)
            ppm_outs.append(upsampled)

        fpn_outs[-1] = self.ppm_bottleneck(torch.cat([deepest] + ppm_outs, dim=1))

        target_size = fpn_outs[0].shape[2:]
        aligned = []
        for feat in fpn_outs:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size,
                                     mode='bilinear', align_corners=False)
            aligned.append(feat)

        fused = torch.cat(aligned, dim=1)
        return self.fusion(fused)


# ================= LOAD BACKBONE =================
backbone_archs = {
    "small": "vits14",
    "base":  "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}
layer_map = {
    "small": [2, 5, 8, 11],
    "base":  [2, 5, 8, 11],
    "large": [5, 11, 17, 23],
    "giant": [9, 19, 29, 39],
}

backbone_name = f"dinov2_{backbone_archs[BACKBONE_SIZE]}"
layer_indices = layer_map[BACKBONE_SIZE]

backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
backbone.eval().to(DEVICE)

feature_extractor = MultiLayerFeatureExtractor(backbone, layer_indices).to(DEVICE)

dummy = torch.zeros(1, 3, H, W, device=DEVICE)
with torch.no_grad():
    embed_dim = feature_extractor(dummy)[0].shape[1]

classifier = UPerNetHead(embed_dim, N_CLASSES, len(layer_indices))
classifier.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
classifier.eval().to(DEVICE)

# ================= INFERENCE =================
@torch.no_grad()
def segment_image(pil_image):
    img = transform(pil_image).unsqueeze(0).to(DEVICE)
    feats = feature_extractor(img)
    logits = classifier(feats)
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    pred = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

    color_mask = CLASS_COLORS[pred]
    resized_img = np.array(pil_image.resize((W, H)))
    overlay = cv2.addWeighted(resized_img, 0.6, color_mask, 0.4, 0)

    return overlay
