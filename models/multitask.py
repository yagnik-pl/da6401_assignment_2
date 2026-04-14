from __future__ import annotations

import os

import torch
import torch.nn as nn

from common import load_checkpoint, resolve_path
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
   

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet.pth",
        dropout_p: float = 0.5,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        # -- Auto-install gdown and download checkpoints from Google Drive ----
        import subprocess, sys
        try:
            import gdown
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
            import gdown
        os.makedirs(os.path.dirname(classifier_path) or "checkpoints", exist_ok=True)
        if not os.path.exists(classifier_path):
            gdown.download(id="1re6kH1RDNXZE0up4isG7lczACYkG6lTx", output=classifier_path, quiet=False, fuzzy=True)
        if not os.path.exists(localizer_path):
            gdown.download(id="1WWIeWpT1L5J_snUEqmz9WD8j6rQ095ZL", output=localizer_path, quiet=False, fuzzy=True)
        if not os.path.exists(unet_path):
            gdown.download(id="1Etjn2Je51deMqFAz3EDlPPkLnKHRMoyi",output=unet_path, quiet=False, fuzzy=True)

        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.image_size = 224

        self.classifier_model = VGG11Classifier(
            num_classes=num_breeds,
            in_channels=in_channels,
            dropout_p=dropout_p,
            use_batch_norm=use_batch_norm,
        )
        self.localizer_model = VGG11Localizer(
            in_channels=in_channels,
            dropout_p=dropout_p,
            image_size=self.image_size,
            use_batch_norm=use_batch_norm,
        )
        self.segmenter_model = VGG11UNet(
            num_classes=seg_classes,
            in_channels=in_channels,
            dropout_p=max(dropout_p * 0.5, 0.1),
            use_batch_norm=use_batch_norm,
        )

        self._initialize_from_checkpoints(
            classifier_path=classifier_path,
            localizer_path=localizer_path,
            unet_path=unet_path,
            num_breeds=num_breeds,
            seg_classes=seg_classes,
            in_channels=in_channels,
            dropout_p=dropout_p,
            use_batch_norm=use_batch_norm,
        )

    def _resolve_checkpoint(self, path: str) -> str:
        candidate = resolve_path(path, base_dir=self.project_root)
        if os.path.exists(candidate):
            return candidate
        cwd_candidate = os.path.abspath(path)
        if os.path.exists(cwd_candidate):
            return cwd_candidate
        basename = os.path.basename(path)
        for search_root in [os.getcwd(), self.project_root]:
            for subdir in ["checkpoints", "checkpoint", "."]:
                p = os.path.join(search_root, subdir, basename)
                if os.path.exists(p):
                    return p
        return candidate

    def _safe_load(self, model: nn.Module, checkpoint_path: str) -> bool:
        resolved = self._resolve_checkpoint(checkpoint_path)
        if not os.path.exists(resolved):
            return False
        try:
            state_dict = load_checkpoint(resolved, map_location="cpu")
        except Exception:
            return False
        try:
            model.load_state_dict(state_dict, strict=True)
            return True
        except RuntimeError:
            try:
                model.load_state_dict(state_dict, strict=False)
                return True
            except Exception:
                return False

    def _initialize_from_checkpoints(
        self,
        classifier_path: str,
        localizer_path: str,
        unet_path: str,
        num_breeds: int,
        seg_classes: int,
        in_channels: int,
        dropout_p: float,
        use_batch_norm: bool,
    ) -> None:
        self._safe_load(self.classifier_model, classifier_path)
        self._safe_load(self.localizer_model, localizer_path)
        self._safe_load(self.segmenter_model, unet_path)

    def _predict_localization(self, bottleneck: torch.Tensor, bottleneck_f: torch.Tensor) -> torch.Tensor:
        box_logits = self.localizer_model.head(bottleneck)
        flipped_boxes = self.localizer_model.head(bottleneck_f).clone()
        flipped_boxes[:, 0] = float(self.image_size) - flipped_boxes[:, 0]
        return 0.5 * (box_logits + flipped_boxes)

    def _predict_segmentation(self, bottleneck: torch.Tensor, skips: dict, bottleneck_f: torch.Tensor, skips_f: dict) -> torch.Tensor:
        logits = self.segmenter_model.decoder(bottleneck, skips)
        flipped_logits = torch.flip(self.segmenter_model.decoder(bottleneck_f, skips_f), dims=[3])
        return 0.5 * (logits + flipped_logits)

    def _predict_classification(self, bottleneck: torch.Tensor, bottleneck_f: torch.Tensor) -> torch.Tensor:
        logits = self.classifier_model.head(bottleneck)
        flip_logits = self.classifier_model.head(bottleneck_f)
        return 0.5 * (logits + flip_logits)

    def forward(self, x: torch.Tensor):
        # 1. Run the shared encoder ONCE on the original image
        bottleneck, skips = self.segmenter_model.encoder(x, return_features=True)
        
        # 2. Run the shared encoder ONCE on the flipped image for TTA
        x_flipped = torch.flip(x, dims=[3])
        bottleneck_f, skips_f = self.segmenter_model.encoder(x_flipped, return_features=True)
        
        # 3. Route the pre-computed feature maps to the individual task heads
        localization = self._predict_localization(bottleneck, bottleneck_f)
        segmentation = self._predict_segmentation(bottleneck, skips, bottleneck_f, skips_f)
        classification = self._predict_classification(bottleneck, bottleneck_f)
        
        return {
            "classification": classification,
            "localization": localization,
            "segmentation": segmentation,
        }