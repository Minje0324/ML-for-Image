from vit_pytorch import ViT

class VisionTransformer(ViT):
    def __init__(self, num_classes, image_size=32, patch_size=8):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=128,
            depth=6,
            heads=8,
            mlp_dim=256
        )