from vit_pytorch import ViT


class VisionTransformer(ViT):
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 1000,
        image_size: int = 32,
        patch_size: int = 8,
        dim: int = 128,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 256,
    ):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=input_channels,
        )