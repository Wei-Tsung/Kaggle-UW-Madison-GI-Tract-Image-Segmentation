import numpy as np
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    RandSpatialCropd,
    EnsureTyped,
    CastToTyped,
    NormalizeIntensityd,
    RandFlipd,
    CenterSpatialCropd,
    ScaleIntensityRanged,
    RandAffined,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandCoarseDropoutd,
    Rand2DElasticd,
    Lambdad,
    Resized,
    AddChanneld,
    RandGaussianNoised,
    RandGridDistortiond,
    RepeatChanneld,
    Transposed,
    OneOf,
    EnsureChannelFirstd,
    RandLambdad,
    Spacingd,
    FgBgToIndicesd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    ToDeviced,
    SpatialPadd,

)

from default_config import basic_cfg

cfg = basic_cfg

# train
cfg.train = True
cfg.eval = True
cfg.eval_epochs = 2
cfg.start_eval_epoch = 1  # when use large lr, can set a large num
cfg.run_org_eval = False
cfg.run_tta_val = True
cfg.load_best_weights = False
cfg.amp = False
cfg.val_amp = False
# lr
# warmup_restart, cosine
cfg.lr_mode = "cosine"
cfg.lr = 1e-4
cfg.min_lr = 1e-5
cfg.weight_decay = 3e-6
cfg.epochs = 100
cfg.restart_epoch = 3  # only for warmup_restart

cfg.finetune_lb = -1

# dataset
cfg.img_size = (160, 160, 80)
cfg.spacing = (1.5, 1.5, 1.5)
cfg.batch_size = 4
cfg.val_batch_size = 1
cfg.train_cache_rate = 0.0
cfg.val_cache_rate = 1.0
cfg.gpu_cache = False
cfg.val_gpu_cache = False

# val
cfg.roi_size = (224, 224, 80)
cfg.sw_batch_size = 4

# model

#cfg.output_dir = "./output/unet_3d_multilabel_large_net"
        
# transforms
cfg.train_transforms = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        # Spacingd(keys=["image", "mask"], pixdim=cfg.spacing, mode=("bilinear", "nearest")),
        RandSpatialCropd(
            keys=("image", "mask"),
            roi_size=cfg.img_size,
            random_size=False,
        ),
        # CropForegroundd(keys=["image", "mask"], source_key="image", select_fn=lambda x: x > 100),
        # RandCropByPosNegLabeld(
        #     keys=["image", "mask"],
        #     label_key="mask",
        #     spatial_size=cfg.img_size,
        #     pos=4,
        #     neg=1,
        #     num_samples=1,
        #     image_key="image",
        #     image_threshold=0,
        # ),
        Lambdad(keys="image", func=lambda x: x / x.max()),
        # SpatialPadd(keys=("image", "mask"), spatial_size=cfg.img_size),
        # Resized(keys=("image", "mask"), spatial_size=(224, 224, -1), mode="nearest"),
        # ScaleIntensityRanged(
        #     keys=["image"], a_min=0.0, a_max=255.0,
        #     b_min=0.0, b_max=1.0, clip=True,
        # ),
        # FgBgToIndicesd(
        #     keys="mask",
        #     fg_postfix="_fg",
        #     bg_postfix="_bg",
        #     image_key="image",
        # ),
        # ToDeviced(keys=["image", "mask"], device="cuda:0"),
        RandFlipd(keys=("image", "mask"), prob=0.5, spatial_axis=[0]),
        RandFlipd(keys=("image", "mask"), prob=0.5, spatial_axis=[1]),
        # RandFlipd(keys=("image", "mask"), prob=0.5, spatial_axis=[2]),
        RandAffined(
            keys=("image", "mask"),
            prob=0.5,
            rotate_range=np.pi / 12,
            translate_range=(cfg.img_size[0]*0.0625, cfg.img_size[1]*0.0625),
            scale_range=(0.1, 0.1),
            mode="nearest",
            padding_mode="reflection",
        ),
        OneOf(
            [
                RandGridDistortiond(keys=("image", "mask"), prob=0.5, distort_limit=(-0.05, 0.05), mode="nearest", padding_mode="reflection"),
                RandCoarseDropoutd(
                    keys=("image", "mask"),
                    holes=5,
                    max_holes=8,
                    spatial_size=(1, 1, 1),
                    max_spatial_size=(12, 12, 12),
                    fill_value=0.0,
                    prob=0.5,
                ),
            ]
        ),
        RandScaleIntensityd(keys="image", factors=(-0.2, 0.2), prob=0.5),
        RandShiftIntensityd(keys="image", offsets=(-0.1, 0.1), prob=0.5),
        EnsureTyped(keys=("image", "mask"), dtype=torch.float32),
    ]
)

cfg.val_transforms = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        # Spacingd(keys=["image", "mask"], pixdim=cfg.spacing, mode=("bilinear", "nearest")),
        Lambdad(keys="image", func=lambda x: x / x.max()),
        # SpatialPadd(keys=("image", "mask"), spatial_size=cfg.img_size),
        # Resized(keys=("image", "mask"), spatial_size=(224, 224, -1), mode="nearest"),
        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # ScaleIntensityRanged(
        #     keys=["image"], a_min=0.0, a_max=255.0,
        #     b_min=0.0, b_max=1.0, clip=True,
        # ),
        # CropForegroundd(keys=["image", "mask"], source_key="image"),
        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=("image", "mask"), dtype=torch.float32),
        # ToDeviced(keys=["image", "mask"], device="cuda:0"),
    ]
)

cfg.org_val_transforms = Compose(
    [
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        # Spacingd(keys="image", pixdim=cfg.spacing, mode="bilinear"),
        Lambdad(keys="image", func=lambda x: x / x.max()),
        # SpatialPadd(keys="image", spatial_size=cfg.img_size),
        EnsureTyped(keys="image", dtype=torch.float32),
    ]
)


