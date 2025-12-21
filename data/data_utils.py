import os
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image

from models import ConvNeXtPartModel


def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize(
            img_size,
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_model(checkpoint_path, backbone_name, device):
    print(f"Loading model from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint["model_state_dict"]
    num_classes = state_dict["cls_global.weight"].shape[0]

    model = ConvNeXtPartModel(
        num_classes=num_classes,
        backbone_name=backbone_name
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def extract_feature(model, img_path, transform, device):
    try:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        feat = model(img_tensor)
        return feat.cpu()
    except Exception:
        return torch.zeros(1, model.output_dim)


def prepare_data(test_txt, test_img_dir):
    df = pd.read_csv(
        test_txt,
        sep=r"\s+",
        names=["image", "label"],
        dtype={"image": str, "label": int}
    )

    df["path"] = df["image"].apply(
        lambda x: os.path.join(test_img_dir, f"{x}.jpg")
    )

    query_df = (
        df.groupby("label", group_keys=False)
        .sample(n=1, random_state=42)
        .reset_index(drop=True)
    )

    gallery_df = df.drop(query_df.index).reset_index(drop=True)

    return query_df, gallery_df
