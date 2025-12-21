import torch
import matplotlib.pyplot as plt
from PIL import Image


def visualize_10_columns_grid(cfg):
    num_queries = cfg["num_queries"]
    device = cfg["device"]

    from data import (
        prepare_data,
        load_model,
        get_transform,
        extract_feature,
    )

    query_df, gallery_df = prepare_data(
        cfg["test_txt"],
        cfg["test_img_dir"]
    )

    model = load_model(
        checkpoint_path=cfg["checkpoint_path"],
        backbone_name=cfg["backbone_name"],
        device=device,
    )

    transform = get_transform(cfg["img_size"])

    print("Extracting gallery features...")
    gallery_feats = []
    gallery_paths = gallery_df["path"].values
    gallery_labels = gallery_df["label"].values

    for path in gallery_paths:
        feat = extract_feature(model, path, transform, device)
        gallery_feats.append(feat)

    gallery_feats = torch.cat(gallery_feats, dim=0)

    selected_queries = query_df.sample(
        n=min(num_queries, len(query_df)),
        random_state=42
    )

    fig, axes = plt.subplots(
        nrows=6,
        ncols=len(selected_queries),
        figsize=(25, 15)
    )

    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    for col_idx, (_, row) in enumerate(selected_queries.iterrows()):
        q_path = row["path"]
        q_label = row["label"]

        q_feat = extract_feature(
            model, q_path, transform, device
        )

        dists = torch.cdist(q_feat, gallery_feats).squeeze(0)
        topk = torch.argsort(dists)[:5]

        # Query row
        ax = axes[0, col_idx]
        img = Image.open(q_path).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"ID: {q_label}", color="blue", fontweight="bold")
        ax.axis("off")

        for spine in ax.spines.values():
            spine.set_edgecolor("blue")
            spine.set_linewidth(4)

        # Gallery rows
        for rank, idx in enumerate(topk):
            ax = axes[rank + 1, col_idx]
            g_path = gallery_paths[idx]
            g_label = gallery_labels[idx]

            img = Image.open(g_path).convert("RGB")
            ax.imshow(img)

            match = g_label == q_label
            color = "green" if match else "red"

            ax.set_title(f"ID: {g_label}", color=color, fontsize=10)
            ax.axis("off")

            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(4)

    plt.show()
