import os                  # EKLENDİ
import pandas as pd        # EKLENDİ
import torch
from PIL import Image
from torchvision import transforms

from models.convnext_part_model import ConvNeXtPartModel


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
    print(f"Liste dosyası işleniyor: {test_txt}")

    # --- 1. ADIM: Dosyayı Okuma ve Format Düzeltme (Sizin Kodunuzun Entegresi) ---
    if os.path.exists(test_txt):
        # Önce mevcut içeriği okuyalım
        with open(test_txt, "r", encoding="utf-8") as f_in:
            lines = f_in.readlines()

        formatted_lines = []
        is_modified = False  # Dosyanın değişip değişmediğini takip edelim

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Eğer satır zaten "isim id" formatındaysa (boşluk varsa) dokunmayalım
            # Ancak sizin senaryonuzda dosya bozuk olduğu için zorla düzeltiyoruz.

            # Sizin mantığınız:
            filename = os.path.basename(line)

            # ".jpg" uzantısını kaldır -> "0003_c014_00018740_0"
            name_no_ext = os.path.splitext(filename)[0]

            try:
                # İlk alandan ID'yi çek -> "0003" -> 3
                id_str = name_no_ext.split("_")[0]
                vid = int(id_str)

                # Yeni satırı oluştur: "dosya_adi id"
                new_line = f"{name_no_ext} {vid}\n"
                formatted_lines.append(new_line)
                is_modified = True
            except ValueError:
                # Eğer dosya adı beklenen formatta değilse (örn: ID ile başlamıyorsa)
                print(f"Uyarı: '{line}' satırından ID çıkarılamadı, atlanıyor.")
                continue

        # Eğer düzenleme yapıldıysa dosyayı üzerine yazalım
        if is_modified and len(formatted_lines) > 0:
            print("Dosya formatı düzeltiliyor ve üzerine yazılıyor...")
            with open(test_txt, "w", encoding="utf-8") as f_out:
                f_out.writelines(formatted_lines)
            print("İşlem tamamlandı: test_list.txt güncellendi.")

    # --- 2. ADIM: Pandas ile Okuma (Artık format düzgün olduğu için hata vermez) ---
    print("Veri seti yükleniyor...")
    df = pd.read_csv(
        test_txt,
        sep=r"\s+",  # Boşluk ile ayır
        names=["image", "label"],
        dtype={"image": str, "label": int}  # Artık doğrudan int yapabiliriz
    )

    print(f"Toplam {len(df)} adet resim işleme alındı.")

    # Resim yollarını tam yol (absolute path) haline getir
    # Not: Dosyada artık .jpg yok, resim klasöründe .jpg varsa sonuna eklememiz gerekebilir.
    # Eğer resim dosyalarınız diskte ".jpg" uzantılıysa, aşağıya + ".jpg" eklemeliyiz.
    # Şimdilik sizin kodunuzun çıktısına sadık kalarak uzantısız bırakıyorum.
    # Eğer resim bulunamadı hatası alırsanız buraya `.apply(lambda x: os.path.join(test_img_dir, x + '.jpg'))` yazmalıyız.

    df["path"] = df["image"].apply(lambda x: os.path.join(test_img_dir, x + ".jpg"))

    return df, df
