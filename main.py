import torch
import yaml
from visualization import visualize_10_columns_grid


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    # 1. Konfigürasyonu yükle
    print("Konfigürasyon dosyası okunuyor...")
    cfg = load_config("test.yaml")

    # 2. 'device' hatası çözümü: Otomatik cihaz seçimi
    # Eğer sistemde GPU varsa 'cuda', yoksa 'cpu' olarak ayarlar.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["device"] = device
    print(f"Çalışma cihazı ayarlandı: {device}")

    # 3. Olası 'backbone' hatası için önlem
    # test.yaml dosyasında 'backbone' yazıyor ama visualize.py 'backbone_name' arıyor olabilir.
    if "backbone" in cfg and "backbone_name" not in cfg:
        cfg["backbone_name"] = cfg["backbone"]

    # 4. Fonksiyonu çalıştır
    visualize_10_columns_grid(cfg)