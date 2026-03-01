import cv2
import numpy as np
from pathlib import Path

# cartelle
input_dir = Path(__file__).parent / "raw"
output_dir = Path(__file__).parent / "processed"

size = 128

for img_path in input_dir.rglob("*"):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    # mantieni struttura cartelle
    relative = img_path.relative_to(input_dir)
    save_dir = output_dir / relative.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    # carica immagine
    img = cv2.imread(str(img_path))

    if img is None:
        print("skip:", img_path)
        continue

    # resize
    img = cv2.resize(img, (size, size))

    # conversione BGR → Luv
    img_luv = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)

    # salva array numpy
    save_path = save_dir / (img_path.stem + ".npy")
    np.save(save_path, img_luv)

print("Preprocessing completato.")