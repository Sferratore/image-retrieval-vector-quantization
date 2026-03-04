import cv2
from pathlib import Path

# cartelle
input_dir = Path(__file__).parent.parent / "data" / "raw"
output_dir = Path(__file__).parent.parent / "data" / "processed"

size = 256

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

    # salva immagine (BGR, senza conversione colore)
    save_path = save_dir / (img_path.stem + ".jpg")
    cv2.imwrite(str(save_path), img)

print("Preprocessing completato.")