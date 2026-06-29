import os
import time
import cv2
import yaml
from pathlib import Path
from doctr.models import ocr_predictor
from doctr.io import DocumentFile


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_doctr(image_path, config_path, model=None):
    config = load_config(config_path)
    model_settings = config.get("model_settings", {})

    det_arch = model_settings.get("det_arch", "db_resnet50")
    reco_arch = model_settings.get("reco_arch", "crnn_vgg16_bn")
    pretrained = model_settings.get("pretrained", True)

    if model is None:
        model = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=pretrained)

    img = cv2.imread(image_path)

    start_time = time.time()  

    doc = DocumentFile.from_images(image_path)
    result = model(doc)
    text = result.render()

    execution_time = round(time.time() - start_time, 4)

    config_p = Path(config_path)
    engine = config_p.parent.name 
    model_name = config_p.stem           

    base_viz_dir = os.path.join('outputs', engine, model_name, 'viz')
    highlighted_dir = os.path.join(base_viz_dir, 'highlighted')
    masked_dir = os.path.join(base_viz_dir, 'masked')

    os.makedirs(highlighted_dir, exist_ok=True)
    os.makedirs(masked_dir, exist_ok=True)

    img_name = os.path.basename(image_path)
    h, w, _ = img.shape

    overlay = img.copy()
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    geom = word.geometry
                    if isinstance(geom[0], tuple):
                        x1, y1 = geom[0]
                        x2, y2 = geom[1]
                    else:
                        x1, y1, x2, y2 = geom

                    x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)

    alpha = 0.3
    highlighted_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.imwrite(os.path.join(highlighted_dir, f"highlighted_{img_name}"), highlighted_img)

    synthetic_pages = result.synthesize()
    cv2.imwrite(os.path.join(masked_dir, f"masked_{img_name}"), synthetic_pages[0])

    return {
        "text": text,
        "execution_time_seconds": execution_time,
        "model_used": f"doctr-{det_arch}-{reco_arch}",
        "settings_used": {
            "det_arch": det_arch,
            "reco_arch": reco_arch,
            "pretrained": pretrained,
        },
    }

