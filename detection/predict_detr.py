import glob
import os
import supervision as sv
import torch
from PIL import Image
from rfdetr import RFDETRMedium


OBJECT_TYPES = {
    0: "standard_molecule",
    1: "markush_structure",
    2: "miscellaneous",
    3: "unclassified",
    4: "standard_fragment",
    5: "markush_fragment"
}


def main():
    PTH_PATH = "ckpt/markush_RFDETRMedium_0618/last_rfdetr.pth"
    if not os.path.exists(PTH_PATH):
        print(f"{PTH_PATH} does not exist, converting")
        src = torch.load("ckpt/markush_RFDETRMedium_0618/last.ckpt", map_location="cpu", weights_only=False)
        sd = src.get("state_dict", src)

        # Strip the Lightning module prefix (often "model." — adjust if your print shows "net." etc.)
        prefix = "model."
        new_sd = {
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in sd.items()
        }

        torch.save({"model": new_sd}, PTH_PATH)

        print(f"{PTH_PATH} saved.")

    # Load a pretrained RF-DETR model
    model = RFDETRMedium(pretrain_weights=PTH_PATH)

    fl = sorted(glob.glob("../data/markush_annotations/test/*.png"))

    detected_dir = "predictions/markush_annotations_detected_RFDETRM_0618"
    cropped_dir = "predictions/markush_annotations_cropped_RFDETRM_0618"
    os.makedirs(detected_dir, exist_ok=True)
    os.makedirs(cropped_dir, exist_ok=True)

    # fl = fl[:200]
    # fl = fl[200:400]
    # fl = fl[400:600]
    # fl = fl[600:800]
    # fl = fl[800:1000]
    # fl = fl[1000:1200]
    # fl = fl[1200:]

    box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW)
    label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW)

    for i, fn in enumerate(fl):
        print(f"Predicting for {fn} and saving")

        image = Image.open(fn).convert("RGB")
        detections = model.predict(image, confidence=0.5)
        # print(predictions)
        #
        # detections = sv.Detections.from_inference(predictions)
        # print(detections)

        # labels = [prediction.class_name for prediction in predictions.predictions]
        labels = [OBJECT_TYPES[_id] for _id in detections.class_id]

        annotated_image = image.copy()
        annotated_image = box_annotator.annotate(annotated_image, detections)
        annotated_image = label_annotator.annotate(annotated_image, detections, labels)
        annotated_image.save(os.path.join(detected_dir, os.path.basename(fn)))

        stem = os.path.splitext(os.path.basename(fn))[0]
        cropped_path = os.path.join(cropped_dir, stem)

        for j, (xyxy, label) in enumerate(zip(detections.xyxy, labels)):
            class_dir = os.path.join(cropped_path, label)
            os.makedirs(class_dir, exist_ok=True)
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
            crop = image.crop((x1, y1, x2, y2))
            crop.save(os.path.join(class_dir, f"{stem}_{j}.png"))

        del detections, annotated_image, image


if __name__ == "__main__":
    main()
