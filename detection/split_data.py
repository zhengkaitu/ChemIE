import supervision as sv

# Load your full dataset
dataset = sv.DetectionDataset.from_coco(
    images_directory_path="../data/markush_annotations/raw",
    annotations_path="../data/markush_annotations/raw/annotations.json"
)

# Split the dataset (e.g., 80% train, 10% val, 10% test)
# Adjust the split ratios to fit your specific needs
train_dataset, test_val_dataset = dataset.split(split_ratio=0.8)
val_dataset, test_dataset = test_val_dataset.split(split_ratio=0.5)

# Save the split datasets back into COCO format
train_dataset.as_coco(
    images_directory_path="../data/markush_annotations/train",
    annotations_path="../data/markush_annotations/train/_annotations.coco.json"
)
val_dataset.as_coco(
    images_directory_path="../data/markush_annotations/valid",
    annotations_path="../data/markush_annotations/valid/_annotations.coco.json"
)
test_dataset.as_coco(
    images_directory_path="../data/markush_annotations/test",
    annotations_path="../data/markush_annotations/test/_annotations.coco.json"
)
