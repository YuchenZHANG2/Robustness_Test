import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

# -----------------------------
#  Download ~5000 images
# -----------------------------
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="test",
    label_types=["detections"],
    max_samples=10000,
)

# -----------------------------
#  Export FULL dataset
# -----------------------------
dataset.export(
    export_dir="/home/yuchen/YuchenZ/lab/Detector_test/OOD_dataset/OpenImage/Dataset",
    dataset_type=fo.types.COCODetectionDataset,
)

print("Initial dataset exported.")

# -----------------------------
#  Filter images (>1 target detections)
# -----------------------------
# Fixed: using exact case-sensitive labels from Open Images
target_classes = [
    "Ambulance", "Bicycle", "Bus", "Car", "Motorcycle", "Truck",
    "Land vehicle", "Van", "Vehicle",
    "Stop sign", "Traffic light", "Traffic sign",
    "Bicycle wheel", "Wheel"
]

print(f"Total samples before filtering: {len(dataset)}")
print(f"Dataset schema: {dataset.get_field_schema()}")

# -----------------------------------------
# Keep samples that contain AT LEAST ONE
# detection whose label is in target_classes
# (All detections are kept, not just target ones)
# -----------------------------------------
# Manual filtering to keep all samples with at least one target detection
matching_sample_ids = []
for sample in dataset:
    if sample.ground_truth:
        has_target = any(det.label in target_classes for det in sample.ground_truth.detections)
        if has_target:
            matching_sample_ids.append(sample.id)

view = dataset.select(matching_sample_ids)
print(f"Samples after requiring ≥1 target class: {len(view)}")

# -----------------------------------------
# Optional: require more than 1 TOTAL detection
# (of any class — all detections preserved)
# -----------------------------------------
view = view.match(
    F("ground_truth.detections").length() > 1
)

print(f"Samples after requiring >1 total detection: {len(view)}")

# Optional: limit to 1000 samples
view = view.limit(1500)

# -----------------------------------------
# Export dataset (ALL original detections kept)
# -----------------------------------------
view.export(
    export_dir="/home/yuchen/YuchenZ/lab/Detector_test/OOD_dataset/OpenImage/Dataset_final",
    dataset_type=fo.types.COCODetectionDataset,
)

print("Filtered dataset exported.")