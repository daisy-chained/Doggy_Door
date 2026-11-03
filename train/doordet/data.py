"""COCO export folder -> torch Dataset of (PIL RGB image, target) with 0-based category ids.

Ignore boxes (iscrowd 1, from the exporter's --ignore-below) are ignore in scoring (pycocotools reads them from the gt json)
and left out of the training targets, like DETR's COCO loader. A true ignore in D-FINE's loss (matched but unpunished)
would need a custom loss: its denoising queries are built from every target box."""
import json, pathlib
from PIL import Image
from torch.utils.data import Dataset


class CocoDetection(Dataset):
    def __init__(self, export_dir, split):
        self.root = pathlib.Path(export_dir)
        self.gt_path = self.root / "coco" / f"{split}.json"
        coco = json.loads(self.gt_path.read_text())
        self.images = sorted(coco["images"], key=lambda im: im["id"])
        self.id2label = {c["id"] - 1: c["name"] for c in coco["categories"]}
        self.anns = {im["id"]: [] for im in self.images}
        for a in coco["annotations"]:
            if a.get("iscrowd", 0):
                continue
            self.anns[a["image_id"]].append({"bbox": list(a["bbox"]), "category_id": a["category_id"] - 1,
                                             "area": a["area"], "iscrowd": 0})

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = self.images[i]
        img = Image.open(self.root / im["file_name"]).convert("RGB")
        return img, {"image_id": im["id"], "annotations": self.anns[im["id"]]}
