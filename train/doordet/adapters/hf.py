"""Hugging Face object detectors (D-FINE, RT-DETR v2, ...) via AutoModelForObjectDetection."""
from collections.abc import Mapping
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def _to(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, Mapping):
        return {k: _to(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_to(v, device) for v in x]
    return x


class HFAdapter:
    def __init__(self, cfg, id2label):
        ckpt, size = cfg["model"]["checkpoint"], {"height": cfg["imgsz"], "width": cfg["imgsz"]}
        self.threshold = cfg["score_threshold"]
        self.processor = AutoImageProcessor.from_pretrained(ckpt, size=size, do_resize=True)
        self.model = AutoModelForObjectDetection.from_pretrained(
            ckpt, id2label=dict(id2label), label2id={v: k for k, v in id2label.items()}, ignore_mismatched_sizes=True)

    def collate(self, batch):
        images = [img for img, _ in batch]
        targets = [{"image_id": t["image_id"], "annotations": t["annotations"]} for _, t in batch]
        enc = self.processor(images=images, annotations=targets, return_tensors="pt")
        labels = enc.pop("labels")
        return {"enc": dict(enc), "labels": labels, "image_ids": [t["image_id"] for _, t in batch],
                "sizes": [img.size[::-1] for img in images]}

    def loss(self, inputs, device):
        enc, labels = _to(inputs["enc"], device), _to(inputs["labels"], device)
        return self.model(**enc, labels=labels).loss

    @torch.no_grad()
    def predict(self, inputs, device):
        out = self.model(**_to(inputs["enc"], device))
        outputs = self.processor.post_process_object_detection(
            out, threshold=self.threshold, target_sizes=torch.tensor(inputs["sizes"]))
        return {i: {"boxes": r["boxes"].float().cpu().numpy(), "scores": r["scores"].float().cpu().numpy(),
                    "labels": r["labels"].cpu().numpy()} for i, r in zip(inputs["image_ids"], outputs)}

    def example_inputs(self, imgsz, device):
        return {"pixel_values": torch.zeros(1, 3, imgsz, imgsz, device=device)}

    def save(self, path):
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
