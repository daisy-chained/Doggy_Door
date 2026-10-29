# train/ - door/handle detector training

Fine-tunes a detector on an export from `sim/doorsim/export.py` (COCO json + images), logs to MLflow.

## Setup
```
uv venv train/.venv --python 3.12 && uv pip install --python train/.venv/bin/python torch torchvision transformers timm mlflow pycocotools pyyaml pillow pytest
```

## Run
```
train/.venv/bin/python train/train.py train/configs/dfine_s_640.yaml [key=value ...]   # e.g. imgsz=480 model.checkpoint=... name=sweep480
train/.venv/bin/python train/train.py train/configs/smoke.yaml name=smoke              # quick end-to-end check
train/.venv/bin/python -m pytest train/tests -q                                        # unit tests, no GPU or downloads
```
Runs go to `runs/<name>/` (`best/` = best val-mAP weights, `config.json`); metrics, params and artifacts to `mlruns/`.

## MLflow UI
```
train/.venv/bin/mlflow ui --backend-store-uri mlruns
```
Logged: `train_loss`, `val_<metric>` per epoch, `test_<metric>` once for the best epoch, `train_seconds`, `imgs_per_sec_eval`,
params (`model.*`, `imgsz`, `lr`, ..., `export.git_commit`, `export.seed`, `git_commit`, `git_dirty`).
Metrics: `map`, `map_50`, `map_75`, `map_small/medium/large`, `ap_door`, `ap_handle`.

## Config keys (`doordet/config.py`, unknown keys are an error; `key=value` and dotted `model.checkpoint=x` override the YAML)
`export` (required), `model.family`, `model.checkpoint`, `imgsz` (square resize), `epochs`, `batch`, `lr`, `weight_decay`,
`warmup_steps`, `grad_clip`, `amp`, `workers`, `seed`, `device`, `out`, `name`, `mlflow.tracking_uri`, `mlflow.experiment`,
`eval_every` (epochs), `score_threshold` (low, so mAP is meaningful).

## Adapter interface (`doordet/adapters/`)
Register a class in `FAMILIES` (`adapters/__init__.py`); `Adapter(cfg, id2label)` has:
- `model` - the `torch.nn.Module` (loop moves it to the device, optimises it, keeps the best state_dict)
- `collate(batch)` - list of `(PIL image, target)` -> inputs dict (target: `image_id`, `annotations` with 0-based `category_id`, xywh `bbox`)
- `loss(inputs, device)` -> scalar loss tensor
- `predict(inputs, device)` -> `{image_id: {"boxes": xyxy in original pixels, "scores", "labels" 0-based}}` (numpy)
- `save(path)`
