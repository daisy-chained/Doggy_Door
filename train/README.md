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
Runs go to `runs/<name>/` (`best/` = best val-mAP weights, `config.json`); metrics + params to `mlflow.db` (SQLite), artifacts to `mlruns/`.

## MLflow UI
```
train/.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Logged: `train_loss`, `val_<metric>` per epoch, `test_<metric>` once for the best epoch, `train_seconds`, `imgs_per_sec_eval`,
params (`model.*`, `imgsz`, `lr`, ..., `export.git_commit`, `export.seed`, `git_commit`, `git_dirty`).
Metrics: `map`, `map_50`, `map_75`, `map_small/medium/large`, `ap_door`, `ap_handle`.

## Config keys (`doordet/config.py`, unknown keys are an error; `key=value` and dotted `model.checkpoint=x` override the YAML)
`export` (required), `model.family`, `model.checkpoint`, `imgsz` (square resize), `epochs`, `batch`, `lr`, `weight_decay`,
`warmup_steps`, `grad_clip`, `amp`, `workers`, `seed`, `device`, `out`, `name`, `mlflow.tracking_uri`, `mlflow.experiment`,
`eval_every` (epochs), `score_threshold` (low, so mAP is meaningful), `test_eval` (default true; false = skip the test split),
`select_metric` (val metric that picks the best epoch, default `map`; any metric name below), `mlflow.tags` (dict, set on the run).

`loop.train(cfg, run_name=None, on_eval=None)`: `on_eval(epoch, val_metrics)` is called after every val eval (an exception stops the run, the search prunes this way).
Returns `best_metric`, `best_value`, `best_val_map`, `test` (None when `test_eval` is false). `epochs: 0` = no training, only the test eval of the loaded weights
(`model.checkpoint=runs/<name>/best` loads what `best/` saved).

## Adapter interface (`doordet/adapters/`)
Register a class in `FAMILIES` (`adapters/__init__.py`); `Adapter(cfg, id2label)` has:
- `model` - the `torch.nn.Module` (loop moves it to the device, optimises it, keeps the best state_dict)
- `collate(batch)` - list of `(PIL image, target)` -> inputs dict (target: `image_id`, `annotations` with 0-based `category_id`, xywh `bbox`)
- `loss(inputs, device)` -> scalar loss tensor
- `predict(inputs, device)` -> `{image_id: {"boxes": xyxy in original pixels, "scores", "labels" 0-based}}` (numpy)
- `save(path)` - must write what `__init__` loads from `model.checkpoint` (so `best/` can be loaded again)
- `example_inputs(imgsz, device)` -> kwargs dict for `model(**kw)` at batch 1 (speed table, `doordet/speed.py`)

## Auto trainer (`train/configs/search_v2.yaml`)
Per hardware budget (GFLOPs per image by default, `budget_metric: ms` for 4060 time), an Optuna search for the best val score among the models and resolutions that fit.
```
train/.venv/bin/python train/speed_table.py train/configs/search_v2.yaml                  # 1. ms + GFLOPs per model x imgsz (GPU, once)
train/.venv/bin/python train/search.py train/configs/search_v2.yaml --budget small_30g     # 2. one search per budget (or --all)
train/.venv/bin/python train/plot_search.py train/configs/search_v2.yaml                  # 3. runs/search/<name>/accuracy_vs_cost.png
```
- Search keys (`doordet/search.py`, unknown keys are an error): `name`, `base` (training config each trial starts from), `overrides` (applied on top of base),
  `speed_table`, `budget_metric` (gflops / ms), `budgets` (name -> max per image, null = no limit), `space` (models, imgsz low/high/step, lr, weight_decay, warmup_steps, batch), `objective`, `pruner`, `stop`,
  `keep_checkpoints`, `seed`, `storage` (Optuna SQLite, default `optuna.db`). `key=value` overrides work like in the training configs.
- A trial only picks model x imgsz combinations measured in the speed table at or under the budget (GFLOPs per image, or ms at batch 1 in training precision; OOM entries are skipped). Why GFLOPs: on the 4060 at batch 1 the ms barely changed with resolution (overhead + noise), GFLOPs grow steadily and hold on any machine.
- `objective`: val metric to maximise, `map` by default, or `ap_door`, `ap_handle`, `map_small`, ... Trials never touch the test split.
- `pruner`: `median` (default) or `hyperband`; `grace_epochs` (default 5) = no trial is stopped before that many epochs.
- `stop`: stops after `patience` finished trials without a gain > `min_delta` (not before `min_trials`), or after `max_hours`.
- `keep_checkpoints`: only the `best/` weights of the best N trials of each search are kept.
- The test split is scored once per search, for the winning trial (a run named `<study>_test`, tag `final_test`).
- Studies are named `<name>_<budget>` and resume if you rerun. MLflow runs carry tags `study`, `trial`, `budget`, `speed_ms`, `gflops`; the speed table is also logged in experiment `doggy-door-speed`.
