# DCapNet

Unified PyTorch pipeline: segmentation (U-Net) -> mask & bbox cropping -> ensemble classification.

## Repo structure

```
DCapNet/
├── models/           # UNet and ensemble wrappers
├── utils/            # Dataset, metrics, visualization
├── train.py          # training helpers
├── DCapNet.py        # main pipeline script (DCapNet function)
├── requirements.txt
└── README.md
```
## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```
## Data layout (expected)

```
data/
├── train/
│   ├── images/
│   ├── masks/
│   └── bboxes/
├── test/
│   ├── images/
│   ├── masks/
│   └── bboxes/
```
Put images in `images/`, segmentation masks in `masks/` (same filenames) and bounding boxes as `.txt` files in `bboxes/` (one numeric row: `x_min y_min width height`).

## Usage

- Run end-to-end pipeline (segmentation + classification):

```bash
python DCapNet.py --train_image_dir data/train/images --train_mask_dir data/train/masks --train_bbox_dir data/train/bboxes \
                  --test_image_dir data/test/images --test_mask_dir data/test/masks --test_bbox_dir data/test/bboxes \
                  --out_dir outputs --seg_epochs 50 --cls_epochs 50
```

Outputs (models, metrics JSONs, plots) will be written to `--out_dir`.
