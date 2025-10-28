import os
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class HandSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, bbox_dir=None, input_size=(256,256), transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.bbox_dir = Path(bbox_dir) if bbox_dir else None
        self.files = sorted([p.name for p in self.image_dir.iterdir() if p.is_file()])
        self.input_size = input_size
        self.transform = transform or T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img_path = self.image_dir / name
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        mask = None
        if self.mask_dir:
            mpath = self.mask_dir / name
            if mpath.exists():
                m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
                mask = (m > 127).astype('uint8')
            else:
                mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        else:
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        bbox = None
        if self.bbox_dir:
            bpath = self.bbox_dir / (Path(name).stem + '.txt')
            if bpath.exists():
                try:
                    arr = np.loadtxt(str(bpath), dtype=int)
                    if arr.ndim>1:
                        arr = arr[0]
                    bbox = [int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])]
                except Exception:
                    bbox = None

        img_resized = cv2.resize(img, self.input_size)
        img_t = T.ToTensor()(img_resized)
        mask_resized = cv2.resize((mask*255).astype('uint8'), self.input_size, interpolation=cv2.INTER_NEAREST)
        mask_t = torch.from_numpy(mask_resized).unsqueeze(0).float()/255.0

        return {'name': name, 'image': img_t, 'mask': mask_t, 'orig_image': img, 'orig_size': (orig_h, orig_w), 'bbox': bbox}

def apply_mask_and_bbox(orig_img, pred_mask_bin, bbox=None):
    masked = orig_img.copy()
    masked[pred_mask_bin==0] = 0
    if bbox is not None:
        x,y,w,h = bbox
        x2 = min(x+w, orig_img.shape[1])
        y2 = min(y+h, orig_img.shape[0])
        cropped = masked[y:y2, x:x2]
    else:
        ys, xs = np.where(pred_mask_bin>0)
        if len(xs)==0:
            cropped = masked
        else:
            x1,x2 = xs.min(), xs.max()
            y1,y2 = ys.min(), ys.max()
            cropped = masked[y1:y2+1, x1:x2+1]
    return cropped

def create_cropped_from_preds(unet, dataset, out_dir, device, target_size=224, thresh=0.5):
    os.makedirs(out_dir, exist_ok=True)
    unet.eval()
    with torch.no_grad():
        for item in dataset:
            name = item['name']
            orig = item['orig_image']
            img_t = item['image'].unsqueeze(0).to(device)
            pred = unet(img_t).cpu().numpy()[0,0]
            pred_resized = cv2.resize(pred, (orig.shape[1], orig.shape[0]))
            bin_mask = (pred_resized >= thresh).astype('uint8')
            cropped = apply_mask_and_bbox(orig, bin_mask, bbox=item['bbox'])
            if cropped.size == 0:
                cropped = cv2.resize(orig, (target_size, target_size))
            else:
                cropped = cv2.resize(cropped, (target_size, target_size))
            out_path = os.path.join(out_dir, name)
            cv2.imwrite(out_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
    return out_dir
