import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T, datasets as tv_datasets
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

from models import UNet, EnsembleClassifier
from utils.data import HandSegDataset, create_cropped_from_preds
from utils.visualization import plot_confusion_matrix, plot_training_curves
from train import train_unet, train_classifier

def compute_seg_metrics(unet, loader, device, thresh=0.5):
    unet.to(device)
    unet.eval()
    ious = []
    dices = []
    accs = []
    iou_per_image = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            masks = batch['mask'].cpu().numpy().astype(np.uint8)
            preds = unet(imgs).cpu().numpy()
            preds_bin = (preds >= thresh).astype(np.uint8)
            for t, p in zip(masks, preds_bin):
                t = t.squeeze()
                p = p.squeeze()
                inter = np.logical_and(t, p).sum()
                uni = np.logical_or(t, p).sum()
                iou = inter / uni if uni > 0 else 0.0
                dice = (2*inter) / (t.sum() + p.sum()) if (t.sum()+p.sum())>0 else 0.0
                acc = (t.flatten() == p.flatten()).mean()
                ious.append(iou); dices.append(dice); accs.append(acc)
                iou_per_image.append(iou)
    tp = sum(i >= 0.5 for i in iou_per_image)
    mAP50 = tp / len(iou_per_image) if len(iou_per_image)>0 else 0.0
    return {'mean_iou': float(np.mean(ious)), 'mean_dice': float(np.mean(dices)), 'mean_accuracy': float(np.mean(accs)), 'mAP50': float(mAP50)}

def DCapNet(train_image_dir, train_mask_dir, train_bbox_dir,
            test_image_dir, test_mask_dir, test_bbox_dir,
            out_dir='./dcapnet_output', device=None,
            seg_epochs=50, cls_epochs=50,
            model_names=['resnet50','efficientnet_b0','xception','inceptionv3'],
            batch_size_seg=8, batch_size_cls=32, patience=10, lr_seg=1e-3, lr_cls=1e-4):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, 'plots'); os.makedirs(plots_dir, exist_ok=True)

    # Segmentation dataset & loaders (use train_image_dir/train_mask_dir for training UNet)
    ds = HandSegDataset(train_image_dir, mask_dir=train_mask_dir, bbox_dir=train_bbox_dir, input_size=(256,256))
    train_n = int(0.8 * len(ds)); val_n = len(ds) - train_n
    train_ds, val_ds = random_split(ds, [train_n, val_n], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=batch_size_seg, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size_seg, shuffle=False, num_workers=4)

    # Train UNet
    unet = UNet(in_ch=3, out_ch=1)
    print('Training UNet on', len(train_ds), 'images; validating on', len(val_ds))
    seg_history = train_unet(unet, train_loader, val_loader, device, epochs=seg_epochs, lr=lr_seg, patience=patience)
    torch.save(unet.state_dict(), os.path.join(out_dir, 'unet_final.pth'))
    plot_training_curves(seg_history, out_dir=plots_dir, name='segmentation_history')

    # Evaluate UNet
    seg_metrics = compute_seg_metrics(unet, val_loader, device)
    print('Segmentation metrics:', seg_metrics)
    with open(os.path.join(out_dir, 'seg_metrics.json'), 'w') as f:
        json.dump(seg_metrics, f, indent=2)

    # Create cropped images for classification using the trained UNet
    cropped_train = os.path.join(out_dir, 'cropped_train')
    cropped_val = os.path.join(out_dir, 'cropped_val')
    create_cropped_from_preds(unet, [train_ds[i] for i in range(len(train_ds))], cropped_train, device, target_size=224)
    create_cropped_from_preds(unet, [val_ds[i] for i in range(len(val_ds))], cropped_val, device, target_size=224)
    print('Cropped images saved to', cropped_train, cropped_val)

    # Classification datasets using ImageFolder
    transform_train = T.Compose([T.Resize((224,224)), T.RandomHorizontalFlip(), T.ToTensor(),
                                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    transform_test = T.Compose([T.Resize((224,224)), T.ToTensor(),
                               T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    train_ds_cls = tv_datasets.ImageFolder(cropped_train, transform=transform_train)
    val_ds_cls = tv_datasets.ImageFolder(cropped_val, transform=transform_test)
    train_loader_cls = DataLoader(train_ds_cls, batch_size=batch_size_cls, shuffle=True, num_workers=4)
    val_loader_cls = DataLoader(val_ds_cls, batch_size=batch_size_cls, shuffle=False, num_workers=4)

    num_classes = len(train_ds_cls.classes)
    print('Num classes for classification:', num_classes, 'Classes:', train_ds_cls.classes)

    # Ensemble creation
    ensemble = EnsembleClassifier(model_names, num_classes, pretrained=True, device=device)

    histories = {}
    for name in model_names:
        print('\n🔧 Training', name.upper())
        model = ensemble.models[name]
        history = train_classifier(model, train_loader_cls, val_loader_cls, device, epochs=cls_epochs, lr=lr_cls, patience=patience, name=name)
        histories[name] = history
        plot_training_curves(history, out_dir=plots_dir, name=f'{name}_history')

    # load best weights if saved
    for name in model_names:
        path = f"{name}_best.pth"
        if os.path.exists(path):
            ensemble.models[name].load_state_dict(torch.load(path, map_location=device))

    # Ensemble inference
    def make_fn(s):
        return lambda imgs: torch.nn.functional.interpolate(imgs, size=(s,s), mode='bilinear', align_corners=False)
    resize_fns = { 'resnet50': make_fn(224), 'efficientnet_b0': make_fn(224), 'inceptionv3': make_fn(299), 'xception': make_fn(299) }

    all_preds = []; all_labels = []
    ensemble.eval()
    with torch.no_grad():
        for images, labels in DataLoader(val_ds_cls, batch_size=batch_size_cls, shuffle=False, num_workers=2):
            images = images.to(device)
            probs = ensemble.predict_batch(images, resize_fns)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy()); all_labels.extend(labels.numpy())

    # Classification metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    rmse = float(np.sqrt(mean_squared_error(all_labels, all_preds)))
    error_rate = 1.0 - acc

    cls_metrics = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'rmse': rmse, 'error_rate': error_rate}
    print('Classification metrics:', cls_metrics)
    with open(os.path.join(out_dir, 'cls_metrics.json'), 'w') as f:
        json.dump(cls_metrics, f, indent=2)

    # Confusion matrix and plots
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, train_ds_cls.classes, out_path=os.path.join(plots_dir, 'ensemble_confusion_matrix.png'))

    # Save histories and models
    with open(os.path.join(out_dir, 'cls_histories.json'), 'w') as fh:
        json.dump(histories, fh, default=lambda o: o, indent=2)

    torch.save(unet.state_dict(), os.path.join(out_dir, 'unet_final.pth'))
    for name in model_names:
        torch.save(ensemble.models[name].state_dict(), os.path.join(out_dir, f"{name}_final.pth"))

    print('DCapNet pipeline completed. Outputs saved to', out_dir)
    return {'seg_metrics': seg_metrics, 'cls_metrics': cls_metrics, 'seg_history': seg_history, 'cls_histories': histories}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_image_dir', required=True)
    parser.add_argument('--train_mask_dir', required=True)
    parser.add_argument('--train_bbox_dir', required=False, default=None)
    parser.add_argument('--test_image_dir', required=True)
    parser.add_argument('--test_mask_dir', required=True)
    parser.add_argument('--test_bbox_dir', required=False, default=None)
    parser.add_argument('--out_dir', default='./dcapnet_output')
    parser.add_argument('--seg_epochs', type=int, default=50)
    parser.add_argument('--cls_epochs', type=int, default=50)
    args = parser.parse_args()
    DCapNet(args.train_image_dir, args.train_mask_dir, args.train_bbox_dir, args.test_image_dir, args.test_mask_dir, args.test_bbox_dir, out_dir=args.out_dir, seg_epochs=args.seg_epochs, cls_epochs=args.cls_epochs)
