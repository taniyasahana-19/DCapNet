import os
import torch
import torch.nn as nn
import torchvision.models as tv
import timm

class EnsembleClassifier(nn.Module):
    def __init__(self, model_names, num_classes, pretrained=True, device=None):
        super().__init__()
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.models = nn.ModuleDict()
        for name in model_names:
            self.models[name] = self._create_model(name, num_classes, pretrained)
        self.to(self.device)

    def _create_model(self, name, num_classes, pretrained=True):
        if name == 'resnet50':
            m = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
        elif name == 'efficientnet_b0':
            m = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        elif name == 'inceptionv3':
            m = tv.inception_v3(weights=tv.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None, aux_logits=True)
            m.AuxLogits.fc = nn.Linear(m.AuxLogits.fc.in_features, num_classes)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
        elif name == 'xception':
            m = timm.create_model('xception', pretrained=pretrained)
            try:
                m.reset_classifier(num_classes)
            except Exception:
                if hasattr(m, 'fc'):
                    m.fc = nn.Linear(m.fc.in_features, num_classes)
        else:
            raise ValueError(f'Unknown model: {name}')
        return m

    def to(self, device):
        self.device = device
        self.models.to(device)

    def eval(self):
        self.models.eval()

    def train(self):
        self.models.train()

    def save_all(self, prefix):
        for k,m in self.models.items():
            torch.save(m.state_dict(), f"{k}_{prefix}.pth")

    def load_all(self, prefix, map_location=None):
        for k,m in self.models.items():
            path = f"{k}_{prefix}.pth"
            if os.path.exists(path):
                m.load_state_dict(torch.load(path, map_location=map_location))

    def predict_batch(self, images, resize_fns=None):
        probs = []
        with torch.no_grad():
            for name, m in self.models.items():
                imgs = images
                if resize_fns and name in resize_fns:
                    imgs = resize_fns[name](images)
                out = m(imgs)
                if isinstance(out, tuple):
                    logits = out[0]
                elif hasattr(out, 'logits'):
                    logits = out.logits
                else:
                    logits = out
                p = torch.softmax(logits, dim=1)
                probs.append(p)
        return torch.mean(torch.stack(probs), dim=0)
