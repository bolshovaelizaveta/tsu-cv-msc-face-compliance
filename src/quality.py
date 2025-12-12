import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from src.config import BASE_DIR

# Определение архитектуры 
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self, block, layers, dropout=0, num_features=512, groups=1, width_per_group=64):
        super(IResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * 7 * 7, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        # x = self.features(x)
        return x

def iresnet50():
    return IResNet(IBasicBlock, [3, 4, 14, 3])

class FaceQualityController:
    def __init__(self):
        # Путь к исходному файлу весов
        # Ищем файл .pth в корне или в папке models
        self.model_path = os.path.join(BASE_DIR, "magface_iresnet50_MS1MV2.pth")
        
        if not os.path.exists(self.model_path):
             self.model_path = os.path.join(BASE_DIR, "models", "magface_iresnet50_MS1MV2.pth")

        # Определяем устройство: MPS (Mac M-chip) 
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        print(f"Загрузка PyTorch модели из {self.model_path}...")
        
        self.model = iresnet50()
        
        # Загрузка весов
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            # Чистка ключей (module. и features.features.)
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            state_dict = {k.replace('features.features.', 'features.'): v for k, v in state_dict.items()}
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval() # Режим инференса 
        except Exception as e:
            print(f"ERROR: Не удалось загрузить веса PyTorch: {e}")
            raise e

        # Точки для выравнивания
        self.ref_pts = np.array([
            [30.2946, 51.6963], [65.5318, 51.6963],
            [48.0252, 71.7366],
            [33.5493, 92.3655], [62.7299, 92.3655]
        ], dtype=np.float32)
        self.mp_indices = [33, 263, 1, 61, 291]

    def align_face(self, image: np.ndarray, landmarks_obj) -> np.ndarray:
        h, w, _ = image.shape
        src_pts = []
        for idx in self.mp_indices:
            lm = landmarks_obj.landmark[idx]
            src_pts.append([lm.x * w, lm.y * h])
        src_pts = np.array(src_pts, dtype=np.float32)

        tform = cv2.estimateAffinePartial2D(src_pts, self.ref_pts)[0]
        if tform is None:
            return cv2.resize(image, (112, 112))
        return cv2.warpAffine(image, tform, (112, 112))

    def get_quality_score(self, image: np.ndarray, landmarks_obj) -> float:
        # Выравнивание
        aligned = self.align_face(image, landmarks_obj)
        
        # Препроцессинг (BGR -> RGB)
        img_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        
        # Нормализация как в оригинале: (x - 127.5) / 128.0
        img_tensor = torch.from_numpy(img_rgb).float()
        img_tensor = (img_tensor - 127.5) / 128.0
        
        # (H, W, C) -> (C, H, W) и добавляем Batch dimension -> (1, 3, 112, 112)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        # Инференс
        with torch.no_grad():
            embedding = self.model(img_tensor)
            
            # embedding shape: [1, 512]
            norm = torch.norm(embedding, p=2, dim=1).item()
            # Масштабирование 
            final_score = norm * 20.0
            
        return float(final_score)