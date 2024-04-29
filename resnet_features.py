"""Image encoder model."""

from torchvision.models import ResNet50_Weights, resnet50
from torchvision import transforms
from config import DEVICE

resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE)
for param in resnet.parameters():
    param.requires_grad = False

def get_feature_map(img):
    preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img / 255.)

    x = input_tensor
    x = resnet.conv1(x)
    x = resnet.bn1(x)
    x = resnet.relu(x)
    x = resnet.maxpool(x)

    return x
