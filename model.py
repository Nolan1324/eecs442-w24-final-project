import torch.nn
import torchvision
import torchvision.ops

from bbox_utils import convert_detections, xywh_to_xyxy
from resnet_features import get_feature_map


class TrackerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(128, 64, 5, 1, 2)
        self.conv2 = torch.nn.Conv2d(64, 32, 5, 1, 2)
        self.conv3 = torch.nn.Conv2d(32, 8, 5, 1, 2)
        self.fc = torch.nn.Linear(8*8*8, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU(inplace=True)

        self.roi_align_size = (64, 64)
        self.roi_align = torchvision.ops.RoIAlign(output_size=self.roi_align_size, spatial_scale=0.25, sampling_ratio=-1)

        self.seq = torch.nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.conv3,
            self.relu,
            self.pool,
            torch.nn.Flatten(start_dim=1),
            self.fc,
            torch.nn.Sigmoid()
        )

        self.init_weights()

    def init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.fc]:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.constant_(layer.bias, 0.0)

    def forward(self, prev_frame: torch.Tensor, next_frame: torch.Tensor, prev_bbox: torch.Tensor, next_bbox: torch.Tensor):
        """
        - prev_frame - (N, C, H, W)
        - next_frame - (N, C, H, W)
        - prev_bbox - (L, 4)
        - next_bbox - (L, 4)
        """
        prev_frame = get_feature_map(prev_frame)
        next_frame = get_feature_map(next_frame)

        prev_roi = self.roi_align(prev_frame, [xywh_to_xyxy(prev_bbox)])
        next_roi = self.roi_align(next_frame, [xywh_to_xyxy(next_bbox)])

        input_ = torch.cat((prev_roi, next_roi), dim=1)
        result = self.seq(input_)
        
        converted = convert_detections(next_bbox, result[:,:4])

        return converted, result[:,4]
