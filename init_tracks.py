import cv2
import numpy as np
import torch
import torchvision.ops

from bbox_utils import xywh_to_xyxy, xyxy_to_xywh
from config import DEVICE
from flow_roi import tensor_to_cv_img

LABELS_OF_INTEREST = [2, 5, 7, 8]  # person, bus, truck, boat

yolo = None

def init_tracks(frame, current_tracks, current_track_ids, next_track_id):
    """Run a detector on the current frame and use something like NMS to filter out tracks
    that we already have
    """
    global yolo
    if yolo is None:
        yolo = torch.hub.load("ultralytics/yolov5", "yolov5x6", pretrained=True, device=DEVICE)

    img = tensor_to_cv_img(frame)
    results = yolo(img)

    detections = results.xyxy[0]

    mask = torch.any(
        detections[:, 5, None] == torch.tensor(LABELS_OF_INTEREST, device=DEVICE), dim=1
    )

    yolo_filtered = detections[mask][:, 0:4]
    current_xyxy = xywh_to_xyxy(current_tracks)

    all_bboxes = torch.cat((current_xyxy, yolo_filtered))
    scores = torch.cat(
        (
            torch.ones(current_xyxy.shape[0], device=DEVICE),
            torch.zeros(yolo_filtered.shape[0], device=DEVICE),
        )
    )

    all_track_ids = np.concatenate((current_track_ids, np.arange(next_track_id, next_track_id + yolo_filtered.shape[0])))

    merged_indices = torchvision.ops.nms(all_bboxes, scores, 0.1)
    merged = all_bboxes[merged_indices]
    merged_track_ids = all_track_ids[merged_indices.cpu()]
    new_next_track_id = max(next_track_id, merged_indices.cpu().max())

    return xyxy_to_xywh(merged), merged_track_ids, new_next_track_id
