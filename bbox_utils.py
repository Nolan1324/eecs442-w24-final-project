import torch


def convert_detections(ROIs: torch.Tensor, detections: torch.Tensor) -> torch.Tensor:
    outer = ROIs.reshape((-1, 2, 2))
    xy = outer[:, 0, :]
    wh = outer[:, 1, :]
    inner = detections.reshape((-1, 2, 2))
    inner_pixels = inner * wh.unsqueeze(1)
    inner_pixels[:, 0, :] += xy
    converted = inner_pixels.reshape((-1, 4))
    rois_two_point = xywh_to_xyxy(ROIs)
    converted_two_point = xywh_to_xyxy(converted)
    assert (rois_two_point[:, 0] <= converted_two_point[:, 0]).all() and (
        converted_two_point[:, 0] <= rois_two_point[:, 2]
    ).all()
    assert (rois_two_point[:, 1] <= converted_two_point[:, 1]).all() and (
        converted_two_point[:, 1] <= rois_two_point[:, 3]
    ).all()
    return converted


def xywh_to_xyxy(bbox: torch.Tensor) -> torch.Tensor:
    bbox = bbox.clone()
    bbox[:, 2:4] += bbox[:, 0:2]
    return bbox


def xyxy_to_xywh(bbox: torch.Tensor) -> torch.Tensor:
    bbox = bbox.clone()
    bbox[:, 2:4] -= bbox[:, 0:2]
    return bbox


def xywh_corner_to_center(bbox: torch.Tensor) -> torch.Tensor:
    bbox = bbox.clone()
    bbox[:, 0:2] += bbox[:, 2:4] / 2
    return bbox


def xywh_center_to_corner(bbox: torch.Tensor) -> torch.Tensor:
    bbox = bbox.clone()
    bbox[:, 0:2] -= bbox[:, 2:4] / 2
    return bbox


def normalize_xywh(bbox, W, H):
    return bbox / torch.tensor([W, H, W, H], device=bbox.device)
