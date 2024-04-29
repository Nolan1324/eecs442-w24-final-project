"""
Given bboxs from the previous frame, use flow to find rois for the next frame
"""

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional

from config import DEVICE

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def tensor_to_cv_img(img):
    return img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

def tensor_to_cv_img_bgr(img):
    return img[[2,1,0]].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

def tensor_to_cv_greyscale(img):
    return torchvision.transforms.functional.rgb_to_grayscale(img).squeeze().cpu().numpy().astype(np.uint8)

def linspace_points_in_bbox(bbox, num_x_steps=5, num_y_steps=5, size_to_add=5):
    x, y, w, h = bbox
    X2D, Y2D = np.meshgrid(np.linspace(y-size_to_add, y+h+2*size_to_add, num_x_steps), np.linspace(x-size_to_add, x+w+2*size_to_add, num_y_steps))
    return np.column_stack((Y2D.ravel(),X2D.ravel())).astype(np.float32)

def estimate_rois(prev_frame, next_frame, bboxes, size_to_add=5): 
    # calculate linearly spaced points in each bbox
    rois = []
    prev_grey = tensor_to_cv_greyscale(prev_frame)
    next_grey = tensor_to_cv_greyscale(next_frame)
    for bbox in list(bboxes):
        points = linspace_points_in_bbox(bbox.cpu().numpy(), size_to_add=size_to_add)    
        # apply flow to the points
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_grey, next_grey, points, None, **lk_params)
        # breakpoint()
        new_points = p1[st[:,0]==1]
        # generate bboxes around the transformed points
        if new_points.shape[0] > 0:
            x1, y1 = np.min(new_points, axis=0)
            x2, y2 = np.max(new_points, axis=0)
            w, h = x2 - x1, y2 - y1
            rois.append(torch.tensor([x1, y1, w, h], dtype=torch.float32, device=DEVICE))
        else:
            rois.append(bbox)
    rois = torch.stack(rois)
    return rois

def plot_bbox(frame, bbox, color, label=None):
    bbox_left,bbox_top,bbox_width,bbox_height = bbox.cpu().numpy().astype(int)
    frame = cv2.rectangle(frame, (bbox_left,bbox_top), (bbox_left+bbox_width,bbox_top+bbox_height), color)
    if label is not None:
        cv2.putText(frame, str(label), (bbox_left,bbox_top), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
    return frame
