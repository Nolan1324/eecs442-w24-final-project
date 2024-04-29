import numpy as np
import torch

import bbox_utils
import flow_roi
import init_tracks
from config import DEVICE
from model import TrackerModel


class Tracker:
    """
    Tracker class for tracking objects in a video.

    This includes the ML model for tracking objects, as well as the flow-guided ROI estimation.
    """

    def __init__(self, model_file_name: str):
        self.model = TrackerModel().to(DEVICE)
        self.model.load_state_dict(torch.load(model_file_name))
        self.model.eval()
        self.bboxs = torch.empty((0,4), device=DEVICE)
        self.estimated_rois = torch.empty((0,4), device=DEVICE)
        self.track_ids = np.array([], dtype=int)

        self.prev_frame = None
        self.current_frame = None

        self.next_track_id = 0

        self.updated = False

    def add_frame(self, frame: torch.Tensor):
        """
        Adds a frame to the tracker.
        """
        if self.current_frame is None:
            self.current_frame = frame
        else:
            self.prev_frame = self.current_frame
            self.current_frame = frame
        self.updated = False


    def add_tracks(self, new_bboxs: torch.Tensor):
        self._add_new_track_ids(new_bboxs.shape[0])
        self.bboxs = torch.cat((self.bboxs, new_bboxs))

    def acquire_new_tracks(self):
        """
        Acquires new tracks for the current frame.
        """
        self.bboxs, self.track_ids, self.next_track_id = init_tracks.init_tracks(self.current_frame, self.bboxs, self.track_ids, self.next_track_id)
        self.updated = True
        assert(self.track_ids.shape[0] == self.bboxs.shape[0])

    def update_tracks(self):
        """
        Updates the tracks using the ML model and flow-guided ROI estimates.
        """
        if len(self.track_ids) == 0: return
        if self.updated:
            print("=== Warning: already updated tracks for this frame ===")
        self.estimated_rois = flow_roi.estimate_rois(self.prev_frame, self.current_frame, self.bboxs)
        pred_bboxs, pred_conf = self.model(self.prev_frame.unsqueeze(0), self.current_frame.unsqueeze(0), self.bboxs, self.estimated_rois)
        pred_bboxs = pred_bboxs[pred_conf>0.3]        
        self.track_ids = self.track_ids[pred_conf.cpu()>0.3]
        self.bboxs = bbox_utils.xywh_center_to_corner(pred_bboxs)
        self.updated = True
        assert(self.track_ids.shape[0] == self.bboxs.shape[0])

    def update_tracks_flow_only(self):
        """
        Updates the tracks purely on flow-guided ROI estimates (no ML model).
        """
        if len(self.track_ids) == 0: return
        if self.updated:
            print("=== Warning: already updated tracks for this frame ===")
        self.estimated_rois = flow_roi.estimate_rois(self.prev_frame, self.current_frame, self.bboxs, size_to_add=0)        
        self.bboxs = self.estimated_rois
        self.updated = True

    def get_tracks(self):
        """
        Gets tracks in (x,y,w,h) where (x,y) is the top-left corner
        """
        if not self.updated:
            print("=== Warning: have not yet ran update_tracks for this frame ===")
        return self.track_ids, self.bboxs
    
    def get_estimated_rois(self):
        """
        Get the estimated ROIs used for updating the tracks
        """
        if not self.updated:
            print("=== Warning: have not yet ran update_tracks for this frame ===")
        return self.estimated_rois

    def get_current_frame(self):
        return self.current_frame

    def _add_new_track_ids(self, num_new_bboxs):
        self.track_ids = np.concatenate((self.track_ids, np.arange(self.next_track_id, self.next_track_id + num_new_bboxs)))
        self.next_track_id += num_new_bboxs
