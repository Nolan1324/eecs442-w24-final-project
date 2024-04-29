from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd
import torch
import torchvision

from config import DEVICE


class UAVDataset:
    def __init__(self, vides_pathname='UAV-benchmark-M', gt_pathname='UAV-benchmark-MOTD_v1.0/GT'):
        self.videos_path = Path(vides_pathname)
        self.gt_path = Path(gt_pathname)
        self.gt_data = {}

    def get_clip_names(self):
        for clip_path in sorted(self.videos_path.iterdir()):
            yield clip_path.name

    def get_tracks(self, clip_name: str) -> dict[int, dict[int, list]]:
        """
        Gets dictionary with tracks[track_id][frame_id]
        """
        self._load_gt_data(clip_name)
        gt_data = self.gt_data[clip_name]

        tracks = defaultdict(lambda: {})
        for _, row in gt_data.iterrows():
            frame_index,target_id,bbox_left,bbox_top,bbox_width,bbox_height,score,in_view,occlusion = row.values.flatten().tolist()
            tracks[target_id][frame_index] = [bbox_left,bbox_top,bbox_width,bbox_height]

        return tracks

    def get_bboxs_in_frame(self, clip_name: str, frame_id: int):
        frame_id += 1

        tracks = self.get_tracks(clip_name)
        num_objects = sum(1 for frames in tracks.values() if frame_id in frames)
        output = torch.empty((num_objects,4))
        i = 0
        track_ids = []
        for track_id, frames in tracks.items():
            if frame_id in frames:
                track_ids.append(track_id)
                output[i] = torch.tensor(frames[frame_id], dtype=torch.float32)
                i += 1
        return track_ids, output.to(device=DEVICE)

    def get_specific_bboxs_in_frame(self, clip_name: str, frame_id: int, track_ids: list[int]):
        frame_id += 1

        tracks = self.get_tracks(clip_name)
        output = torch.empty((len(track_ids),4))
        statuses = torch.empty((len(track_ids)))
        for i, track_id in enumerate(track_ids):
            if frame_id in tracks[track_id]:
                output[i] = torch.tensor(tracks[track_id][frame_id], dtype=torch.float32)
                statuses[i] = 1
            else:
                output[i] = torch.zeros(4, dtype=torch.float32)
                statuses[i] = 0
        return output.to(DEVICE), statuses.to(DEVICE)

    def get_capture(self, clip_name: str) -> cv2.VideoCapture:
        video_path = self.get_clip_path(clip_name)
        cap = cv2.VideoCapture(str(video_path / "img%06d.jpg"), cv2.CAP_IMAGES)
        return cap

    def get_frames(self, clip_name: str) -> torch.Tensor:
        video_path = self.get_clip_path(clip_name)
        first_path = video_path / 'img000001.jpg'
        first_image = torchvision.io.read_image(str(first_path))
        num_files = len(list(video_path.iterdir()))
        frames = torch.empty((num_files,) + first_image.shape, device=DEVICE)
        for i, frame_path in enumerate(sorted(video_path.iterdir())):
            frames[i] = torchvision.io.read_image(str(frame_path))
        return frames

    def get_num_frames(self, clip_name: str) -> int:
        video_path = self.get_clip_path(clip_name)
        num_files = len(list(video_path.iterdir()))
        return num_files

    def get_clip_path(self, clip_name: str):
        return self.videos_path / clip_name

    def _load_gt_data(self, clip_name: str):
        if clip_name not in self.gt_data:
            gt_file_path = self.gt_path / f'{clip_name}_gt.txt'
            self.gt_data[clip_name] = pd.read_csv(gt_file_path, header=None)
