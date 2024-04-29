import argparse
from pathlib import Path

import numpy as np
from metrics import MotMetrics
from tracker import Tracker
from uav_dataset import UAVDataset
from flow_roi import plot_bbox
import flow_roi
import cv2
import torch
import sys

class Evaluator:
    def __init__(self, model_name, metrics_name=None,
                 acquire_tracks_periodically=True, 
                 flow_only=False, 
                 viz=False, 
                 metric='iou', 
                 step_size=1, 
                 clip_names=[],
                 num_frames=sys.maxsize
                ):
        self.model_name = model_name
        self.metrics_name = metrics_name
        self.acquire_tracks_periodically = acquire_tracks_periodically
        self.flow_only = flow_only
        self.viz = viz
        self.metric = metric
        self.step_size = step_size
        self.clip_names = clip_names
        self.num_frames = num_frames

    def evaluate(self):
        with torch.no_grad():
            dataset = UAVDataset()
            metrics = MotMetrics(metric=self.metric)
            tracker = Tracker(self.model_name)

            clip_name = self.clip_names[0]
            frames = dataset.get_frames(clip_name)
            
            it = range(0, min(self.num_frames, frames.shape[0]), self.step_size)
            print(f"Number of evaluations: {len(it)}")
            for i in it:
                tracker.add_frame(frames[i])

                if self.flow_only:
                    tracker.update_tracks_flow_only()
                else:
                    tracker.update_tracks()

                if self.acquire_tracks_periodically:
                    if i % 10*self.step_size == 0:
                        tracker.acquire_new_tracks()
                else:
                    if i == 0:                
                        gt_ids, gt_bboxs = dataset.get_bboxs_in_frame(clip_name, i)
                        tracker.add_tracks(gt_bboxs)

                pred_track_ids, pred_bboxs = tracker.get_tracks()
                estimated_rois = tracker.get_estimated_rois()
                
                if self.acquire_tracks_periodically:
                    gt_ids, gt_bboxs = dataset.get_bboxs_in_frame(clip_name, i)
                else:
                    gt_ids, _ = dataset.get_bboxs_in_frame(clip_name, 0)
                    gt_bboxs, status = dataset.get_specific_bboxs_in_frame(clip_name, i, gt_ids)
                    gt_bboxs = gt_bboxs[status==1]
                    gt_ids = np.array(gt_ids)[status.cpu()==1]

                if self.metrics_name is not None:
                    metrics.update(pred_bboxs, gt_bboxs, pred_tracks=pred_track_ids, gt_tracks=gt_ids)
                    print(metrics.get_mot_challenge_metrics())
                
                if self.viz:
                    frame = flow_roi.tensor_to_cv_img_bgr(tracker.get_current_frame()).copy()
                    for track_id, bbox in zip(pred_track_ids, list(pred_bboxs)):
                        frame = flow_roi.plot_bbox(frame, bbox, (255,255,255), label=track_id)
                    for bbox in list(estimated_rois):
                        frame = plot_bbox(frame, bbox, (255,0,0))
                    for bbox in list(gt_bboxs):
                        frame = plot_bbox(frame, bbox, (0,255,0))
                        
                    cv2.namedWindow(clip_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(clip_name, frame)
                    cv2.waitKey(30)

            cv2.destroyAllWindows()

            if self.metrics_name is not None:
                metrics_dir = Path('metrics')
                metrics_dir.mkdir(exist_ok=True)
                metrics.plot_metrics(['idf1', 'idr', 'idp', 'mota'], fname=metrics_dir / f'{self.metrics_name}.png')
                metrics.save_metrics(['idf1', 'idr', 'idp', 'mota'], fname=metrics_dir / f'{self.metrics_name}.csv')

if __name__ == '__main__':
    torch.set_printoptions(precision=2, sci_mode=False)

    parser = argparse.ArgumentParser('evaluate', description='Evaluate the model on a clip')
    parser.add_argument('model', type=str,
                        help='Name of the model file to load from')
    parser.add_argument('-m', '--metrics_name', type=str, default=None,
                        help='Name of file to output metrics to.')
    parser.add_argument('-c','--clip_names', nargs='+', help='Clip names', required=True)
    parser.add_argument('-v', '--viz', action='store_true', 
                        help='Visualize the predictions')
    parser.add_argument('-f', '--flow_only', action='store_true', 
                        help='Update detections using only flow and no network')
    parser.add_argument('-i', '--init_only', action='store_true', 
                        help='Just initialize tracks using ground truths. Evaluation will only occur against these initial tracks. \
                                Do not add new tracks periodically using YOLO')
    parser.add_argument('-M', '--metric', choices=['iou', 'distance'], default='iou',
                        help='Similarity metric for tracks.')
    parser.add_argument('-s', '--step_size', type=int, default=1,
                        help='Number of frames to increment by.')
    parser.add_argument('-n', '--num_frames', type=int, default=sys.maxsize,
                        help='Maximum number of frames to compute. Defaults to all frames.')
    args = parser.parse_args()

    evaluator = Evaluator(
        model_name=args.model,
        metrics_name=args.metrics_name,
        acquire_tracks_periodically=(not args.init_only),
        flow_only=args.flow_only,
        viz=args.viz,
        metric=args.metric,
        step_size=args.step_size,
        clip_names=args.clip_names,
        num_frames=args.num_frames
    )
    evaluator.evaluate()
