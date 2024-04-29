from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import motmetrics as mm
import pandas as pd
import torch


class MotMetrics:
    def __init__(self, metric: str = 'iou') -> None:
        self.acc = mm.MOTAccumulator(auto_id=True)
        if metric == 'iou':
            self.metric = partial(mm.distances.iou_matrix, max_iou=0.95)
        elif metric == 'distance':
            self.metric = self.center_euclidian_distance
        else:
            raise NotImplementedError("Argument metric must be one of ('iou', 'center_euclidian')")

    @staticmethod
    def get_bbox_center(bbox):
        return bbox[:, 0:2] + bbox[:, 2:4] / 2

    @staticmethod
    def center_euclidian_distance(bbox1, bbox2):
        return mm.distances.norm2squared_matrix(MotMetrics.get_bbox_center(bbox1), MotMetrics.get_bbox_center(bbox2))

    def update(self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, pred_tracks: list | None = None, gt_tracks: list | None = None) -> None:
        self.acc.update(
            gt_tracks if gt_tracks is not None else list(range(gt_bboxes.shape[0])),
            pred_tracks if pred_tracks is not None else list(range(pred_bboxes.shape[0])),
            self.metric(gt_bboxes.cpu().numpy(), pred_bboxes.cpu().numpy())
        )

    def compute_metrics(self, metrics: list[str], name: str = 'acc') -> pd.DataFrame:
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=metrics, name=name)
        return summary

    def get_mot_challenge_metrics(self) -> str:
        last_frame = self.acc.events.index.get_level_values(0)[-1]
        mh = mm.metrics.create()
        summary = mh.compute_many(
            [self.acc, self.acc.events.loc[last_frame:last_frame]],
            metrics=mm.metrics.motchallenge_metrics,
            names=['full', 'part'])

        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )

        return strsummary

    def plot_metrics(self, metrics: list[str], fname: Path | str, cumulative: bool = True):
        mh = mm.metrics.create()
        last_frame = self.acc.events.index.get_level_values(0)[-1]

        summaries = {i: mh.compute(self.acc.events.loc[(0 if cumulative else i):i], metrics=metrics) for i in range(last_frame + 1)}

        plt.close()
        plt.figure()
        for metric in metrics:
            plt.plot(summaries.keys(), [float(df[metric].iloc[0]) for df in summaries.values()], label=metric)
        plt.legend()
        plt.xlabel('Frame')
        plt.ylabel('Metric value')
        plt.title('Metrics over time')
        plt.savefig(fname)
        plt.close()

    def save_metrics(self, metrics: list[str], fname: Path | str, cumulative: bool = True):
        mh = mm.metrics.create()
        last_frame = self.acc.events.index.get_level_values(0)[-1]

        summaries = {i: mh.compute(self.acc.events.loc[(0 if cumulative else i):i], metrics=metrics) for i in range(last_frame + 1)}

        df = pd.DataFrame({metric: [float(df[metric].iloc[0]) for df in summaries.values()] for metric in metrics})
        df.to_csv(fname)
