import torchvision.ops
from config import DEVICE
from model import TrackerModel
from uav_dataset import UAVDataset
import torch
from flow_roi import estimate_rois, plot_bbox, tensor_to_cv_img
from bbox_utils import xywh_to_xyxy
import bbox_utils
import cv2
import wandb
import matplotlib.pyplot as plt

from wandb.sdk.wandb_run import Run

import argparse

class Trainer:
    """
    Trainer class for the model. Handles training the model on a set of clips.
    """

    def __init__(self, clip_names: list[str], save_model: str = 'tracker_model.pt', viz_progress: bool = False, num_epochs: int = 3):
        self.model = TrackerModel().to(DEVICE)
        self.dataset = UAVDataset()
        self.mse_loss_fn = torch.nn.MSELoss()
        self.bce_loss_fn = torch.nn.BCELoss()

        self.training_clips = clip_names
        self.num_epochs = num_epochs

        self.save_model = save_model
        self.viz_progress = viz_progress

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.run = None
        self.losses = []

    def set_run(self, run: Run):
        self.run = run

    def train(self):
        """
        Main training loop. Trains the model on the set of clips for the specified number of epochs.
        """
        for epoch in range(self.num_epochs):
            self.model.train()
            avg_loss = self.train_one_epoch()
            print('AVG LOSS', avg_loss)

            if self.run is not None:
                self.run.log({"Average Loss": avg_loss})

            torch.save(self.model.state_dict(), self.save_model)

            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                pass

    def train_one_epoch(self):
        """
        Trains the model on the set of clips for one epoch.
        """
        running_loss = 0.
        count = 0

        for clip_name in self.training_clips:
            frames = self.dataset.get_frames(clip_name)
            num_frames = frames.shape[0]

            for i in range(num_frames - 1):
                current_frame = frames[i]
                next_frame = frames[i+1]
                track_ids, prev_bboxs = self.dataset.get_bboxs_in_frame(clip_name, i)

                grow = torch.rand((prev_bboxs.shape[0], 2)).to(DEVICE) * 100
                shift = torch.rand((prev_bboxs.shape[0], 2)).to(DEVICE) * grow
                estimated_rois = prev_bboxs.clone()
                estimated_rois[:,2:4] += grow
                estimated_rois[:,0:2] -= shift

                current_frame = current_frame.unsqueeze(0)
                next_frame = next_frame.unsqueeze(0)

                self.optimizer.zero_grad()

                pred_bboxs, pred_conf = self.model(current_frame, next_frame, prev_bboxs, estimated_rois)
                gt_bboxs, gt_statuses = self.dataset.get_specific_bboxs_in_frame(clip_name, i+1, track_ids=track_ids)
                pred_bboxs_corner = bbox_utils.xywh_center_to_corner(pred_bboxs)

                if self.viz_progress and i % 20 == 0:
                    self.visualize_progress(clip_name, next_frame, pred_bboxs_corner, estimated_rois, gt_bboxs)

                # gt_bboxs_centered = bbox_utils.xywh_corner_to_center(gt_bboxs)
                pred_bbox_xyxy = xywh_to_xyxy(pred_bboxs_corner[gt_statuses==1])
                gt_bbox_xyxy = xywh_to_xyxy(gt_bboxs[gt_statuses==1])
                iou_loss = torchvision.ops.generalized_box_iou_loss(pred_bbox_xyxy, gt_bbox_xyxy, reduction='mean')
                bce_loss = self.bce_loss_fn(pred_conf, gt_statuses)
                # mse_loss = self.mse_loss_fn(pred_bboxs[gt_statuses==1], gt_bboxs_centered[gt_statuses==1])
                # loss = mse_loss / 540 + bce_loss + iou_loss
                loss = bce_loss + iou_loss

                loss.backward()

                print(f'Frame #{i} loss', loss.item())
                self.losses.append(loss.item())

                if self.run is not None:
                    self.run.log({"Loss": loss.item()})

                self.optimizer.step()

                running_loss += loss.item()

            count += num_frames-1

        return running_loss / count

    def visualize_progress(self, clip_name: str, next_frame: torch.Tensor, pred_bboxs, estimated_rois, gt_bboxs):
        """
        Visualizes the progress of the model during training.
        """
        frame = tensor_to_cv_img(next_frame.squeeze().detach()).copy()

        for bbox in list(pred_bboxs):
            frame = plot_bbox(frame, bbox.detach(), (255,255,255))
        for bbox in list(estimated_rois):
            frame = plot_bbox(frame, bbox.detach(), (255,0,0))
        for bbox in list(gt_bboxs):
            frame = plot_bbox(frame, bbox.detach(), (0,255,0))

        cv2.namedWindow('progress', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('progress', frame.shape[1], frame.shape[0])
        cv2.imshow('progress', frame)
        cv2.waitKey(1)

    def show_loss_graph(self):
        """
        Shows the loss graph for the training process.
        """
        plt.figure()
        plt.plot(range(len(self.losses)), torch.tensor(self.losses, device = 'cpu'))
        plt.xlabel('Iteration')
        plt.ylabel('loss')
        plt.title('Training loss')
        plt.show()


if __name__ == '__main__':
    torch.set_printoptions(precision=2, sci_mode=False)

    parser = argparse.ArgumentParser('train', description='Main training loop')
    parser.add_argument('-c','--clip_names', nargs='+', help='Clip names to train on.', required=True)
    parser.add_argument("-w", "--wandb",
                        help='Name of Weights and Biases run to log to')
    parser.add_argument('-s', '--save_model', type=str, default='tracker_model.pt', 
                        help='Name of the model file to save to')
    parser.add_argument('-v', '--viz_progress', action='store_true', 
                        help='Visualize training progress')
    parser.add_argument('-e', '--epochs', type=int, default=3,
                        help='Number of training epochs. Each epoch is a full pass through the set of training clips')
    args = parser.parse_args()

    trainer = Trainer(args.clip_names, save_model=args.save_model, viz_progress=args.viz_progress, num_epochs=args.epochs)
    if args.wandb:
        with wandb.init(
            project="uav_flow_tracking",
            name=args.wandb
        ) as run:
            trainer.set_run(run)
            trainer.train()
    else:
        try:
            trainer.train()
        except KeyboardInterrupt:
            print('Training interrupted')
            trainer.show_loss_graph()
