import cv2
import numpy as np

from uav_dataset import UAVDataset

dataset = UAVDataset()

def show_clip(clip_name):
    cap = dataset.get_capture(clip_name)
    tracks = dataset.get_tracks(clip_name)

    i = 1
    while True:
        i += 1

        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break

        print(frame.shape)
        for track_id, track_frames in tracks.items():
            if i not in track_frames: continue
            bbox_left,bbox_top,bbox_width,bbox_height = np.array(track_frames[i])
            frame = cv2.rectangle(frame, (bbox_left,bbox_top), (bbox_left+bbox_width,bbox_top+bbox_height), (255,255,255))
            cv2.putText(frame, str(track_id), (bbox_left,bbox_top), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        
        cv2.namedWindow(clip_name, cv2.WINDOW_NORMAL)
        cv2.imshow(clip_name, frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cv2.destroyAllWindows()

for clip_name in dataset.get_clip_names():
    show_clip(clip_name)
