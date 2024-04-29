import shutil
from pathlib import Path

import bbox_utils
from uav_dataset import UAVDataset

if __name__ == '__main__':
    output_path = Path('yolo_data')
    output_path.mkdir(exist_ok=True)

    W, H = 1024, 540

    if len(list(output_path.iterdir())) > 0:
        print("Output directory should be empty")
        exit(1)

    dataset = UAVDataset()
    clip_names = ['M1301', 'M1302', 'M1303', 'M1304', 'M1305', 'M1306']
    frame_step = 20

    for clip_name in clip_names:
        print(f'=== Writing labels for {clip_name} ===')
        images_dir_path = output_path / 'images' / clip_name
        labels_dir_path = output_path / 'labels' / clip_name
        images_dir_path.mkdir(parents=True, exist_ok=True)
        labels_dir_path.mkdir(parents=True, exist_ok=True)
        tracks = dataset.get_tracks(clip_name)
        for i in range(0, dataset.get_num_frames(clip_name), frame_step):
            img_name = f'img{(i+1):06d}'
            image_src_path = Path('UAV-benchmark-M') / clip_name / f'{img_name}.jpg'
            image_dest_path = images_dir_path / f'{img_name}.jpg'
            shutil.copy(str(image_src_path), str(image_dest_path))
            labels_path = labels_dir_path / f'{img_name}.txt'
            with labels_path.open('w') as f:
                _, bboxs = dataset.get_bboxs_in_frame(clip_name, i)
                bboxs = bbox_utils.xywh_corner_to_center(bboxs)
                bboxs = bbox_utils.normalize_xywh(bboxs, W, H)
                for bbox in list(bboxs):
                    f.write('0 ')
                    f.write(' '.join(str(x) for x in bbox.tolist()))
                    f.write('\n')
