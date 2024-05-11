import os
import numpy as np
import json
import cv2


# Use the same script for MOT16
DATA_PATH = 'datasets/myset_fire_person'
# DATA_PATH = 'datasets/myset_walk'
OUT_PATH = os.path.join(DATA_PATH, 'annotations')
pic_type = 'png'
SPLITS = ['train']  # --> split training data to train_half and val_half.
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True
CREATE_SPLITTED_DET = True


if __name__ == '__main__':

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:

        data_path = os.path.join(DATA_PATH, split) 
        out_path = os.path.join(OUT_PATH, '{}.json'.format(split))

        out = {
            'images': [], 
            'annotations': [],  # 跟踪器相关参数
            'videos': [],
            'categories': [
                {
                    'id': 1,
                    'name': 'pedestrain'
                }
            ]
        }
        img_path = os.path.join(data_path, 'img')
        ann_path = os.path.join(data_path, 'gt/gt.txt')
        images = os.listdir(img_path)
        num_images = len([image for image in images if pic_type in image])  # half and half

        for i in range(num_images):
            img = cv2.imread(os.path.join(data_path, 'img/{}.{}'.format(i + 1, pic_type)))
            height, width = img.shape[:2]
            image_info = {'file_name': 'img/{}.{}'.format(i + 1, pic_type),  # image name.
                            'id': i + 1,  # image number in the entire training set.
                            'frame_id': i + 1,  # image number in the video sequence, starting from 1.
                            'prev_image_id': i if i > 0 else -1,  # image number in the entire training set.
                            'next_image_id': i + 2 if i < num_images - 1 else -1,
                            'video_id': 1,
                            'height': height,
                            'width': width}
            out['images'].append(image_info)
        print('{} images'.format(num_images))

        if split != 'test':
            anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
            for i in range(anns.shape[0]):
                frame_id = int(anns[i][0])
                track_id = int(anns[i][1])
                cat_id = int(anns[i][7])
                category_id = 1
                ann = {'id': 1,
                        'category_id': category_id,
                        'image_id': frame_id,
                        'track_id': track_id,
                        'bbox': anns[i][2:6].tolist(),
                        'conf': float(anns[i][6]),
                        'iscrowd': 0,
                        'area': float(anns[i][4] * anns[i][5])}
                out['annotations'].append(ann)
            print('{} ann images'.format(int(anns[:, 0].max())))
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))