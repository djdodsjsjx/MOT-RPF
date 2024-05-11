import os
import json
import glob
import numpy as np

DATA_PATH = 'datasets/myset_fire_person/train/json'  # json文件夹
OUT_PATH = 'datasets/myset_fire_person/train/gt'

if __name__ == '__main__':
    labelme_json = glob.glob('{}/*.json'.format(DATA_PATH))
    if not os.path.exists(OUT_PATH): os.mkdir(OUT_PATH)
    output = []
    for num in range(len(labelme_json)):
        json_file = DATA_PATH + "/{}.json".format(num + 1)
        with open(json_file, 'r') as fp:
            data = json.load(fp)  # 加载json文件
            for shapes in data['shapes']:  # 一张图中有多个检测对象
                points = shapes['points']
                x1, y1 = points[0]
                x2, y2 = points[1]
                mix, miy = min(x1, x2), min(y1, y2)
                mxx, mxy = max(x1, x2), max(y1, y2)
                output.append([num + 1, 0, mix, miy, mxx - mix, mxy - miy, 1, 1, 1])
    np.savetxt(OUT_PATH + '/gt.txt', output, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')