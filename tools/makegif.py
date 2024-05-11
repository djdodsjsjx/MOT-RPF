import os
import sys
sys.path.insert(0, '/home/zhy/Track/ocsort')
import numpy as np
from tools import gif
import matplotlib.pyplot as plt
import matplotlib

if __name__ == '__main__':
    fs = []
    @gif.frame
    def plott(bind):  # 没有历史轨迹
        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)  # 轨迹图初始化
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 3000)
        for i in range(bind+1):
            pv = distances[i - 1]
            v = distances[i]
            l = matplotlib.lines.Line2D([i, i + 1], [pv, v], color="blue")
            ax.add_line(l)
    
    distances = np.loadtxt("YOLOX_outputs/output.txt", delimiter=',')

    for i, v in enumerate(distances):
        f = plott(i)
        fs.append(f)
    
    gif.save(fs, "hh.gif", duration=50)