from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import shutil
import os
import glob
from matplotlib.font_manager import FontProperties
################Change this section
MainDir = "datasets/icvs"

##Put your result file here
GROUND_TRUE = "/GroundTruth.txt"
OAB_RESULT = "/results/OAB_result.txt"
SOAB_RESULT = "/results/SOAB_result.txt"
ASE_RESULT = "/results/ASE_result.txt"
CNN_RGBBD_RESULT = "/results/CNN_V1_result.txt"
CNN_RGB_BD_RESULT = "/results/CNN_V2_result.txt"
CNN_RGB_RESULT = "/results/CNN_V3_result.txt"
OCSORT_RESULT = "/results/ocsort.txt"
BAMSORT_RESULT = "/results/bamsort.txt"
FILES      = [GROUND_TRUE, OAB_RESULT, SOAB_RESULT, ASE_RESULT, CNN_RGBBD_RESULT, CNN_RGB_BD_RESULT, CNN_RGB_RESULT, OCSORT_RESULT, BAMSORT_RESULT]
NAMES      = [             'OAB',      'SOAB',      'ASE',       'CNN$\_$v1',      'CNN$\_$v2',       'CNN$\_$v3',      'OC-SORT',     'MOT-RPF(Ours)']
LINESTYLES = [             '-',         '-',        ':',              '-',               '-',            '-',       '-',         '-']
COLOURS    = [             'b',         'g',        'b',              'm',               'c',            'k',        'y',          'r']
COLOURS2    = [             'b',         'c',        'g',              'm',               'c',            'k',        'y',          'r']
################
################ Do not change the code below, unless you have to.


def cal_err(filedir):
    all_points = []
    for fnames in FILES:
        #print fnames
        points = []
        for line in open(filedir+fnames):
            #print line
            line = line.replace(",", " ")
            line = line.replace(";", " ")
            line = line.replace("\t", " ")
            l = line.split()
            #print l
            try:
                x, y, w, d  = int(l[1]), int(l[2]), int(l[3]), int(l[4])
            except (IndexError):
                try:
                    x, y, w, d  = int(l[0]), int(l[1]), int(l[2]), int(l[3])
                except (ValueError):
                    x, y, w, d = 0, 0, 0, 0
            except (ValueError):
                x, y, w, d = 0, 0, 0, 0
            points.append([x+w/2.0,y+d/2.0])
        all_points.append(points)
    all_points = np.array(all_points)
    return all_points

def eval_plt(all_points, save_path, save_file_pix50, acc_file):

    successful_rates = np.zeros([len(FILES)-1, 150])
    for i in range(len(NAMES)):
        error = all_points[i+1,:,:] - all_points[0,:,:]
        error = np.square(error)
        error = [j[0]+j[1] for j in error]
        error = np.sqrt(error)
        #error = np.cumsum(error)
        for j in range(150):
            successful_rates[i,j]  = error[ np.where( error < j )].size/float(error.size)

    # 准确率数值保存
    pixs = np.arange(150)
    pixs_acc = np.concatenate((pixs.reshape(1, -1), successful_rates), axis=0).T
    np.savetxt(acc_file, pixs_acc, fmt="%d %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f")


    ## 柱形图  各模型在50误差内的精确度
    plt.figure(figsize=(12,6))
    font = {'family': 'Times New Roman','size'   : 15}
    matplotlib.rc('font', **font)
    x=np.arange(len(NAMES))
    barlist = plt.bar(x, successful_rates[:,50], align='center', color=COLOURS2)
    plt.xticks(x, NAMES, fontsize=14)
    for i, label in enumerate(plt.gca().get_xticklabels()):
        if NAMES[i] == 'MOT-RPF(Ours)':
            label.set_fontweight('bold')
    plt.ylabel('Precision', fontsize=28)
    plt.title('Location Error Threshold = 50 pixels', fontsize=28)
    plt.gcf().subplots_adjust(left=0.11)
    plt.gcf().subplots_adjust(bottom=0.09)
    plt.gcf().subplots_adjust(right=0.99)
    plt.gcf().subplots_adjust(top=0.91)
    # plt.show()
    plt.savefig(save_file_pix50, format='pdf')

    # 折线图
    plt.clf()
    plt.figure(figsize=(7,6))
    font = {'family': 'Times New Roman', 'size'   : 24}
    matplotlib.rc('font', **font)
    plt.xlabel('Location Error Threshold (pixel)',fontsize=28)
    plt.ylabel('Precision', fontsize=34)
    plt.title('')
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(right=0.95)
    plt.gcf().subplots_adjust(top=0.95)

    lines = []
    t = np.arange(0, successful_rates.shape[1], 1)
    for i in range(len(NAMES)):
        line, = plt.plot(t, successful_rates[i], label=NAMES[i], linestyle=LINESTYLES[i], linewidth=3, c=COLOURS[i])
        lines.append(line)
    bold_font = FontProperties()
    bold_font.set_weight('bold')
    legend = plt.legend(loc=4, prop={'size': 24})
    # a = legend.get_texts()
    plt.setp(legend.get_texts()[7], fontproperties=bold_font)
    plt.savefig(save_path, format='pdf')

def pt_res_to_datasets(folder_a, folder_b):
    txt_files = [f for f in os.listdir(folder_a) if f.endswith('.txt')]

    for txt_file in txt_files:
        corresponding_folder = os.path.join(folder_b, txt_file[:-4]) 
        if os.path.isdir(corresponding_folder):
            original_file_path = os.path.join(folder_a, txt_file)
            target_file_path = os.path.join(corresponding_folder, "results", "bamsort.txt")
            shutil.copy2(original_file_path, target_file_path)
            print(f"Copied {txt_file} to {corresponding_folder}")
        else:
            print(f"No corresponding folder found for {txt_file}")

def eval_pt(res_path, save_path):
    pt_res_to_datasets(res_path, "datasets/icvs")
    txt_files = glob.glob(os.path.join(res_path, '*.txt'))
    all_points = None
    for DIR in txt_files:
        file_name = os.path.basename(DIR)[:-4]
        filedir = os.path.join(MainDir, file_name)
        save_file = os.path.join(save_path, f"image-{file_name}.pdf")
        save_file_pix50 = os.path.join(save_path, f"image-{file_name}_pix50.pdf")
        cur_points = cal_err(filedir)
        if all_points is None:
            all_points = cur_points
        else:
            all_points = np.concatenate((all_points, cur_points), axis=1)
        acc_file = os.path.join(save_path, f"{file_name}.txt")
        eval_plt(cur_points, save_file, save_file_pix50, acc_file)
    save_file = os.path.join(save_path, "image-all_points.pdf")
    save_file_pix50 = os.path.join(save_path, f"image-all_points_pix50.pdf")
    acc_file = os.path.join(save_path, "all_points_acc.txt")
    eval_plt(all_points, save_file, save_file_pix50, acc_file)

if __name__ == "__main__":

    res_path = "evaldata/trackers/icvs/test/bamsort/data"
    save_path = "evaldata/trackers/icvs/test/bamsort/res_pdf"
    eval_pt(res_path, save_path)

