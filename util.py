import cv2
import numpy as np
#import torch
from sklearn.cluster import DBSCAN
import os
import shutil

def addWhiteBlock(x1,y1,x2,y2,orginalFile,fileName):
    name = orginalFile + ".jpg"
    image = cv2.imread(name)

    rectangle = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)

    fileName = fileName + ".jpg"
    cv2.imwrite(fileName, rectangle)



def makeBlackImg(x1,y1,x2,y2,fileName):
    width = 640
    height = 480

    black_image = np.zeros((height, width, 3), dtype='uint8')

    black_image[:] = 0

    cv_image = black_image

    rectangle = cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 255, 255), -1)
    fileName = "masks/"+fileName + ".jpg"
    cv2.imwrite(fileName, rectangle)


def block_region(mkpts):
    # 应用 DBSCAN 算法
    dbscan = DBSCAN(eps=30, min_samples=10)  # 这里的参数可以根据你的实际数据调整
    dbscan.fit(mkpts)

    # 获取聚类结果
    cluster_labels = dbscan.labels_

    # 查看聚类标签
    print("Cluster labels:", cluster_labels)

    # 找出最大的簇
    labels, counts = np.unique(cluster_labels[cluster_labels >= 0], return_counts=True)
    largest_cluster_label = labels[np.argmax(counts)]
    largest_cluster_points = mkpts[cluster_labels == largest_cluster_label]

    # 计算矩形边界
    x_min, y_min = np.min(largest_cluster_points, axis=0)
    x_max, y_max = np.max(largest_cluster_points, axis=0)

    # 确保坐标是整数
    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

    return x_min, y_min, x_max, y_max

def returnFilename(path):
    if os.path.isabs(path):
        # 获取文件名
        filename = os.path.basename(path)
        return filename
def copy_file(source_path, destination_path):
    try:
        # 复制文件
        shutil.copy(source_path, destination_path)

    except Exception as e:
        print("copy error", e)
