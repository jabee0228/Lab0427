

import torch
import cv2
import numpy as np
import matplotlib.cm as cm

from src.utils.plotting import make_matching_figure
from src.loftr import LoFTR, default_cfg
def loftrGenerate(img0_pth, img1_pth):
    image_pair = [img1_pth, img0_pth]
    matcher = LoFTR(config=default_cfg)
    if image_type == 'indoor':
      matcher.load_state_dict(torch.load("weights/indoor_ds.ckpt")['state_dict'])
    elif image_type == 'outdoor':
      matcher.load_state_dict(torch.load("weights/outdoor_ds.ckpt")['state_dict'])
    else:
      raise ValueError("Wrong image_type is given.")
    matcher = matcher.eval().cuda()

    img0_raw = cv2.imread(image_pair[0], cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(image_pair[1], cv2.IMREAD_GRAYSCALE)
    img0_raw = cv2.resize(img0_raw, (640, 480))
    img1_raw = cv2.resize(img1_raw, (640, 480))

    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
    batch = {'image0': img0, 'image1': img1}

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()

    if (len(mkpts0))

    # Draw
    color = cm.jet(mconf, alpha=0.7)
    text = [
        'LoFTR',
        'Matches: {}'.format(len(mkpts0)),
    ]
    fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, mkpts0, mkpts1, text)

    H, status = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)

    # find the coordinate of the pos in img2 corespond to img1 blocked area

    left_up = np.array([[260], [277], [1]])
    left_down = np.array([[260], [360], [1]])
    right_up = np.array([[480], [277], [1]])
    right_down = np.array([[480], [360], [1]])

    # 使用矩阵乘法进行变换
    transformed_left_up = np.dot(H, left_up)
    transformed_left_down = np.dot(H, left_down)
    transformed_right_up = np.dot(H, right_up)
    transformed_right_down = np.dot(H, right_down)

    # 将齐次坐标转换回笛卡尔坐标
    transformed_left_up = (transformed_left_up / transformed_left_up[-1])[:-1]
    transformed_left_down = (transformed_left_down / transformed_left_down[-1])[:-1]
    transformed_right_up = (transformed_right_up / transformed_right_up[-1])[:-1]
    transformed_right_down = (transformed_right_down / transformed_right_down[-1])[:-1]

    # 打印转换后的坐标
    print("Transformed left_up:", transformed_left_up.T)
    print("Transformed left_down:", transformed_left_down.T)
    print("Transformed right_up:", transformed_right_up.T)
    print("Transformed right_down:", transformed_right_down.T)


    x1_src = int(transformed_left_up[0])
    y1_src = int(transformed_left_up[1])
    src_top_left = (x1_src, y1_src)
    x2_src = int(transformed_right_down[0])
    y2_src = int(transformed_right_down[1])
    src_bottom_right = (x2_src, y2_src)
    x1_dst = int(left_up[0])
    y1_dst = int(left_up[1])
    dst_top_left = (x1_dst, y1_dst)


    width = src_bottom_right[0] - src_top_left[0]
    height = src_bottom_right[1] - src_top_left[1]
    height = int(height)
    width = int(width)
    result_img0 = img0_raw.copy()
    target_image2 = img1_raw
    for y in range(height):
        for x in range(width):
            src_x, src_y = src_top_left[0] + x, src_top_left[1] + y
            dst_x, dst_y = dst_top_left[0] + x, dst_top_left[1] + y

            result_img0[dst_y, dst_x] = target_image2[src_y, src_x]

    filename = "transformed_image.png"  # Replace with your preferred name and format

    # Save the image using OpenCV's imwrite function
    cv2.imwrite(filename, result_img0)