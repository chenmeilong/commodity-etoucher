import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/yolov3_master')
from yolo_models import Darknet
from utils.datasets import LoadImages, LoadStreams
from utils.utils import load_classes, non_max_suppression, scale_coords, plot_one_box, load_classes

cfg = 'yolov3_master/cfg/yolov3-tiny.cfg'
data = 'yolov3_master/data/bag.data'
weights = 'yolov3_master/weights/best.pt'
img_size = 416
conf_thres = 0.25  # conf_thres, nms_thres: 目标检测置信度，非极大抑制阈值
nms_thres = 0.25
device = "cuda:0"

# 实例化模型
model = Darknet(cfg, img_size)  # 实例化模型   #传入后变成（416,416）
# 加载权重值
if weights.endswith('.pt'):  # pytorch format
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
model.to(device).eval()  # 模型放在GPU或cpu上  开始eval


def detect(source):  #
    dataset = LoadImages(source, img_size=img_size)  # 加载测试数据集
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # 增加一个维度
        pred = model(img)[0]  # #模型输出结果 torch.Size([1, 1755, 6])
        pred = non_max_suppression(pred, conf_thres, nms_thres)  # 返回如下  将多个预测框过滤、合并
        # [tensor([[135.79984, 228.46407, 219.11792, 377.94257,   0.91529,   1.00000,   0.00000]], device='cuda:0')]  #
        # 位置四个角坐标 ，准确率，非极大抑制值，种类

        for i, det in enumerate(pred):  # 侦测每张图片    det即是上面的包框tensor    只循环一遍
            p, s, im0 = path, '', im0s  # im0s 输入没有resize的图片
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()  # 将box 计算到原图上
                # 根据box 截取图片
                numpy_det = det.cpu().detach().numpy()
                x0 = int(numpy_det[0][0])
                y0 = int(numpy_det[0][1])
                x1 = int(numpy_det[0][2])
                y1 = int(numpy_det[0][3])
                crop_img = im0[y0:y1, x0:x1]  # 剪切图片   ##暂时不考虑 一张图多个包的情况，后面再考虑
            else:
                crop_img = im0  # yolo3没有检测到框图  返回原图
    return crop_img


if __name__ == '__main__':
    crop_img = detect(source='yolov3_master/data/samples/000001.jpg')
