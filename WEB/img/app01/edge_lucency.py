#filter=7   将边缘透明化    不是前面的50%开始计算    可以解决 空白地方的黑边问题
import cv2
import numpy as np


#边缘方向  一共八个方向  或者无方向
def direction_edge(i,j,mask_img,mask_threshold):
    if mask_img[i][j][0] < mask_threshold:
        if mask_img[i][j + 1][0] < mask_threshold:
            if mask_img[i + 1][j][0] < mask_threshold:
                if mask_img[i + 1][j + 1][0] > mask_threshold:
                    filter = 5
                else:
                    filter = 0
            else:
                if mask_img[i + 1][j + 1][0] < mask_threshold:
                    filter = 8
                else:
                    filter = 1
        else:
            if mask_img[i + 1][j][0] < mask_threshold:  # 黑白黑
                if mask_img[i + 1][j + 1][0] < mask_threshold:  # 黑
                    filter = 6
                else:
                    filter = 2
            else:
                filter = 0
    else:
        if mask_img[i][j + 1][0] < mask_threshold:  #
            if mask_img[i + 1][j][0] < mask_threshold:  # 白黑黑
                if mask_img[i + 1][j + 1][0] < mask_threshold:  # 黑
                    filter = 7
                else:
                    filter = 0
            else:  # 白黑白
                if mask_img[i + 1][j + 1][0] < mask_threshold:  # 黑
                    filter = 3
                else:
                    filter = 0
        else:  # 白白
            if mask_img[i + 1][j][0] < mask_threshold:
                if mask_img[i + 1][j + 1][0] < mask_threshold:  # 黑
                    filter = 4
                else:
                    filter = 0
            else:
                filter = 0
    return  filter

#透明滤边   让边缘逐渐透明
#sourse_img 四通道 且大部分为透明的图片
#mask_img   普通三通道图片 0-255 黑白图片
def lucency_edge(sourse_img,mask_img,mask_threshold=128,filter_size=3):      #mask_threshold 像素分类阈值  ?????????????????? 非常影响效果
    filter=0
    #不透明度
    transp=[0,255*0.1,255*0.2,255*0.3,255*0.4,255*0.5,255*0.6,255*0.7,255*0.8,255*0.9,255]
    if filter_size==3:
        filter3_list = [[[255, 255, 255], [255, 255, 255], [255, 255, 255]],
                        [[transp[7], transp[7], transp[7]], [transp[5], transp[5], transp[5]],
                         [transp[3], transp[3], transp[3]]],
                        [[transp[7], transp[5], transp[3]], [transp[7], transp[5], transp[3]],
                         [transp[7], transp[5], transp[3]]],
                        [[transp[3], transp[5], transp[7]], [transp[3], transp[5], transp[7]],
                         [transp[3], transp[5], transp[7]]],
                        [[transp[3], transp[3], transp[3]], [transp[5], transp[5], transp[5]],
                         [transp[7], transp[7], transp[7]]],
                        [[transp[7], transp[7], transp[5]], [transp[7], transp[5], transp[3]],
                         [transp[5], transp[3], transp[3]]],
                        [[transp[5], transp[3], transp[3]], [transp[7], transp[5], transp[3]],
                         [transp[7], transp[7], transp[5]]],
                        [[transp[3], transp[3], transp[5]], [transp[3], transp[5], transp[7]],
                         [transp[5], transp[7], transp[7]]],
                        [[transp[5], transp[7], transp[7]], [transp[3], transp[5], transp[7]],
                         [transp[3], transp[3], transp[5]]],]
        for i in range(sourse_img.shape[0] - 1):
            for j in range(sourse_img.shape[1] - 1):
                filter = direction_edge(i, j, mask_img, mask_threshold)
                if filter == 1:
                    sourse_img[i - 2:i + 1, j - 1:j + 2, 3] = filter3_list[filter]
                elif filter == 2:
                    sourse_img[i - 1:i + 2, j - 2:j + 1, 3] = filter3_list[filter]
                elif filter == 3:
                    sourse_img[i - 1:i + 2, j + 1:j + 4, 3] = filter3_list[filter]
                elif filter == 4:
                    sourse_img[i + 1:i + 4, j - 1:j + 2, 3] = filter3_list[filter]
                elif filter == 5:
                    sourse_img[i - 1:i + 2, j - 1:j + 2, 3] = filter3_list[filter]
                elif filter == 6:
                    sourse_img[i:i + 3, j - 1:j + 2, 3] = filter3_list[filter]
                elif filter == 7:
                    sourse_img[i:i + 3, j:j + 3, 3] = filter3_list[filter]
                elif filter == 8:
                    sourse_img[i - 1:i + 2, j:j + 3, 3] = filter3_list[filter]

    elif filter_size==5:
        filter5_list = [
            [[255, 255, 255, 255, 255], [255, 255, 255, 255, 255], [255, 255, 255, 255, 255], [255, 255, 255, 255, 255],
             [255, 255, 255, 255, 255]],
            [[transp[8], transp[8], transp[8], transp[8], transp[8]],
             [transp[6], transp[6], transp[6], transp[6], transp[6]],
             [transp[5], transp[5], transp[5], transp[5], transp[5]],
             [transp[4], transp[4], transp[4], transp[4], transp[4]],
             [transp[2], transp[2], transp[2], transp[2], transp[2]]],
            [[transp[8], transp[6], transp[5], transp[4], transp[2]],
             [transp[8], transp[6], transp[5], transp[4], transp[2]],
             [transp[8], transp[6], transp[5], transp[4], transp[2]],
             [transp[8], transp[6], transp[5], transp[4], transp[2]],
             [transp[8], transp[6], transp[5], transp[4], transp[2]]],
            [[transp[2], transp[4], transp[5], transp[6], transp[8]],
             [transp[2], transp[4], transp[5], transp[6], transp[8]],
             [transp[2], transp[4], transp[5], transp[6], transp[8]],
             [transp[2], transp[4], transp[5], transp[6], transp[8]],
             [transp[2], transp[4], transp[5], transp[6], transp[8]]],
            [[transp[2], transp[2], transp[2], transp[2], transp[2]],
             [transp[4], transp[4], transp[4], transp[4], transp[4]],
             [transp[5], transp[5], transp[5], transp[6], transp[5]],
             [transp[6], transp[6], transp[6], transp[6], transp[6]],
             [transp[8], transp[8], transp[8], transp[8], transp[8]]],
            [[transp[9], transp[8], transp[7], transp[6], transp[5]],
             [transp[8], transp[7], transp[6], transp[5], transp[4]],
             [transp[7], transp[6], transp[5], transp[4], transp[3]],
             [transp[6], transp[5], transp[4], transp[3], transp[2]],
             [transp[5], transp[4], transp[3], transp[2], transp[1]]],
            [[transp[5], transp[4], transp[3], transp[2], transp[1]],
             [transp[6], transp[5], transp[4], transp[3], transp[2]],
             [transp[7], transp[6], transp[5], transp[4], transp[3]],
             [transp[8], transp[7], transp[6], transp[5], transp[4]],
             [transp[9], transp[8], transp[7], transp[6], transp[5]]],
            [[transp[1], transp[2], transp[3], transp[4], transp[5]],
             [transp[2], transp[3], transp[4], transp[5], transp[6]],
             [transp[3], transp[4], transp[5], transp[6], transp[7]],
             [transp[4], transp[5], transp[6], transp[7], transp[8]],
             [transp[5], transp[6], transp[7], transp[8], transp[9]]],
            [[transp[5], transp[6], transp[7], transp[8], transp[9]],
             [transp[4], transp[5], transp[6], transp[7], transp[8]],
             [transp[3], transp[4], transp[5], transp[6], transp[7]],
             [transp[2], transp[3], transp[4], transp[5], transp[6]],
             [transp[1], transp[2], transp[3], transp[4], transp[5]]],
            ]
        for i in range(sourse_img.shape[0] - 1):
            for j in range(sourse_img.shape[1] - 1):
                filter = direction_edge(i, j, mask_img, mask_threshold)
                if filter == 1:
                    sourse_img[i - 4:i + 1, j - 2:j + 3, 3] = filter5_list[filter]
                elif filter == 2:
                    sourse_img[i - 2:i + 3, j - 4:j + 1, 3] = filter5_list[filter]
                elif filter == 3:
                    sourse_img[i - 2:i + 3, j + 1:j + 6, 3] = filter5_list[filter]
                elif filter == 4:
                    sourse_img[i + 1:i + 6, j - 2:j + 3, 3] = filter5_list[filter]
                elif filter == 5:
                    sourse_img[i - 3:i + 2, j - 3:j + 2, 3] = filter5_list[filter]
                elif filter == 6:
                    sourse_img[i:i + 5, j - 3:j + 2, 3] = filter5_list[filter]
                elif filter == 7:
                    sourse_img[i:i + 5, j:j + 5, 3] = filter5_list[filter]
                elif filter == 8:
                    sourse_img[i - 3:i + 2, j:j + 5, 3] = filter5_list[filter]
    elif filter_size==7:
        filter7_list=[[[255,255,255,255,255,255,255],[255,255,255,255,255,255,255],[255,255,255,255,255,255,255],[255,255,255,255,255,255,255],[255,255,255,255,255,255,255],[255,255,255,255,255,255,255],[255,255,255,255,255,255,255]],
                      [[transp[8], transp[8], transp[8], transp[8], transp[8], transp[8], transp[8]],
                       [transp[7], transp[7], transp[7], transp[7], transp[7], transp[7], transp[7]],
                       [transp[6], transp[6], transp[6], transp[6], transp[6], transp[6], transp[6]],
                       [transp[5], transp[5], transp[5], transp[5], transp[5], transp[5], transp[5]],
                       [transp[4], transp[4], transp[4], transp[4], transp[4], transp[4], transp[4]],
                       [transp[3], transp[3], transp[3], transp[3], transp[3], transp[3], transp[3]],
                       [transp[2], transp[2], transp[2], transp[2], transp[2], transp[2], transp[2]]],
                      [[transp[8], transp[7],transp[6], transp[5], transp[4], transp[3],  transp[2]],
                       [transp[8], transp[7],transp[6], transp[5], transp[4], transp[3],  transp[2]],
                       [transp[8], transp[7],transp[6], transp[5], transp[4], transp[3],  transp[2]],
                       [transp[8], transp[7],transp[6], transp[5], transp[4], transp[3],  transp[2]],
                       [transp[8], transp[7],transp[6], transp[5], transp[4], transp[3],  transp[2]],
                       [transp[8], transp[7],transp[6], transp[5], transp[4], transp[3],  transp[2]],
                       [transp[8], transp[7],transp[6], transp[5], transp[4], transp[3],  transp[2]]],
                      [[transp[2], transp[3], transp[4], transp[5], transp[6], transp[7], transp[8]],
                       [transp[2], transp[3], transp[4], transp[5], transp[6], transp[7], transp[8]],
                       [transp[2], transp[3], transp[4], transp[5], transp[6], transp[7], transp[8]],
                       [transp[2], transp[3], transp[4], transp[5], transp[6], transp[7], transp[8]],
                       [transp[2], transp[3], transp[4], transp[5], transp[6], transp[7], transp[8]],
                       [transp[2], transp[3], transp[4], transp[5], transp[6], transp[7], transp[8]],
                       [transp[2], transp[3], transp[4], transp[5], transp[6], transp[7], transp[8]]],
                      [[transp[2], transp[2], transp[2], transp[2], transp[2], transp[2], transp[2]],
                       [transp[3], transp[3], transp[3], transp[3], transp[3], transp[3], transp[3]],
                       [transp[4], transp[4], transp[4], transp[4], transp[4], transp[4], transp[4]],
                       [transp[5], transp[5], transp[5], transp[5], transp[5], transp[5], transp[5]],
                       [transp[6], transp[6], transp[6], transp[6], transp[6], transp[6], transp[6]],
                       [transp[7], transp[7], transp[7], transp[7], transp[7], transp[7], transp[7]],
                       [transp[8], transp[8], transp[8], transp[8], transp[8], transp[8], transp[8]]],
                      [[transp[10], transp[10], transp[9], transp[8], transp[7], transp[6], transp[5]],
                       [transp[10], transp[9], transp[8], transp[7], transp[6], transp[5], transp[4]],
                       [transp[9], transp[8], transp[7], transp[6], transp[5], transp[4], transp[3]],
                       [transp[8], transp[7], transp[6], transp[5], transp[4], transp[3], transp[2]],
                       [transp[7], transp[6], transp[5], transp[4], transp[3], transp[2], transp[1]],
                       [transp[6], transp[5], transp[4], transp[3], transp[2], transp[1], transp[0]],
                       [transp[5], transp[4], transp[3], transp[2], transp[1], transp[0], transp[0]]],
                      [[transp[5], transp[4], transp[3], transp[2], transp[1], transp[0], transp[0]],
                       [transp[6], transp[5], transp[4], transp[3], transp[2], transp[1], transp[0]],
                       [transp[7], transp[6], transp[5], transp[4], transp[3], transp[2], transp[1]],
                       [transp[8], transp[7], transp[6], transp[5], transp[4], transp[3], transp[2]],
                       [transp[9], transp[8], transp[7], transp[6], transp[5], transp[4], transp[3]],
                       [transp[10], transp[9], transp[8], transp[7], transp[6], transp[5], transp[4]],
                       [transp[10], transp[10], transp[9], transp[8], transp[7], transp[6], transp[5]]],
                      [[transp[0], transp[0], transp[1], transp[2], transp[3], transp[4], transp[5]],
                       [transp[0], transp[1], transp[2], transp[3], transp[4], transp[5], transp[6]],
                       [transp[1], transp[2], transp[3], transp[4], transp[5], transp[6], transp[7]],
                       [transp[2], transp[3], transp[4], transp[5], transp[6], transp[7], transp[8]],
                       [transp[3], transp[4], transp[5], transp[6], transp[7], transp[8], transp[9]],
                       [transp[4], transp[5], transp[6], transp[7], transp[8], transp[9], transp[10]],
                       [transp[5], transp[6], transp[7], transp[8], transp[9], transp[10], transp[10]]],
                      [[transp[5], transp[6], transp[7], transp[8], transp[9], transp[10], transp[10]],
                       [transp[4], transp[5], transp[6], transp[7], transp[8], transp[9], transp[10]],
                       [transp[3], transp[4], transp[5], transp[6], transp[7], transp[8], transp[9]],
                       [transp[2], transp[3], transp[4], transp[5], transp[6], transp[7], transp[8]],
                       [transp[1], transp[2], transp[3], transp[4], transp[5], transp[6], transp[7]],
                       [transp[0], transp[1], transp[2], transp[3], transp[4], transp[5], transp[6]],
                       [transp[0], transp[0], transp[1], transp[3], transp[3], transp[4], transp[5]]]]
        for i in range(sourse_img.shape[0]-1):
            for j in range(sourse_img.shape[1]-1):
                filter=direction_edge(i, j, mask_img, mask_threshold)
                if filter==1 :
                    sourse_img[i - 6:i + 1, j - 3:j + 4, 3] = filter7_list[filter]
                elif filter==2 :
                    sourse_img[i - 3:i + 4, j - 6:j + 1, 3] = filter7_list[filter]
                elif filter==3 :
                    sourse_img[i - 3:i + 4, j + 1:j + 8, 3] = filter7_list[filter]
                elif filter==4 :
                    sourse_img[i + 1:i + 8, j - 3:j + 4, 3] = filter7_list[filter]
                elif filter==5 :
                    sourse_img[i - 5:i + 2, j - 5:j + 2, 3] = filter7_list[filter]
                elif filter==6 :
                    sourse_img[i :i + 7, j - 5:j + 2, 3] = filter7_list[filter]
                elif filter==7 :
                    sourse_img[i :i + 7, j :j + 7, 3] = filter7_list[filter]
                elif filter==8:
                    sourse_img[i - 5:i + 2, j :j + 7, 3] = filter7_list[filter]
    return sourse_img

def paste_ROI_to_image(image_in, ROI):
    image = image_in.copy()
    y1, x1, y2, x2 = 0,0,image_in.shape[0]-1,image_in.shape[1]-1
    ROI = cv2.resize(ROI,(x2- x1 + 1, y2 - y1 + 1))   # cv.resize(src, dsize=(width, height))  resize成适当的大小
    image = image.astype(np.float)
    ROI = ROI.astype(np.float)
    # alpha通道
    alpha_image = image[y1:y2 + 1, x1:x2 + 1, 3] / 255.0    # 将alpha通道值取值范围由0-255转换到0-1
    alpha_ROI = ROI[:, :, 3] / 255.0
    alpha = 1 - (1 - alpha_image) * (1 - alpha_ROI)        #计算合成后的图像的透明度   剩余透明度（1 - alpha_image） 和（1 - alpha_ROI）混合后，得到的图像透明度。
    # BGR通道
    for i in range(3):
        image[y1:y2 + 1, x1:x2 + 1, i] = (image[y1:y2 + 1, x1:x2 + 1, i] * alpha_image * (1 - alpha_ROI) + ROI[:, :,i] * alpha_ROI) / alpha
        #按照透明度来分配image和ROI的RGB混合比例。
    image[y1:y2 + 1, x1:x2 + 1, 3] = alpha * 255      #合并alpha通道和RGB通道
    image = image.astype(np.uint8)
    return image


if __name__ == '__main__':
    img = cv2.imread("test/image3.png",cv2.IMREAD_UNCHANGED)
    mask=  cv2.imread("test/mask3.jpg",cv2.IMREAD_UNCHANGED)        #大部分是255    会有 0 1 2 3 等小像素点
    # print (img.shape[0])     #形状
    # print (img.dtype)      #类型
    lucency_edge_img=lucency_edge(img,mask,mask_threshold=128,filter_size=3)   #mask_threshold 像素分类阈值    #得到优化好边缘的透明图片

    image= 255 * np.ones(img.shape, img.dtype)     #生成四通道白色背景图
    print(image.shape)
    print(lucency_edge_img.shape)
    image_R = paste_ROI_to_image(image, lucency_edge_img)   #将两张PNG按照透明度合成到一张图上
    # print(image_R.shape)

    # cv2.imshow("img_BGRA",image_R)
    # cv2.waitKey(0)
    cv2.imwrite('test/new_outlucency3.jpg', image_R)