import torch
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
import cv2
import edge_lucency

def fcn_pre(img_cv=0,filter_size=0,image_dir="",back_dir=""):  #filter_size0,3,5,7  cv2图片 原图，背景图
    if len(image_dir)>2:
        img_cv = cv2.imread(image_dir)
    img = cv2.resize(img_cv, (640, 640))   # [160, 160, 3]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    img= transform(img)
    img=img.unsqueeze(0)     ##torch.Size([1, 3, 160, 160])
    if torch.cuda.is_available():
        im = Variable(img.cuda())

    fcn_model =torch.load('static/checkpoints/640fcn_model_250_0.022_0.071.pkl')
    fcn_model.eval()
    pre=fcn_model(im)
    numpy_array =pre.cpu().detach().numpy()
    numpy_array=np.squeeze(numpy_array,axis =0)
    output_np = np.argmin(numpy_array, axis=0)      #输出
    #print (output_np .shape)     #形状 160*160
    img1=output_np.astype(np.uint8)
    # print(img1.dtype)
    kernel = np.ones((3,3),np.uint8)
    img1 = cv2.morphologyEx(img1,cv2.MORPH_CLOSE,kernel)   #膨胀腐蚀

    img1.dtype = "int8"
    img1 = abs(img1 - 1)  # 图片颜色翻转   颜色fanzhaun：  0-1  1-0
    img1.dtype = "uint8"
    img1 = img1 * 255
    # array = img1.tolist()
    # print(array)

    img2 =img_cv
    # print (img2.shape)     #形状
    img1 = cv2.resize(img1,(img2.shape[1],img2.shape[0]),cv2.INTER_CUBIC)  #扩大推荐使用 “cv2.INTER_CUBIC”
    #print(img1.shape)
    image = np.expand_dims(img1, axis=2)
    img1 = np.concatenate((image, image, image), axis=-1)
    # print(img1.shape)
    # print(img2.shape)
    img1 = cv2.medianBlur(img1, 3)  # 中值滤波函数  效果好

    r_channl = cv2.split(img2)[2]  # R通道
    g_channl = cv2.split(img2)[1]  # g通道
    b_channl = cv2.split(img2)[0]  # b通道
    a_channl = cv2.split(255 * np.ones(img2.shape, img2.dtype))[0]  # a通道
    # print(r_channl.shape, "r_channl")
    # print(a_channl.shape, "a_channl")
    merged = cv2.merge((b_channl, g_channl, r_channl, a_channl))  # 前面分离出来的三个通道

    for i in range(img2.shape[0]):    ##黑底换白底
        for j in range(img2.shape[1]):
            if img1[i][j][0]==0:
                merged[i][j][0:3] = 0
                merged[i][j][3] = 0

    cv2.imwrite('mask.jpg',  img1)
    lucency_img=merged
    mask = img1
    if filter_size!=0:
        lucency_img = edge_lucency.lucency_edge(lucency_img, mask,
                                                 mask_threshold=128,filter_size=filter_size)  # mask_threshold 像素分类阈值    #得到优化好边缘的透明图片  filter_size=3 3，5,7三种可选
    if  len(back_dir)>1:
        back= cv2.imread(back_dir)
        r_channl = cv2.split(back)[2]  # R通道
        g_channl = cv2.split(back)[1]  # g通道
        b_channl = cv2.split(back)[0]  # b通道
        a_channl = cv2.split(255 * np.ones(back.shape, back.dtype))[0]  # a通道
        back = cv2.merge((b_channl, g_channl, r_channl, a_channl))  # 前面分离出来的三个通道
        back=cv2.resize( back, (lucency_img.shape[0],lucency_img.shape[1]))
    else:
        back = 255 * np.ones(merged.shape, merged.dtype)  # 生成四通道白色背景图   back_dir
    # print(image.shape)
    # print(lucency_edge_img.shape)
    image_R = edge_lucency.paste_ROI_to_image(back, lucency_img)  # 将两张PNG按照透明度合成到一张图上  返回四通道png图片
    # print(image_R.shape)
    r_channl = cv2.split(image_R)[2]  # R通道
    g_channl = cv2.split(image_R)[1]  # g通道
    b_channl = cv2.split(image_R)[0]  # b通道
    return_img=cv2.merge((b_channl, g_channl, r_channl))
    return return_img

# if __name__ == '__main__':
# 
#     image = cv2.imread("test/2.jpg")
#     fcn_img=fcn_pre(img_cv=image,filter_size=3,back_dir="test/17.jpg")   #背景图片 ，前景图可加可不加
#     #fcn_img = fcn_pre(image_dir="test/32e95dc5181a5d42d5ba5bc6dc2f.jpg")
#     cv2.namedWindow("fcn_img", cv2.WINDOW_NORMAL)
#     cv2.imshow("fcn_img", fcn_img)
#     cv2.waitKey(0)
# 
#     # cv2.imwrite('fcn.png', fcn_img)
#     # cv2.imwrite('fcn.jpg', fcn_img)
