from torchvision import transforms
from run_code import run_style_transfer
from load_img import load_img, show_img
from torch.autograd import Variable
import PIL.Image as Image
import cv2
import numpy

def trans_style(img_cv=0,image_dir="",style_dir="",style=2):
    if len(image_dir)>2:
        content_img = load_img(image_dir)
    else:
        content_img = Image.fromarray(img_cv)
        content_img = content_img.resize((512, 512))
        content_img = transforms.ToTensor()(content_img)
        content_img = content_img.unsqueeze(0)
    content_img = Variable(content_img).cuda()
    if len(style_dir)>1:
        style_img = load_img(style_dir)
    else:
        if style==1:
            style_img = load_img('static/style/style1.jpg')
        elif style==2:
            style_img = load_img('static/style/style2.jpg')
        elif style==3:
            style_img = load_img('static/style/style3.jpg')
    style_img = Variable(style_img).cuda()
    input_img = content_img.clone()
    out = run_style_transfer(content_img, style_img, input_img, num_epoches=50)
    out_style = transforms.ToPILImage()(out.cpu().squeeze(0))          #转化为PILImage并显示   squeeze从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    out_style = cv2.cvtColor(numpy.asarray(out_style), cv2.COLOR_RGB2BGR)
    return  out_style

# if __name__ == '__main__':            #1简约 2古典 3水墨
#     image = cv2.imread("test/32e95dc5181a5d42d5ba5bc6dc2f.jpg")
#     out_style=trans_style(img_cv=image,style_dir="style/cut.jpg")       # 不填style_dir 为 style=1,2,3    填完路径为对应路径的jpg图片
#     #out_style=trans_style(img_cv=image,style=2)       # 不填style_dir 为 style=1,2,3    填完路径为对应路径的jpg图片
#     #out_style= trans_style(image_dir="test/32e95dc5181a5d42d5ba5bc6dc2f.jpg")
#     cv2.namedWindow("out_style", cv2.WINDOW_NORMAL)
#     cv2.imshow("out_style", out_style)
#     cv2.waitKey(0)

