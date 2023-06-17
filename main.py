#图形化窗口加入 前景图替换 ，按钮点击变成多选按钮
#前景图必须为PNG透明图像

import tkinter.filedialog
from tkinter import *
import os
import cv2
import time
import use_yolo3
import use_fcn
import use_style

addr_source_img=""
addr_back_img=""
addr_fore_img=""
addr_style_img=""

def ensure_1():
    global addr_source_img
    addr_source_img =tkinter.filedialog.askopenfilename(filetypes=[("JPG", ".jpg")])  #返回文件名
    v0.set(addr_source_img)
    if (len(addr_source_img)<=1):
        return
    print (addr_source_img)

def ensure_2():
    global addr_back_img
    addr_back_img =tkinter.filedialog.askopenfilename(filetypes=[("JPG", ".jpg")])  #返回文件名
    v1.set(addr_back_img)
    if (len(addr_back_img)<=1):
        return
    print (addr_back_img)

def ensure_3():
    global addr_fore_img
    addr_fore_img =tkinter.filedialog.askopenfilename(filetypes=[("PNG", ".PNG")])  #返回文件名
    v2.set(addr_fore_img)
    if (len(addr_fore_img)<=1):
        return
    print (addr_fore_img)


def ensure_4():
    global addr_style_img
    addr_style_img =tkinter.filedialog.askopenfilename(filetypes=[("JPG", ".jpg")])  #返回文件名
    v3.set(addr_style_img)
    if (len(addr_style_img)<=1):
        return
    print (addr_style_img)

def add_foreground(back_img,foreground_dir)  :
    foreground = cv2.imread(foreground_dir, cv2.IMREAD_UNCHANGED)
    foreground = cv2.resize(foreground, (back_img.shape[1], back_img.shape[0]),cv2.INTER_AREA)  # 缩小推荐使用 "cv2.INTER_AREA";  扩大推荐使用 “cv2.INTER_CUBIC”
    rows, cols, channels = foreground.shape
    for i in range(rows):
        for j in range(cols):
            if not (foreground[i, j][3] == 0):  # 透明的意思
                back_img[i, j, 0:3] = foreground[i, j, 0:3]
    return back_img

# var_white = IntVar()
# var_back = IntVar()
# var_fore = IntVar()
# var_style = IntVar()
def draw_picture():
    global addr_source_img
    global addr_back_img
    global addr_fore_img
    global addr_style_img
    global out_image
    if (len( addr_source_img))>1:
        out_image = use_yolo3.detect(source=addr_source_img)
        if var_white.get()==1 and var_back.get()!=1:   ##白底图生成
            out_image = use_fcn.fcn_pre(img_cv=out_image, filter_size=0)  # filter_size0,3,5,7
        elif var_white.get()!=1 and var_back.get()==1:   ##白底图生成
            if (len(addr_back_img))>1:
                out_image = use_fcn.fcn_pre(img_cv=out_image, filter_size=0,back_dir=addr_back_img)  # filter_size0,3,5,7
        if var_fore.get()==1:
            if len(addr_fore_img)>1:
                out_image=add_foreground(out_image, addr_fore_img)   #加入前景图
        if var_style .get() == 1:
            if (len(addr_style_img))>1:
                out_image =use_style.trans_style(img_cv=out_image,style_dir=addr_style_img)
        cv2.namedWindow("out_image", cv2.WINDOW_KEEPRATIO)
        # cv2.resizeWindow("out_image", out_image.shape[0],out_image.shape[1]);
        cv2.imshow("out_image", out_image)
        cv2.waitKey(0)

def save_picture():
    global addr_source_img
    if len(addr_source_img)>1:
        b = addr_source_img.split("/")
        now_parent = os.path.dirname(addr_source_img)  # 父级目录
        now_parent = os.path.dirname(now_parent)  # 父级目录
        new_addr_source_img = now_parent + "/new" + b[-2]
        isExists = os.path.exists(new_addr_source_img)  # 没有文件夹时新建一个
        if not isExists:
            os.makedirs(new_addr_source_img)
        time_str = str(time.localtime()[0]) + str(time.localtime()[1]) + str(time.localtime()[2]) + str(
            time.localtime()[3]) + str(time.localtime()[4]) + str(time.localtime()[5])
        new_addr_jpg = new_addr_source_img + "/" + b[-1][0:-4]+"_"+time_str+".jpg"
        cv2.imwrite(new_addr_jpg, out_image)
        print("保存目录为:",new_addr_jpg)


root = Tk()
root.title("基于FCN语义分割模型商品图像提取")
root.iconbitmap('hrbu.ico')

panel = Label(root,justify=RIGHT)                      # 初始化
panel.pack(padx=10, pady=10,expand=True,side=RIGHT)

fm1 = Frame(root.master)
fm1.pack(side=RIGHT, padx=10, fill=BOTH, expand=YES)
v0 =StringVar()        #位置不能动 地址缓存
v1 =StringVar()
v2 =StringVar()
v3 =StringVar()
ensure_button1 = Button(fm1,text="选择原图",command=ensure_1,width=10).grid(row=0, column=0, pady=5,padx=5)               #确定文件
file_output1 = Entry(fm1, width=60, textvariable=v0, state="readonly").grid(row=0, column=1,columnspan=4, pady=5,padx=5)  #这是路径小框的位置

ensure_button2 = Button(fm1,text="选择背景图",command=ensure_2,width=10).grid(row=1, column=0, pady=5,padx=5)               #确定文件
file_output2 = Entry(fm1, width=60, textvariable=v1, state="readonly").grid(row=1, column=1,columnspan=4, pady=5,padx=5)  #这是路径小框的位置

ensure_button3 = Button(fm1,text="选择前景图",command=ensure_3,width=10).grid(row=2, column=0, pady=5,padx=5)               #确定文件
file_output3 = Entry(fm1, width=60, textvariable=v2, state="readonly").grid(row=2, column=1,columnspan=4, pady=5,padx=5)  #这是路径小框的位置

ensure_button4 = Button(fm1,text="选择风格图",command=ensure_4,width=10).grid(row=3, column=0, pady=5,padx=5)               #确定文件
file_output4 = Entry(fm1, width=60, textvariable=v3, state="readonly").grid(row=3, column=1,columnspan=4, pady=5,padx=5)  #这是路径小框的位置

var_white = IntVar()
var_back = IntVar()
var_fore = IntVar()
var_style = IntVar()

Checkbutton(fm1, text="换白底", variable=var_white).grid(row=4, column=0, pady=3,padx=5)
Checkbutton(fm1, text="背景替换", variable=var_back).grid(row=4, column=1, pady=3,padx=5)
Checkbutton(fm1, text="增加前景", variable=var_fore).grid(row=4, column=2, pady=3,padx=5)
Checkbutton(fm1, text="风格生成", variable=var_style).grid(row=4, column=3, pady=3,padx=5)

# Radiobutton(fm1, text="风格生成", variable=pointx, value=2).grid(row=3, column=2, pady=3,padx=5)

save_button = Button(fm1,text="开始生成",command=draw_picture,width=15).grid(row=5, column=1, pady=5,padx=5)              #确定计算
save_button = Button(fm1,text="保存图片",command=save_picture,width=15).grid(row=5, column=2, pady=5,padx=5)              #确定计算

root.mainloop()