from django.shortcuts import render,HttpResponse
from django.http import FileResponse
import os, uuid
import use_yolo3
import use_fcn
import cv2
import use_style
# Create your views here.
def upload(request):
    return render(request,'upload.html')
import json


def upload_img(request):
    nid=str(uuid.uuid4()) # 防止文件名称相同
    ret={"status":True,"data":None,"message":None}
    obj = request.FILES.get("k3")
    sty = request.POST.get("sty")
    print(sty)
    global source_img
    source_img=obj.name
    upfile_path=os.path.join("static/upload_img",nid+obj.name)
    f=open(upfile_path,'wb')
    for line in obj.chunks():    #把选择的图片上传文件到服务器端
        f.write(line)
    f.close()
    global fcn_path
    fcn_path = os.path.join("static/fcn_img/", nid + '.jpg')  ###抠完图要保存在本地服务器

    # print(os.getcwd())
    if sty=='0':
        crop_img = use_yolo3.detect(source=upfile_path)
        out_img = use_fcn.fcn_pre(img_cv=crop_img)
    else:
        crop_img = use_yolo3.detect(source=upfile_path)
        fcn_img = use_fcn.fcn_pre(img_cv=crop_img)
        out_img = use_style.trans_style(img_cv=fcn_img, style=int(sty))
    cv2.imwrite(fcn_path,out_img)                              ######扣完的图片保存在服务器

    ret["data"]=fcn_path   #将文件所在的路径返回给前端
    # ret["data"]=upfile_path   #将文件所在的路径返回给前端
    return HttpResponse(json.dumps(ret))

def download(request):
    file = open(fcn_path, 'rb')
    response = FileResponse(file)
    response['Content-Type']='application/octet-stream'
    response['Content-Disposition']='attachment;filename='+source_img
    return response




