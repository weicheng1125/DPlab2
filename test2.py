
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def show_image(title, image):
    '''
    调用matplotlib显示RGB图片
    :param title: 图像标题
    :param image: 图像的数据
    :return:
    '''
    # plt.figure("show_image")
    # print(image.dtype)
    plt.imshow(image)
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title(title)  # 图像题目
    plt.show()
 
def cv_show_image(title, image):
    '''
    调用OpenCV显示RGB图片
    :param title: 图像标题
    :param image: 输入RGB图像
    :return:
    '''
    channels=image.shape[-1]
    if channels==3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 将BGR转为RGB
    cv2.imshow(title,image)
    cv2.waitKey(0)

def read_nonDir(directory_name):
    for filename in os.listdir(r"./"+directory_name):
        #print(directory_name + "/" + filename)
        #read_non(directory_name + "/" +filename)
        read_non(directory_name,filename)

def read_non(directory,file):
    #non = np.array()
    non = np.genfromtxt(directory + "/" +file,delimiter ="" )
    #print(file[:-4])
    #non = non.reshape([1,250])
    print(non.ndim,non.dtype,non.shape,non[0].size)
    #print(non[49])
    return non




def read_location(nondata,img,imgfilename):
    image_shape=np.shape(img)
    height=image_shape[0]
    width=image_shape[1]
    #print(nondata.size -1)
    
    #print(pd.len(nondata))
    for index in range(0,50,1):
        #print(height,width)
        label = nondata[index,0]
        #num = str(label) #convert to string
        bx = nondata[index,1]
        #print(bx)
        by = nondata[index,2]
        bwidth = nondata[index,3]
        bheight = nondata[index,4]
        #rec = np.array([bx,by,bwidth,bheight])
        #rec = np.array([bx*width,by*height,bwidth*width,bheight*height])
        rec = np.array([bx*width - bwidth *width * 0.5,by*height - bheight * height * 0.5,bwidth*width,bheight*height])
        rec = rec.astype(int).tolist()
        roi_image = get_rec_image(img,rec)
        if label == 0.0:
            dir_lab = 'one'
        elif label == 1.0:
            dir_lab = 'two'
        elif label == 2.0:
            dir_lab = 'three'
        elif label == 3.0:
            dir_lab = 'four'
        elif label == 4.0:
            dir_lab = 'five'        
        elif label == 5.0:
            dir_lab = 'six'
        elif label == 6.0:
            dir_lab = 'seven'
        elif label == 7.0:
            dir_lab = 'eight'
        elif label == 8.0:
            dir_lab = 'nine'
        elif label == 9.0:
            dir_lab = 'ten'
        _imgfilename = imgfilename[:-4] + '_' + str(index) + '.jpg'
        #print(_imgfilename)
        save_img_label(roi_image,'./data/' + dir_lab + '/' + _imgfilename)

        #cv_show_image(label,roi_image)
        #print(rec)
    return rec
        
def save_img_label(img,datapath):
    #if len(rgb_image.shape) == 2:  # 若是灰度图则转为三通道
    #    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)
    #image_shape=np.shape(img_ori)
    #height=image_shape[0]
    #width=image_shape[1]
    resize_width = 128;
    resize_height = 128;
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, dsize=(resize_width, resize_height))
    cv2.imwrite(datapath, img)
    #cv_show_image("",img)
    #channels=img.shape[-1]
    #channels=img.shape
    #print(datapath)

def get_rec_image(image,rect):
    '''
    :param image:
    :param rect: [x,y,w,h]
    :return:
    '''
    x, y, w, h=rect
    cut_img = image[y:(y+ h),x:(x+w)]
    return cut_img

def read_dir(img_dir,anno_dir):
    for imgfile in os.listdir(r"./"+img_dir):
        #print(directory_name + "/" + filename)
        #read_non(directory_name + "/" +filename)
        nondata = read_non(anno_dir,imgfile[:-4]+".txt")
        img = cv2.imread(img_dir + "/" +imgfile)
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rec = read_location(nondata,img,imgfile)
        #get_rec_image(img,rec[0])
        #print(nondata.dtype,nondata.shape)
        #print(nondata)


# this function is for read image,the input is directory name
def read_directory(directory_name):
    array_of_img = [] # this if for store all of the image data
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    #for filename in os.listdir(r"./"+directory_name):
    for filename in os.listdir(r"./"+directory_name):
        #print(filename) #just for test
        #img is used to store the image data 
        img = cv2.imread(directory_name + "/" + filename)
        array_of_img.append(img)
        #print(img)
        print(array_of_img)

def read_image(filename, resize_height=None, resize_width=None, normalization=False):
    '''
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization:是否归一化到[0.,1.0]
    :return: 返回的RGB图片数据
    '''
 
    bgr_image = cv2.imread(filename)
    # bgr_image = cv2.imread(filename,cv2.IMREAD_IGNORE_ORIENTATION|cv2.IMREAD_COLOR)
    if bgr_image is None:
        print("Warning:不存在:{}", filename)
        return None
    if len(bgr_image.shape) == 2:  # 若是灰度图则转为三通道
        print("Warning:gray image", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
 
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    # show_image(filename,rgb_image)
    # rgb_image=Image.open(filename)
    rgb_image = resize_image(rgb_image,resize_height,resize_width)
    rgb_image = np.asanyarray(rgb_image)
    if normalization:
        # 不能写成:rgb_image=rgb_image/255
        rgb_image = rgb_image / 255.0
    # show_image("src resize image",image)
    return rgb_image
 
def fast_read_image_roi(filename, orig_rect, ImreadModes=cv2.IMREAD_COLOR, normalization=False):
    '''
    快速读取图片的方法
    :param filename: 图片路径
    :param orig_rect:原始图片的感兴趣区域rect
    :param ImreadModes: IMREAD_UNCHANGED
                        IMREAD_GRAYSCALE
                        IMREAD_COLOR
                        IMREAD_ANYDEPTH
                        IMREAD_ANYCOLOR
                        IMREAD_LOAD_GDAL
                        IMREAD_REDUCED_GRAYSCALE_2
                        IMREAD_REDUCED_COLOR_2
                        IMREAD_REDUCED_GRAYSCALE_4
                        IMREAD_REDUCED_COLOR_4
                        IMREAD_REDUCED_GRAYSCALE_8
                        IMREAD_REDUCED_COLOR_8
                        IMREAD_IGNORE_ORIENTATION
    :param normalization: 是否归一化
    :return: 返回感兴趣区域ROI
    '''
    # 当采用IMREAD_REDUCED模式时，对应rect也需要缩放
    scale=1
    if ImreadModes == cv2.IMREAD_REDUCED_COLOR_2 or ImreadModes == cv2.IMREAD_REDUCED_COLOR_2:
        scale=1/2
    elif ImreadModes == cv2.IMREAD_REDUCED_GRAYSCALE_4 or ImreadModes == cv2.IMREAD_REDUCED_COLOR_4:
        scale=1/4
    elif ImreadModes == cv2.IMREAD_REDUCED_GRAYSCALE_8 or ImreadModes == cv2.IMREAD_REDUCED_COLOR_8:
        scale=1/8
    rect = np.array(orig_rect)*scale
    rect = rect.astype(int).tolist()
    bgr_image = cv2.imread(filename,flags=ImreadModes)
 
    if bgr_image is None:
        print("Warning:不存在:{}", filename)
        return None
    if len(bgr_image.shape) == 3:  #
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    else:
        rgb_image=bgr_image #若是灰度图
    rgb_image = np.asanyarray(rgb_image)
    if normalization:
        # 不能写成:rgb_image=rgb_image/255
        rgb_image = rgb_image / 255.0
    roi_image=get_rect_image(rgb_image , rect)
    # show_image_rect("src resize image",rgb_image,rect)
    # cv_show_image("reROI",roi_image)
    return roi_image
 
def resize_image(image,resize_height, resize_width):
    '''
    :param image:
    :param resize_height:
    :param resize_width:
    :return:
    '''
    image_shape=np.shape(image)
    height=image_shape[0]
    width=image_shape[1]
    if (resize_height is None) and (resize_width is None):#错误写法：resize_height and resize_width is None
        return image
    if resize_height is None:
        resize_height=int(height*resize_width/width)
    elif resize_width is None:
        resize_width=int(width*resize_height/height)
    image = cv2.resize(image, dsize=(resize_width, resize_height))
    return image
def scale_image(image,scale):
    '''
    :param image:
    :param scale: (scale_w,scale_h)
    :return:
    '''
    image = cv2.resize(image,dsize=None, fx=scale[0],fy=scale[1])
    return image
 
 
def get_rect_image(image,rect):
    '''
    :param image:
    :param rect: [x,y,w,h]
    :return:
    '''
    x, y, w, h=rect
    cut_img = image[y:(y+ h),x:(x+w)]
    return cut_img
def scale_rect(orig_rect,orig_shape,dest_shape):
    '''
    对图像进行缩放时，对应的rectangle也要进行缩放
    :param orig_rect: 原始图像的rect=[x,y,w,h]
    :param orig_shape: 原始图像的维度shape=[h,w]
    :param dest_shape: 缩放后图像的维度shape=[h,w]
    :return: 经过缩放后的rectangle
    '''
    new_x=int(orig_rect[0]*dest_shape[1]/orig_shape[1])
    new_y=int(orig_rect[1]*dest_shape[0]/orig_shape[0])
    new_w=int(orig_rect[2]*dest_shape[1]/orig_shape[1])
    new_h=int(orig_rect[3]*dest_shape[0]/orig_shape[0])
    dest_rect=[new_x,new_y,new_w,new_h]
    return dest_rect
 
def show_image_rect(win_name,image,rect):
    '''
    :param win_name:
    :param image:
    :param rect:
    :return:
    '''
    x, y, w, h=rect
    point1=(x,y)
    point2=(x+w,y+h)
    cv2.rectangle(image, point1, point2, (0, 0, 255), thickness=2)
    cv_show_image(win_name, image)
 
def rgb_to_gray(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image
 
def save_image(image_path, rgb_image,toUINT8=True):
    if toUINT8:
        rgb_image = np.asanyarray(rgb_image * 255, dtype=np.uint8)
    if len(rgb_image.shape) == 2:  # 若是灰度图则转为三通道
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)
    else:
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, bgr_image)
 
def combime_save_image(orig_image, dest_image, out_dir,name,prefix):
    '''
    命名标准：out_dir/name_prefix.jpg
    :param orig_image:
    :param dest_image:
    :param image_path:
    :param out_dir:
    :param prefix:
    :return:
    '''
    dest_path = os.path.join(out_dir, name + "_"+prefix+".jpg")
    save_image(dest_path, dest_image)
 
    dest_image = np.hstack((orig_image, dest_image))
    save_image(os.path.join(out_dir, "{}_src_{}.jpg".format(name,prefix)), dest_image)

if __name__=='__main__':

    #read_directory('./img')
    #read_nonDir('./annotest')
    read_dir('./img','./anno')