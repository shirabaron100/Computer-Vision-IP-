"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from typing import List

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2

def myID() -> np.int:
    return 208761452


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image,  and returns in converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img=cv2.imread(filename)
    if(representation==LOAD_GRAY_SCALE):
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # normalized the mat
    img = img * (1 / 255)
    # mat in float
    img=np.array(img,dtype=float)
    return img

def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img=imReadAndConvert(filename,representation)
    if(representation==LOAD_RGB):
     plt.imshow(img)
    else:
     plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.show()



def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    imgRGBHelp=imgRGB.transpose()
    w=len(imgRGBHelp[0])
    h=len(imgRGBHelp[0][0])
    imgRGBHelp=imgRGBHelp.reshape(3,h*w)
    transMat = np.array([[0.299, 0.587 ,0.114],[0.596, -0.275 ,-0.321],[ 0.212 ,-0.523 ,0.311]], dtype=float)
    imgYIQ=transMat.dot(imgRGBHelp)
    imgYIQ=imgYIQ.reshape(3,w,h).transpose()
    return imgYIQ
    pass

def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    transMat = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]], dtype=float)
    inverse = np.linalg.inv(transMat)
    imgYIQHelp = imgYIQ.transpose()
    w = len( imgYIQHelp[0])
    h = len( imgYIQHelp[0][0])
    imgRGBHelp =  imgYIQHelp.reshape(3, h * w)
    imgRGB= inverse.dot(imgRGBHelp)
    imgRGB = imgRGB.reshape(3, w, h).transpose()
    return imgRGB
    pass


def hsitogramEqualize(imOrig:np.ndarray)->(np.ndarray,np.ndarray,np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :return: (imgEq,histOrg,histEQ)
    """
    # a check if the image is gray or rgb
    # print(imgOrig)
    image=imOrig
    if (isGray(image) == True):
        image = image * 255
        histOrg, bins = np.histogram(image.ravel(), 256, [0, 256])
        # histogram eqalization algorithm:
        cum = np.cumsum(histOrg)
        N = max(cum)
        histEQ = (cum / N) * 255
        histEQ = np.floor(histEQ)
        histEQ = histEQ.astype(int)
        image = image.astype(int)
        imgEq = image
        # mapping the image with the histogram eq
        for i in range(len(image)):
            for j in range(len(image[0])):
                imgEq[i][j] = histEQ[image[i][j]]
        histEQ,dontcare= np.histogram(imgEq.flatten(), 256, [0, 256])

    else:
        #convert the image to YIQ
        imgOrigYIQ = transformRGB2YIQ(imOrig)
        imgOrigYIQ=imgOrigYIQ*255
        #histogram with only Y channel
        image = imgOrigYIQ[:, :, 0]
        histOrg,bins = np.histogram(image.ravel(),256,[0,256])
        #histogram eq algo
        histEQ = np.cumsum(histOrg)
        N = max(histEQ)
        histEQ = (histEQ / N)*255
        histEQ = np.floor(histEQ)
        histEQ = histEQ.astype(int)
        image = image.astype(int)
        imgOrigYIQ=imgOrigYIQ.astype(int)
        imgEq =image
        #mapping the image with histo eq
        for i in range(len(image)):
            for j in range(len(image[0])):
                imgEq[i][j] = histEQ[image[i][j]]
        histEQ, dontcare = np.histogram(imgEq.flatten(), 256, [0, 256])
        imgOrigYIQ[:, :, 0] = imgEq
        imgEq= transformYIQ2RGB(imgOrigYIQ*(1/255))
    return (imgEq,histOrg,histEQ)
    pass

def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    if (isGray(imOrig) == True):
        imgO=imOrig*255
        hist1, bin = np.histogram(imgO.flatten(), 256)
        hist, bins = np.histogram(imgO.flatten(), nQuant)
        z = bins.astype('uint8')
        q = np.zeros(nQuant)
        img_lst=[]
        MSE_lst=[]
        img_lst.append(imgO)
        for i in range(nIter):
            imgO=imOrig*255
            for j in range(nQuant):
                down =(int)(z[j])
                up = (int)(z[j + 1])
                mask_pixels = np.ma.masked_inside(imgO, down, up)
                q[j] = mean(hist1,down,up)
                if j != 0:
                    s=(0.5 *(q[j - 1] + q[j]))
                    # s=s.astype(int)
                    z[j] = s
                np.ma.set_fill_value(mask_pixels, q[j])
                imgO = mask_pixels.filled()
            dif=imOrig*255-imgO
            MSE=np.sqrt(sum(np.power(dif.flatten(), 2)))/sum(hist1)
            MSE_lst.append(MSE)
            img_lst.append(imgO)
    else:
        img_lst = []
        MSE_lst=[]
        img_lst.append(imOrig)
        # convert the image to YIQ
        imgOrigYIQ = transformRGB2YIQ(imOrig)
        imgOrigYIQ = imgOrigYIQ * 255
        # histogram with only Y channel
        image = imgOrigYIQ[:, :, 0]
        hist1, bin = np.histogram(image.flatten(), 256)
        hist, bins = np.histogram(image.flatten(), nQuant)
        z = bins.astype('uint8')
        q = np.zeros(nQuant)

        for i in range(nIter):
            image= imgOrigYIQ[:, :, 0]
            for j in range(nQuant):
                down = (z[j])
                up = (z[j + 1])
                mask_pixels = np.ma.masked_inside(image, down, up)
                q[j] = mean(hist1, down, up)
                if j != 0:
                    s = (0.5 * (q[j - 1] + q[j]))
                    s = s.astype(int)
                    z[j] = s
                np.ma.set_fill_value(mask_pixels, q[j])
                image = mask_pixels.filled()
            imgOrigYIQ[:, :, 0] =image
            imgRGB = transformYIQ2RGB(imgOrigYIQ * (1 / 255))
            img_lst.append(imgRGB)
            dif = imOrig * 255 - imgRGB
            MSE = (np.sqrt(sum(np.power(dif.flatten(), 2))) / sum(hist1))
            MSE_lst.append(MSE)
    return (img_lst,MSE_lst)


#help function from stackOverFlow cheacking if the image is gray
def isGray(imgOrig: np.ndarray) -> (bool):
    if len(imgOrig.shape) < 3: return True
    if imgOrig.shape[2] == 1: return True
    b, g, r = imgOrig[:, :, 0], imgOrig[:, :, 1], imgOrig[:, :, 2]
    if (b == g).all() and (b == r).all(): return True
    return False
    pass

def mean(a, start ,end) -> float :
    sum, counter= 0.0,0.0
    j=end
    if (end==255):
        j=end+1
    for i in range(start,j):
        sum=a[i]*i+sum
        counter=a[i]+counter
    if(counter==0):
        counter=1
    n=(sum/counter)
    return (int(n))