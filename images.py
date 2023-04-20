import cv2
import numpy as np
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, img_as_ubyte

def ReadPathNum(path,index):
    gray =cv2.cvtColor(cv2.imread("./datasrc/{}/{}-{:04d}.jpg".format(path,path,index)), cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # ymin, ymax, xmin, xmax = h//2, h*2//2, w//2, w*2//2
    # crop = gray[ymin:ymax, xmin:xmax]
    resize = cv2.resize(gray, (0,0), fx=0.5, fy=0.5)
    return resize

def ReturnListMeanFromNp(list):
    newlist=[]
    for i in list:
        newlist.append(np.mean(i))
    return newlist

def CreateHistogramListFromImageList(list):
    newlist=[]
    for i in list:
        newlist.append(cv2.calcHist([i],[0],None,[256],[0,256]))
    return newlist

def ReturnListStdFromNp(list):
    newlist=[]
    for i in list:
        newlist.append(np.std(i))
    return newlist

def contrast_feature(matrix_coocurrence):
    contrast = graycoprops(matrix_coocurrence, 'contrast')
    return contrast

def dissimilarity_feature(matrix_coocurrence):
    dissimilarity = graycoprops(matrix_coocurrence, 'dissimilarity')    
    return dissimilarity

def homogeneity_feature(matrix_coocurrence):
    homogeneity = graycoprops(matrix_coocurrence, 'homogeneity')
    return homogeneity

def energy_feature(matrix_coocurrence):
    energy = graycoprops(matrix_coocurrence, 'energy')
    return energy

def correlation_feature(matrix_coocurrence):
    correlation = graycoprops(matrix_coocurrence, 'correlation')
    return correlation

def asm_feature(matrix_coocurrence):
    asm = graycoprops(matrix_coocurrence, 'ASM')
    return asm

def calc_glcm_all_agls(img, label=None, props=None, dists=[1], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    
    glcm = graycomatrix(img, 
                        distances=dists, 
                        angles=agls, 
                        levels=lvl,
                        symmetric=sym, 
                        normed=norm)
    # feature = []
    # print(glcm.shape)
    # glcm_props = [propery for name in props for propery in graycoprops(glcm, name)[0]]
    # for item in glcm_props:
    #         feature.append(item)
    # feature.append(label) 
    
    return glcm

def FromListToGlcm(list):
    newlist=[]
    for i in list:
        newlist.append(calc_glcm_all_agls(i))
    return newlist

def Readnumber(number):
    return [ReadPathNum("happy",number),ReadPathNum("sad",number),ReadPathNum("neutral",number)]