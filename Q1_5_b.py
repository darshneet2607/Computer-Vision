import cv2
from myImageFilter import myImageFilter
img1 = cv2.imread('UCS532_Assignment1_2021/data/img01.jpg')
img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
f=[[0,-0.5,0,0.5,0]]
im2=myImageFilter(img,f)
im3=myImageFilter(im2,f)
print('Second derivative of the image: ')
print(im3)