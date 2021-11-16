import math
import numpy as np
import copy
 
from myImageFilter import myImageFilter

def gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def gradDir(x, y):
    degree = math.atan2(y,x)
    de=0
    if -math.pi/2 <= degree and degree < -3*math.pi/8:
        de = math.pi/2
    elif -3*math.pi/8 <= degree and degree < -math.pi/8:
        de = 3*math.pi/4
    elif -math.pi/8 <= degree and degree < math.pi/8:
        de = 0
    elif math.pi/8 <= degree and degree < 3*math.pi/8:
        de = math.pi/4
    elif 3*math.pi/8 <= degree and degree <= math.pi/2:
        de = math.pi/2,
    return de

def cmp(n1, cent, n2):
  if n1 > cent or n2 > cent:
      val = 0
  else:
      val = cent
  return val

def myEdgeFilter(img0, sigma):
  #print(img0)
  #Your implemention
  hsize = 2*math.ceil(3*sigma)+1
  #print(hsize)
  filter = gauss2D((hsize, hsize), sigma)
  #print(filter)
  im1 = myImageFilter(img0, filter)

  # x-axis gradient
  sobel = [[-0.5, 0, 0.5]]
  imgx = myImageFilter(im1, sobel)
  # y-axis gradient
  sobelz=np.transpose(sobel)
  imgy = myImageFilter(im1, sobelz)

  # image gradient magnitude
  row = img0.shape[0]
  col = img0.shape[1]
  imgm = np.zeros((row, col))
  for r in range(0, row):
      for c in range(0, col):
          imgm[r][c] = (imgx[r][c]**2 + imgy[r][c]**2)**0.5

  # non-max suppression
  img1 = copy.deepcopy(imgm)
  for r in range(1, row-1):
      for c in range(1, col-1):
          x = imgx[r, c]
          y = imgy[r, c]
          de = gradDir(x, y)
          if de==0:
            img1[r][c] = cmp(imgm[r][c-1], imgm[r][c], imgm[r][c+1])
          if de==math.pi/4:
            img1[r][c] = cmp(imgm[r-1][c+1], imgm[r][c], imgm[r+1][c-1])
          if de==math.pi/2:
            img1[r][c] = cmp(imgm[r-1][c], imgm[r][c], imgm[r+1][c])
          if de==3*math.pi/4:
            img1[r][c] = cmp(imgm[r-1][c-1], imgm[r][c], imgm[r+1][c+1])
  return img1