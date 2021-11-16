import numpy as np

def fi(image, h):
  
    # convolution output
    output = np.zeros_like(image)
    h_size=h.shape[0]
    padding=(int)((h_size-1)/2)
    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + h_size-1, image.shape[1] + h_size-1))
    row=0
    col=0
    #fill the padded image matrix of zeros with original image values
    for j in range(padding,image.shape[1]+padding):
      for i in range(padding,image.shape[0]+padding):
        image_padded[i,j]=image[row,col]
        row=row+1
      col=col+1
      row=0
    
    # Loop over every pixel of the image
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y, x] = (h * image_padded[y: y+h_size, x: x+h_size]).sum()

    return output

f=np.array([[1,2,1]])
I=np.array([[0,1,2,3,3,3,1,3,6]])
fi(I,f)