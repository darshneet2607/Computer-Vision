import numpy as np

def myImageFilter(image, kernel):
    """
    This function which takes an image and a kernel and returns the convolution of them.
    :param image: a numpy array of size [image_height, image_width].
    :param kernel: a numpy array of size [kernel_height, kernel_width].
    :return: a numpy array of size [image_height, image_width] (convolution output).
    """
    # Flip the kernel
    kernel = np.flipud(np.fliplr(kernel))
    # convolution output
    output = np.zeros_like(image)
    h_size=kernel.shape[0]
    padding=(int)((h_size-1)/2)
    #print(h_size)
    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + h_size-1, image.shape[1] + h_size-1))
    row=0
    col=0
    for j in range(padding,image.shape[1]+padding):
      for i in range(padding,image.shape[0]+padding):
        image_padded[i,j]=image[row,col]
        row=row+1
      col=col+1
      row=0
    #image_padded[padding:-1, padding:-1] = image
    # Loop over every pixel of the image
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y, x] = (kernel * image_padded[y: y+h_size, x: x+h_size]).sum()

    return output