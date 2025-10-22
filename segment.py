import cv2
import numpy as np

def segment_image(input_path,output_path):
    img= cv2.imread(input_path) # read image from disk into a BGR NumPy array (None if file not found)
    h, w = img.shape[:2]  # get image height (h) and width (w)
    mask= np.zeros((h, w), np.uint8) # initialize grabCut mask (same spatial size, single channel)
    rect = (1, 1, w-1, h-1) # creates a rectangle tuple for grabCut
    bgd , fgd= np.zeros((1,65), np.float64), np.zeros((1,65), np.float64)  #creates a zero-intlized Numpy array for grubcut
    cv2.grabCut(img,mask,rect,bgd,fgd, 5, cv2.GC_INIT_WITH_RECT) # run grabCut to estimate foreground
    mask2= np.where((mask==2)|(mask==0),0,1).astype('uint8') # convert grabCut mask to binary (0=bg,1=fg)
    result = img * mask2[:,:, None] # apply binary mask to each color channel; result keeps foreground pixels
    gray= cv2.cvtColor(result, cv2. COLOR_BGR2GRAY) # convert masked result to grayscale for alpha creation
    _, alpha = cv2.threshold(gray,0,255, cv2.THRESH_BINARY) # threshold grayscale to produce binary alpha channel
    b,g,r = cv2.split(result) # split BGR channels from the masked color image
    rgba = cv2.merge((r,g,b,alpha)) # merge channels into RGBA (reordering to R,G,B + alpha)
    cv2.imwrite(output_path, rgba)  # write the RGBA image to disk

if __name__ == '__main__':
    segment_image('/home/project/Documents/test2/assests/garments/front.png','/home/project/Documents/test2/assests/garments/front_seg.png')
    segment_image('/home/project/Documents/test2/assests/garments/back.png','/home/project/Documents/test2/assests/garments/back_seg.png')  
