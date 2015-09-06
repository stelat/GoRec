import numpy as np
import cv2
from sklearn.preprocessing import normalize

#extract a feature from an image
#is used at the training and testing step
#return the feature vector which can be for example a greyscale histogram
def extract_feature(image):


    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist( [grey.astype('uint8')],
                         channels=[0], 
                         mask=np.ones(grey.shape).astype('uint8'),  # <-- convert to uint8
                         histSize=[256], 
                         ranges=[0,255] )     

    hist= np.squeeze(hist)

    return hist/sum(hist)

