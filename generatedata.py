import numpy as np
import cv2
from time import time

boxes = []

resolution=32
listPoints=[]
width=resolution*19
done=False
M =np.empty([3, 3])
def readImage(frame):

    #Lens distortion rectification does not work. No idea why
    
    # f=0.050
    # cameraMatrix=np.asarray([[ f,  0.,     (frame.shape[1]-1)*0.5    ],
    #                          [ 0.,f ,(frame.shape[0]-1)*0.5   ],
    #                          [ 0., 0., 1]])
 
    # print cameraMatrix
    # img=cv2.undistort(frame, cameraMatrix,np.zeros((1,4),dtype=float),newCameraMatrix)
    
 
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    res = cv2.resize(gray,(0,0), fx = 2, fy = 2)

    return res


def drawRectangles(warp):
    # draw the rectangles 19*19 rectangles of the spot for the stones on the go ban
    #display only
    for i in range(0,19):
        for j in range(0,19):

            tlt=(resolution*i,resolution*i)
            brt=(resolution*(j+1),resolution*(j+1))
            cv2.rectangle(warp,tlt,brt,0,1)

def exportRectangles(warp,filesuffix):
    #save images of the spots 
    for i in range(0,19):
        for j in range(0,19):

            tlt=(resolution*i,resolution*i)
            brt=(resolution*(j+1),resolution*(j+1))
            rect=warp[(resolution*i):(resolution*(i+1)),(resolution*j):(resolution*(j+1))]
            cv2.imwrite( filesuffix+"_"+str(i)+"_"+str(j)+".jpg", rect );

def initSelect():
    # function to crop the image to get only the goban
    #apply an affine transformation to fit the goban into a square
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, width - 1],
        [0, width - 1]], dtype = "float32")
      #matrice of the perspective transformation to transform the goban into a square


    boxes=[]


    def on_mouse(event, x, y, flags, params):
        # global img
        t = time()
        global done
        global boxes
        global M
        # global width
        

        if event == cv2.EVENT_LBUTTONDOWN:
            print 'Start Mouse Position: '+str(x)+', '+str(y)
            sbox = [x, y]
            boxes.append(sbox)
            # print count
            # print sbox

            if len(boxes)==4:
                print boxes
                shift0X=(boxes[1][0]-boxes[0][0])/36.0
                shift0Y=(boxes[2][1]-boxes[0][1])/36.0

                shift1X=(boxes[1][0]-boxes[0][0])/36.0
                shift1Y=(boxes[3][1]-boxes[1][1])/36.0

                shift2X=(boxes[3][0]-boxes[2][0])/36.0
                shift2Y=(boxes[0][1]-boxes[2][1])/36.0

                shift3X=(boxes[3][0]-boxes[2][0])/36.0
                shift3Y=(boxes[3][1]-boxes[1][1])/36.0
                print boxes
                listPoints= [[boxes[0][0]-shift0X,boxes[0][1]-shift0Y],[boxes[1][0]+shift1X,boxes[1][1]-shift1Y],[boxes[3][0]+shift3X,boxes[3][1]+shift3Y],[boxes[2][0]-shift2X,boxes[2][1]-shift2Y]]
                print listPoints
                src = np.array(listPoints,np.float32)


                M = cv2.getPerspectiveTransform(src, dst)
                warp = cv2.warpPerspective(img, M, (width, width))

                drawRectangles(warp)

                boxes=[]
                cv2.imshow('crop',warp)
                print "do you want to keep it? (y/n)"
                k =  cv2.waitKey(0)


                if 1048697== k:#check if "y"
                    done= True
                    print "cropping saved"
            else :
                cv2.destroyWindow('crop')

    count = 0

    while(not done):
        count += 1
    #    img = cv2.imread('CaptureNord.png',0)
    #img = cv2.blur(img, (3,3))
    #    img = cv2.resize(img, None, fx = 0.25,fy = 0.25)

        cv2.namedWindow('real image')
        cv2.setMouseCallback('real image', on_mouse, 0)
        cv2.imshow('real image', img)
        if count < 50:
            if cv2.waitKey(33) == 27:
                cv2.destroyAllWindows()

                break
        elif count >= 50:
            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()
                break
            count = 0



cap = cv2.VideoCapture("goCrop.avi")
ret, frame = cap.read()


img = readImage(frame)





#ask the user 4 points : top-left top-right bottom-left and bottom right 
#warning respect the order
#press "y" on the "crop" window to accept the cropping 
#or press another key anf close the window to try again
initSelect()


idx=1
while(cap.isOpened()):
    
    ret, frame = cap.read()
    
    if (idx %500 ==0): #every 500 frames
        img=readImage(frame)
        
        #apply the affine transformation M
        warp = cv2.warpPerspective(img, M, (width, width))

        #display the rectangles and saves the images in data
        drawRectangles(warp) 
        cv2.imshow('crop',warp)
        exportRectangles(warp,"data/frame"+str(idx))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    idx+=1
