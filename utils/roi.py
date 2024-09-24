import numpy as np
import cv2


def region_of_interest(frames):
    row,column=frames[0].shape[:2]
 
    mask=np.zeros((row,column),dtype=np.uint8)   
    backgorundList=[]
    frontgorundList=[]
    
    for frame in frames:

        #cropped_img=frame[170:,350:1620]

        points = np.array([(400, 175), (1550, 175),(1850, 1080),(20, 1080)])
        
        # background is blackk roi is white
        mask=cv2.fillPoly(mask,[points],(255,255,255))

        # background is black roi is orignal
        masked=cv2.bitwise_and(frame,frame,mask=mask)

        # background is white roi is black
        backgorund=cv2.bitwise_not(mask)

        # background is original roi is black
        backgorund=cv2.bitwise_and(frame,frame,mask=backgorund)

        frontgorundList.append(masked)
        backgorundList.append(backgorund)

        

        # baskround is original roi is original we sum .
        #completed_frame=cv2.add(backgorund,foreground)

        #cv2.imshow('foto1',foreground)  

        #if(cv2.waitKey()&0xFF==27):
             #break



        

    #cv2.destroyAllWindows()   
    return frontgorundList,backgorundList
