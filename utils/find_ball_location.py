from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2


# this function finds out that is ball inside in the court or not. 
def findBallLocate(ball, line_coords):

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(ball, line_coords[0], line_coords[1])
    d2 = sign(ball, line_coords[1], line_coords[2])
    d3 = sign(ball, line_coords[2], line_coords[3])
    d4 = sign(ball, line_coords[3], line_coords[0])

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0) or (d4 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0) or (d4 > 0)

    return not (has_neg and has_pos)


def ball_coords(frames,interpolated_positions,line_keypoints):
    last_detected = datetime.now()
    i=0
    isBallIn=None
    touching_ground=[2,44,83,117,168,200]

    video=[]

    for frame, ball in zip(frames,interpolated_positions):
        orjballx1,orjbally1= ball


        cv2.circle(frame,(round(orjballx1),round(orjbally1)),5,(75,0,110),cv2.FILLED)


        if i in touching_ground:
            k,l = interpolated_positions[i]     
            a=round(k)
            b=round(l)       
            

            isBallIn=findBallLocate((a,b), line_keypoints)
            last_detected = datetime.now()

        else:
            if (datetime.now() - last_detected).total_seconds() < 0.4:
                if(isBallIn is not None):
                    if(isBallIn):
                        cv2.putText(frame, 'In', (a,b+2), cv2.FONT_HERSHEY_PLAIN,2, (0,255,0),2)
                    if(isBallIn==False):
                        cv2.putText(frame, 'Out', (a,b+2), cv2.FONT_HERSHEY_COMPLEX,1, (0,0,255),2)
                    cv2.circle(frame,(a,b),3,(0,255,255),cv2.FILLED)

        video.append(frame)
        i+=1
    return video
