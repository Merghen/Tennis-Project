import cv2
import numpy as np
from ultralytics import YOLO

def detectBall(frames,model_path):

    ball_coordinats=[]
    # create model
    model=YOLO(model_path)
    # edited video frames.
    video=[]

    for frame in frames:
        # get the result
        result=model.track(frame,conf=0.10,persist=True,save=False)
        result=result[0]
        # if there is detected area then continue..
        if len(result.boxes.xywh>0):
            # get locations
            x1,y1,x2,y2=result.boxes.xywh.tolist()[0]
            # making floats int
            x1,y1,x2,y2=round(x1),round(y1),round(x2),round(y2)
            print(x1,y1,x2,y2)
            print(result.boxes.xywh)

            # get the ball coordinats
            ball_coordinats.append([x1,y1])
        
        else:
            ball_coordinats.append(None)
  

    
    interpolated_positions=interpolate_ball_locations(ball_coordinats)


    for frame,poses in zip(frames,interpolated_positions):
        

        x1,y1=round(poses[0]),round(poses[1])     
        cv2.putText(frame,'ball',(x1,y1-10),cv2.FONT_HERSHEY_PLAIN,1, (255,255,255))
        cv2.circle(frame,(x1,y1),6,(0,0,255),thickness=-1)

        video.append(frame)

        


    return video,interpolated_positions


def interpolate_ball_locations(ball_coordinats):

    x_coords = np.array([pos[0] if pos is not None else np.nan for pos in ball_coordinats])
    y_coords = np.array([pos[1] if pos is not None else np.nan for pos in ball_coordinats])

    # print(x_coords)
        # print(y_coords)

    #print(np.isnan(x_coords))

    # get indexes of not null locations
    validIndexes=np.where(~np.isnan(x_coords))[0]

    #print(validIndexes)


    x_valid = x_coords[validIndexes]
    y_valid = y_coords[validIndexes]


    x_interp = np.interp(np.arange(len(x_coords)), validIndexes, x_valid)
    y_interp = np.interp(np.arange(len(y_coords)), validIndexes, y_valid)

    interpolated_positions = list(zip(x_interp, y_interp))

    return interpolated_positions
