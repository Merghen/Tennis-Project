import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

def get_person_boxes(boxes):
    # get index value of the object which is person
    persons_index=[index for index,deger in enumerate(boxes.cls.tolist()) if deger==0]

    persons_boxes=[]
    for i in persons_index:
            # get the location informations of the object which is only person
            x= boxes.xyxy.tolist()[i]

            # Add persons locations to list.
            persons_boxes.append(x)

    # converting it to array
    persons_boxes=np.array(persons_boxes)   
    
    
    return persons_boxes

#----------------------------------------------------------------------
    

my_pose = mp.solutions.pose

# pose detection and params
body=my_pose.Pose(model_complexity=0,
                  static_image_mode=False,
                  min_detection_confidence=0.7, 
                  min_tracking_confidence=0.75) 



# for drawing body boints.
draw=mp.solutions.drawing_utils

def pose_estimation(bbox,masked_frame):
    # getting xmin y min x max y max locations
    x1,y1,x2,y2=bbox
    x1,y1,x2,y2=round(x1),round(y1),round(x2),round(y2)

    # we crop the area that includes person
    cropped_person=masked_frame[y1:y2,x1:x2]

    # convert it to rgb because medipipe works with rgb photos.
    cropped_person_rgb = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)

    result=body.process(cropped_person_rgb)
    
    # if there is any detected area:
    if(result.pose_landmarks!=None):

        # draw the landmarks
        draw.draw_landmarks(cropped_person, result.pose_landmarks, my_pose.POSE_CONNECTIONS)

        #we put our edited person area on original frame
        masked_frame[y1:y2,x1:x2] = cropped_person
    
    


def trackPerson(masked,background,model_path,poseEstimation):
    
    # initialize model.
    model=YOLO(model_path)

    frames=[]
    person_coordinats=[]

    for masked_frame,background_frame in zip(masked,background):
        
        # detection.
        result = model.track(masked_frame, conf=0.5, save=False)
        #ids=result[0].boxes.id.int().tolist()

        if(len(result[0].boxes.xyxy)>0):

            # getting location of persons
            person_boxes=get_person_boxes(result[0].boxes)
            person_coordinats.append(person_boxes)

            for bbox in  person_boxes:

                # if they want to see landmark points:
                if(poseEstimation==True):
                     
                    pose_estimation(bbox,masked_frame)

                # drawing boundry box
                x1,y1,x2,y2=bbox
                x1,y1,x2,y2=round(x1),round(y1),round(x2),round(y2)

                #cv2.putText(masked_frame,'Person:'+str(id),(x1,y1-10),cv2.FONT_HERSHEY_PLAIN,2, (0,0,0))
                cv2.rectangle(masked_frame,(x1,y1),(x2,y2),(255,0,0))


        # we concat our masked frame and background frame to get original state of the video..
        complete_frame=cv2.add(masked_frame,background_frame)

        # add every frame to to list 
        frames.append(complete_frame)
    
    return frames,person_coordinats

    
        
    
        
    
        