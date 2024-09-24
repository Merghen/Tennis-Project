import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np


def line_detection(modelPath,frames):
    
    model=models.resnet50(pretrained=True)
    model.fc=torch.nn.Linear(model.fc.in_features,28)

    keys=[[576, 303],[1334,303],[1564, 854],[365,854]]

    model.load_state_dict(torch.load(modelPath, map_location='cpu'))

    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])
    
    frame=frames[0]
    rgbImage=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #print(rgbImage.shape)

    # unsqueeze(0) ---> we change the shape, we add batch size 50,50,3 -> 1,50,50,3
    tensorImage = transform(rgbImage).unsqueeze(0)
    #print(tensorImage.shape)

    with torch.no_grad():
        # predict the model.
        outputs = model(tensorImage)

    keypoints = outputs.squeeze().cpu().numpy()
    # these keypoints are for 224x224 but our original image has different shape.
    #print(keypoints) # it keep locations like x,y / x,y / x,y / x,y
    
    height,width=frame.shape[0:2]
    
    # first 14 point is x coordinat last 14 point is y coordinat.
    #print(keypoints[:14]*width/224.0)

    #print(keypoints[::2]*width/224)
 
    updated_keypoints=keypoints.copy()
    # we take first location and continue +2. this way, we acces only x locations update them according to width.
    updated_keypoints[::2]=keypoints[::2]*width/224.0
    # we take second location and continue +2. this way, we acces only y locations update them according to height.
    updated_keypoints[1::2]=keypoints[1::2]*height/224.0
    #print(updated_keypoints)
   # return updated_keypoints

    return keys


def draw_lines(frames,keypoints):
    video=[]
    
    cv2.namedWindow('testWindow',cv2.WINDOW_NORMAL)
    for frame in frames:
       
        fr=frame.copy()
     
        if(cv2.waitKey(1)&0xFF==27):
            break
        for indeks in range(len(keypoints)-1):
            #print(indeks)
            x1,y1=keypoints[indeks]
            x2,y2=keypoints[indeks+1]
            cv2.line(fr,(x1,y1),(x2,y2),(0,0,255),3)
            if indeks == 2:
                lx1,ly1=keypoints[-1]
                lx2,ly2=keypoints[0]
                cv2.line(fr,(lx1,ly1),(lx2,ly2),(0,0,255),3)
        
        video.append(fr)
    
                
    return video    
