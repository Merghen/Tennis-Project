import numpy as np
import cv2


def create_mini_court(frames,line_keypoints,bboxes,interpolated_positions):
    img=frames[0].copy()

    # get shape of the original frame
    orj_width=img.shape[1]
    orj_height=img.shape[0]

    # determine width and hight of the mini court
    miniCourt_width=300 #360
    miniCourt_height=560 #640



    video=[]
    miniCourtLocations=[]
    for frame, bbox, ball in zip(frames,bboxes,interpolated_positions):

        # created minicourt with black color
        miniCourt=np.zeros(shape=(miniCourt_height,miniCourt_width,3),dtype=np.uint8)
        
        #  scale original line keypoints into mini tennis court 
        for point in line_keypoints:
    
            orjix1,orjiy1 = point
            minix1=round(miniCourt_width*orjix1/orj_width)
            miniy1=round(miniCourt_height*orjiy1/orj_height)
            miniCourtLocations.append([minix1,miniy1])

        # drawing lines
        for indeks in range(len(miniCourtLocations)-1):

            #print(indeks)
            orjix1,orjiy1=miniCourtLocations[indeks]
            orjix2,orjiy2=miniCourtLocations[indeks+1]
            
            #print(orjix1,orjiy1)
            cv2.line(miniCourt,(orjix1,orjiy1),(orjix2,orjiy2),(0,0,255),2)
            if indeks == 2:
                lx1,ly1=miniCourtLocations[-1]
                lx2,ly2=miniCourtLocations[0]
                cv2.line(miniCourt,(orjix1,orjiy1),(orjix2,orjiy2),(0,0,255),4)
 

        #  persvective
        src_points=np.float32([
       
        miniCourtLocations[0], # kesmek istenilen yerin sol üst nokta  
        
        miniCourtLocations[1], # kesmek istenilen yerin sağ üst nokta
        miniCourtLocations[3], # kesmek istenilen yerin  sol alt nokta
        miniCourtLocations[2]  # kesmek istenilen yerin sag alt nokta
    ]) 
 

        #dst_points = np.float32([[68, 180], [293, 180], miniCourtLocations[3], miniCourtLocations[2]])
        dst_points = np.float32([[miniCourtLocations[3][0], miniCourtLocations[0][1]], [miniCourtLocations[2][0], miniCourtLocations[1][1]], miniCourtLocations[3], miniCourtLocations[2]])
        projective_matrix=cv2.getPerspectiveTransform(src_points, dst_points)

        miniCourt=cv2.warpPerspective(miniCourt, projective_matrix, (miniCourt_width,miniCourt_height))



        # ball detection

        orjballx1,orjbally1= ball

        #scaling the ball coordinats
        ballx1=round(miniCourt_width*orjballx1/orj_width)
        bally1=round(miniCourt_height*orjbally1/orj_height)



        cv2.circle(miniCourt,(ballx1,bally1+10),3,(0,255,255),cv2.FILLED)

 
        # ball_position_original = np.float32([ballx1, bally1])
        # ball_position_transformed = cv2.perspectiveTransform(np.array([[ball_position_original]]), projective_matrix)
        # ballx1,bally1=ball_position_transformed[0][0]

        


        # middle line of the tennis court

        lineX=miniCourtLocations[3][0]-15
        lineY=miniCourtLocations[0][1]+((miniCourtLocations[3][1]-miniCourtLocations[0][1])//2)

        line2X=miniCourtLocations[2][0]+15
        line2Y=miniCourtLocations[1][1] +((miniCourtLocations[2][1] -miniCourtLocations[1][1])//2)  


        cv2.line(miniCourt,(lineX,lineY),(line2X,line2Y),(255,255,255),4)


        # person detection
        for box in bbox:
            x1,y1,x2,y2=box
            orjx1,orjy1,orjx2,orjy2= round(x1),round(y1),round(x2),round(y2)
            #minx1=round(miniCourt_width*orjx1/orj_width)
            #miny1=round(miniCourt_height*orjy1/orj_height)

            # scaling person coordinats
            maxx1=round(miniCourt_width*orjx2/orj_width)
            maxy1=round(miniCourt_height*orjy2/orj_height)

            player_position_original = np.float32([maxx1, maxy1])

            player_position_transformed = cv2.perspectiveTransform(np.array([[player_position_original]]), projective_matrix)  

            x,y=player_position_transformed[0][0]
            x,y=round(x),round(y)

              
            cv2.circle(miniCourt,(x,y),8,(255,0,0),cv2.FILLED)




        frame[:miniCourt_height, orj_width-miniCourt_width:]=miniCourt

        video.append(frame)

    return video    
