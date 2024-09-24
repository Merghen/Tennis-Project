import cv2

# read the video
def read_video(path): # video path

    # read the video
    video=cv2.VideoCapture(path)
    frames=[]
    while (video.isOpened()):
        succes, frame = video.read()

        if(succes==False):
            break

        # Add every frame into 'frames' list.
        frames.append(frame)

    video.release()

    return frames

# save the frames.
def save_video(frames,output_path): # whole video, save path.

    # get width and height of the video
    width=frames[0].shape[1]
    height=frames[0].shape[0]

    output= cv2.VideoWriter(output_path, 
                         cv2.VideoWriter_fourcc(*'XVID'),  
                         30, 
			 (width,height))
    

    # write the video frame by frame
    for frame in frames:
        output.write(frame)

    output.release()

    



