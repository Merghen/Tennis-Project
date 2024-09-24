from utils import (read_video,save_video,region_of_interest,draw_lines,line_detection,ball_coords)
from track import (detectBall,trackPerson)
from mini_court import (create_mini_court)


input_video_path=r'F:\Goruntu Isleme 1\python\projeler\tennis\input_video\input_video.mp4'
ball_model_path=r'F:\Goruntu Isleme 1\python\projeler\tennis\models\yolo5_last.pt'
person_model_path=r'F:\Goruntu Isleme 1\python\projeler\tennis\models\yolov8m.pt'
keypoint_model_path=r"F:\Goruntu Isleme 1\python\projeler\tennis\models\keypointsDetection.pth"
output_video_path=r"F:\Goruntu Isleme 1\python\projeler\tennis\output_video\Proje_Tennis.avi"


frames=read_video(input_video_path)



ball_frames,ball_locations=detectBall(frames,ball_model_path)

mask,background=region_of_interest(frames)

frames,person_coordinats=trackPerson(mask,background,person_model_path,poseEstimation=False)

keypoints=line_detection(keypoint_model_path,frames)

frames=draw_lines(frames,keypoints)

frames=ball_coords(frames,ball_locations,keypoints)

frames=create_mini_court(frames,keypoints,person_coordinats,ball_locations)

save_video(frames,output_video_path)







