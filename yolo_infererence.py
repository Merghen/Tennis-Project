from ultralytics import YOLO







model_path=r'F:\Goruntu Isleme 1\python\projeler\tennis\models\yolo5_last.pt'
input_video_path=r'F:\Goruntu Isleme 1\python\projeler\tennis\input_video\input_video.mp4'
model=YOLO(model_path)
result=model.track(input_video_path,conf=0.2,save=True)