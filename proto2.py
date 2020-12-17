import cv2
import numpy as np
#ffpyplayer for playing audio
from ffpyplayer.player import MediaPlayer
video_path="/home/romans/Downloads/TradeOptionsImplyVolatility"
def PlayVideo(video_path):
    video=cv2.VideoCapture(video_path)
    player = MediaPlayer(video_path)
    brightness = 0
    brightness_delta = 5
    while True:
        grabbed, frame=video.read()
        brightness += brightness_delta
        if brightness > 254:
            brightness_delta = -5
            brightness = 255
            player.set_volume(0.1)
        if brightness < 1:
            brightness_delta = 5
            brightness = 1
            player.set_volume(1.0)
        cv2.normalize(frame, frame, 0, brightness, cv2.NORM_MINMAX) # 0 - 255
        #player.set_volume(1.0) # 0.0 - 1.0
        audio_frame, val = player.get_frame()
        if not grabbed:
            print("End of video")
            break
        if cv2.waitKey(28) & 0xFF == ord("q"):
            break
        cv2.imshow("Video", frame)
        if val != 'eof' and audio_frame is not None:
            #audio
            img, t = audio_frame
    video.release()
    cv2.destroyAllWindows()
PlayVideo(video_path)