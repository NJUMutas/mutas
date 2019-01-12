from video_to_frame import video_2_frame
from resize_frame import frame_resize
from select_frame import frame_select
from detect_frontal_face import frontal_face_detect
from draw_hat import hat_draw
from frame_to_video import frame_2_video
import sys


if __name__ == "__main__":
    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]
    frames, fps = video_2_frame(source)
    # resize_factor = 0.5: 720 * 720 -> 360 * 360
    # fps_factor = 5: get 1 frame per 5 frames
    resize_factor = 0.5
    fps_factor = 1
    resized_frames = frame_resize(frames, resize_factor)
    selected_frames = frame_select(resized_frames, fps_factor)
    bboxes_list = frontal_face_detect(selected_frames)
    drawn_frames = hat_draw(frames, resize_factor, fps_factor, bboxes_list)
    frame_2_video(drawn_frames, fps, source)