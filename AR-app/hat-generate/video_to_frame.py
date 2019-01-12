import cv2


def video_2_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    # CV_CAP_PROP_FPS == 5
    fps = cap.get(5)
    frames = []
    has_frame, frame = cap.read()
    while has_frame:
        frames.append(frame)
        has_frame, frame = cap.read()
    return frames, fps
