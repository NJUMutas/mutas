import cv2


def frame_resize(frames, factor):  # factor = 0.5
    resized_frames = []
    for frame in frames:
        f = frame.copy()
        f = cv2.resize(f, (round(f.shape[1] * factor), round(f.shape[0] * factor)))
        resized_frames.append(f)
    return resized_frames
