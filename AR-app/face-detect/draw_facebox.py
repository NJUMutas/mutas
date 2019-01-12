import cv2


def facebox_draw(frames, resize_factor, fps_factor, bboxes_list):
    # resize_factor = 0.5: 720 * 720 -> 360 * 360
    # fps_factor = 5: get 1 frame per 5 frames
    drawn_frames = []
    i = 0
    for frame in frames:
        drawn_frame = frame.copy()
        index = i // fps_factor
        for bbox in bboxes_list[index]:
            cv2.rectangle(drawn_frame, (round(bbox[0] / resize_factor), round(bbox[1] / resize_factor)),
                          (round(bbox[2] / resize_factor), round(bbox[3] / resize_factor)), (0, 255, 0),
                          int(round(drawn_frame.shape[0] / 150)),
                          8)
        drawn_frames.append(drawn_frame)
        i += 1
    return drawn_frames
