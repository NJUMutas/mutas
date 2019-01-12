import cv2
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import numpy as np


def lpbox_draw(frames, resize_factor, fps_factor, lpboxes_list):
    # resize_factor = 0.5: 720 * 720 -> 360 * 360
    # fps_factor = 5: get 1 frame per 5 frames
    drawn_frames = []
    i = 0
    for frame in frames:
        f = frame.copy()
        index = i // fps_factor
        for LPbox in lpboxes_list[index]:
            pstr = LPbox[0]
            rect = LPbox[1]
            cv2.rectangle(f, (round(rect[0] / resize_factor), round(rect[1] / resize_factor)),
                          (round(rect[2] / resize_factor), round(rect[3] / resize_factor)), (0, 0, 255),
                          int(round(f.shape[0] / 300)),
                          8)  # cv2.LINE_AA
            cv2.rectangle(f, (round(rect[0] / resize_factor), round((rect[1] - 30) / resize_factor)),
                          (round((rect[0] + 120) / resize_factor), round(rect[1] / resize_factor)), (0, 0, 255), -1,
                          8)  # cv2.LINE_AA
            fontC = ImageFont.truetype("./font/platech.ttf", 20, 0)
            img = Image.fromarray(f)
            draw = ImageDraw.Draw(img)
            draw.text((round((rect[0] + 5) / resize_factor), round((rect[1] - 30) / resize_factor)), pstr,
                      (255, 255, 255), font=fontC)
            # pstr.decode("utf-8"),
            f = np.array(img)
        drawn_frames.append(f)
        i += 1
    return drawn_frames
