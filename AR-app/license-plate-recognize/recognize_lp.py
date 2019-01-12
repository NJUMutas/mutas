from hyperlpr import *


def lp_recognize(frames):
    lpboxes_list = []
    index = 0
    length = len(frames)
    for frame in frames:
        f = frame.copy()
        res_set = HyperLPR_PlateRecogntion(f, minSize=20)
        bboxes = []
        for pstr, confidence, rect in res_set:
            if confidence > 0.7:
                # liscence plate content AND its location
                bboxes.append([pstr, rect])
        lpboxes_list.append(bboxes)
        print(index, end='/')
        print(length)
        index += 1
    return lpboxes_list
