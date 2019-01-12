import dlib


def frontal_face_detect(frames):
    predictor_path = "./front_face_detect_model/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    detector = dlib.get_frontal_face_detector()

    bboxes_list = []
    index = 0
    length = len(frames)
    for frame in frames:
        f = frame.copy()
        dets = detector(f, 1)
        bboxes = []
        for det in dets:
            # x,y,w,h = det.left(),det.top(),det.right()-det.left(),det.bottom()-det.top()
            shape = predictor(f, det)
            bboxes.append((det, shape))
        bboxes_list.append(bboxes)
        print(index, end='/')
        print(length)
        index += 1
    return bboxes_list
