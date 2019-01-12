import cv2


def face_detect(frames):
    dnn = "CAFFE"
    #dnn = "TF"
    if dnn == "CAFFE":
        model_file = "./face_detect_model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        config_file = "./face_detect_model/deploy.prototxt"
        net = cv2.dnn.readNetFromCaffe(config_file, model_file)
    else:
        model_file = "./face_detect_model/opencv_face_detector_uint8.pb"
        config_file = "./face_detect_model/opencv_face_detector.pbtxt"
        net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
    conf_threshold = 0.7
    bboxes_list = []
    index = 0
    length = len(frames)
    for frame in frames:
        f = frame.copy()
        frame_height = f.shape[0]
        frame_width = f.shape[1]
        # blob = cv2.dnn.blobFromImage(f, 1.0, (300, 300), [104, 117, 123], False, False)
        blob = cv2.dnn.blobFromImage(f, 1.0, (300, 300))
        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                bboxes.append([x1, y1, x2, y2])
        bboxes_list.append(bboxes)
        print(index, end='/')
        print(length)
        index += 1
    return bboxes_list

