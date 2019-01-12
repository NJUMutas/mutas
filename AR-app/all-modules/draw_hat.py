import cv2


# Refer:
# img.shape[0]: height
# img.shape[1]: width
# cv2.resize(img, width, height)
# img2 = img1[height1 : height2, width1 : width2]
def hat_draw(frames, resize_factor, fps_factor, bboxes_list):
    # resize_factor = 0.5: 720 * 720 -> 360 * 360
    # fps_factor = 5: get 1 frame per 5 frames
    hat_img0 = cv2.imread("./input_image/hat.png", -1)
    drawn_frames = []
    i = 0
    for frame in frames:
        drawn_frame = frame.copy()
        index = i // fps_factor
        for bbox in bboxes_list[index]:
            hat_img = hat_img0.copy()

            # get some data from bbox
            x = round(bbox[0].left() / resize_factor)
            y = round(bbox[0].top() / resize_factor)
            w = round((bbox[0].right() - bbox[0].left()) / resize_factor)
            h = round((bbox[0].bottom() - bbox[0].top()) / resize_factor)
            shape = bbox[1]

            # we get the positions of eyes and nose
            point1 = shape.part(0)  # left side of face
            point1x, point1y = round(point1.x / resize_factor), round(point1.y / resize_factor)
            point2 = shape.part(16)  # right side of face
            point2x, point2y = round(point2.x / resize_factor), round(point2.y / resize_factor)
            point3 = shape.part(33)  # nose
            point3x, point3y = round(point3.x / resize_factor), round(point3.y / resize_factor)
            point4 = shape.part(38)  # left eye
            point4x, point4y = round(point4.x / resize_factor), round(point4.y / resize_factor)
            point5 = shape.part(43)  # right eye
            point5x, point5y = round(point5.x / resize_factor), round(point5.y / resize_factor)
            eyes_center = ((point1x + point2x) // 2, (point4y + point5y) // 2)
            nose = [point3x, point3y]

            # we assume that eyes are at the middle of the bottom edge and the nose
            if eyes_center[1] * 2 - nose[1] <= 0:
                continue

            # adjust the hat size according to face
            factor = 0.75
            resized_hat_w = int(round(w * factor))
            resized_hat_h = int(round(w * factor * hat_img.shape[0] / hat_img.shape[1]))
            hat_img = cv2.resize(hat_img, (resized_hat_w, resized_hat_h))

            # deal with the case where head is too high
            if (eyes_center[1] * 2 - nose[1]) - resized_hat_h < 0:
                hat_img = hat_img[-((eyes_center[1] * 2 - nose[1]) - resized_hat_h): resized_hat_h, 0: resized_hat_w]
                resized_hat_h = eyes_center[1] * 2 - nose[1]

            # deal with the case where head is too left or right
            if hat_img.shape[1] % 2 == 1:
                hat_img = hat_img[0: hat_img.shape[0], 1: hat_img.shape[1]]
                resized_hat_w -= 1
            hat_half_w = resized_hat_w // 2  # hat_half_width is (hat width)/2 now
            left_cut = 0  # left of hat to be cut
            right_cut = 0  # right of hat to be cut
            if eyes_center[0] < hat_half_w:
                left_cut = hat_half_w - eyes_center[0]
                hat_img = hat_img[0: resized_hat_h, left_cut: resized_hat_w]
                resized_hat_w -= left_cut
            if eyes_center[0] + hat_half_w > drawn_frame.shape[1]:
                right_cut = eyes_center[0] + hat_half_w - drawn_frame.shape[1]
                hat_img = hat_img[0: resized_hat_h, 0: resized_hat_w - right_cut]
                resized_hat_w -= right_cut

            # put the hat image onto the frame
            r, g, b, a = cv2.split(hat_img)
            rgb_hat = cv2.merge((r, g, b))

            mask = a
            mask_inv = cv2.bitwise_not(mask)

            bg_roi = drawn_frame[eyes_center[1] * 2 - nose[1] - resized_hat_h: eyes_center[1] * 2 - nose[1],
                     eyes_center[0] - (hat_half_w - left_cut): eyes_center[0] + (hat_half_w - right_cut)]
            bg_roi = bg_roi.astype(float)

            mask_inv = cv2.merge((mask_inv, mask_inv, mask_inv))
            alpha = mask_inv.astype(float) / 255

            alpha = cv2.resize(alpha, (bg_roi.shape[1], bg_roi.shape[0]))

            bg = cv2.multiply(alpha, bg_roi)
            bg = bg.astype('uint8')

            hat = cv2.bitwise_and(rgb_hat, cv2.bitwise_not(mask_inv))

            hat = cv2.resize(hat, (bg_roi.shape[1], bg_roi.shape[0]))

            add_hat = cv2.add(bg, hat)

            drawn_frame[eyes_center[1] * 2 - nose[1] - resized_hat_h: eyes_center[1] * 2 - nose[1],
            eyes_center[0] - (hat_half_w - left_cut): eyes_center[0] + (hat_half_w - right_cut)] = add_hat
            # drawnFrame = cv2.rectangle(drawnFrame, (point1x, point1y), (point1x + 1, point1y + 1), (255, 0, 0), 2)
            # drawnFrame = cv2.rectangle(drawnFrame, (point2x, point2y), (point2x + 1, point2y + 1), (255, 0, 0), 2)
            # drawnFrame = cv2.rectangle(drawnFrame, (point3x, point3y), (point3x + 1, point3y + 1), (255, 0, 0), 2)
            # drawnFrame = cv2.rectangle(drawnFrame, (point4x, point4y), (point4x + 1, point4y + 1), (255, 0, 0), 2)
            # drawnFrame = cv2.rectangle(drawnFrame, (point5x, point5y), (point5x + 1, point5y + 1), (255, 0, 0), 2)

        drawn_frames.append(drawn_frame)
        i += 1
    return drawn_frames
