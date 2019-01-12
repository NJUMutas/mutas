import cv2


def frame_2_video(frames, fps, source):
    if len(frames) == 0:
        return -1
    # vid_writer = cv2.VideoWriter('output-{}.mp4'.format(str(source).split(".")[0]),
    #                              cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps,
    #                              (frames[0].shape[1], frames[0].shape[0]))
    vid_writer = cv2.VideoWriter('./output_video/output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps,
                                 (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        # cv2.imshow("Face Detection Comparison", frame)
        vid_writer.write(frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
    # cv2.destroyAllWindows()
    vid_writer.release()
    return 0
