def frame_select(frames, factor):  # 5 means: select 1 per 5 frames
    selected_frames = []
    i = 0
    for frame in frames:
        if i % factor == 0:
            f = frame.copy()
            selected_frames.append(f)
        i += 1
    return selected_frames
