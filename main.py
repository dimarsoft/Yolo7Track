import os
import cv2
import torch
# from random import random as rnd_v
from random import randrange
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
import numpy as np
from tracker import Tracker

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def model_predict(frame):

    img, _, _ = letterbox(frame)

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
        pred = model(img)[0]
    return pred, img

def get_centrmass(p1, p2):
    res = ( int((p2[0] + p1[0])/2), int(p2[1] + 0.35 * (p1[1] - p2[1])) )
    return res

if __name__ == '__main__':

    x1line, y1line = 277, 384
    x2line, y2line = 437, 348

    fn = "30"
    video_path = os.path.join("data_test", f"{fn}.mp4")
    video_out_path = os.path.join("output", f"{fn}_ann.mp4")
    model_path = os.path.join("ann_mod", "best_b4e54.pt")
    track_model_path = os.path.join("ann_mod", "mars-small128.pb")
    device = "cpu"
    imgsz = 640

    GOFILE = True

    # Load model
    model = attempt_load(model_path, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    colors = [(randrange(255), randrange(255), randrange(255)) for i in range(10)]
    people_id = 0
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if GOFILE:
        cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                  cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))
    else:
        cap_out = None

    tracker_people = Tracker(track_model_path)
    itt = 0
    inp_people, utp_people = 0, 0
    while ret:
        itt += 1
        print(itt)
        if itt > 0:
            pred, img_new = model_predict(frame)
            pred = non_max_suppression(pred)

            detection = []
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img_new.shape[2:], det[:, :4], frame.shape).round()
                    # print(det)
                    for obj_i in det:
                        x1, y1, x2, y2 = int(obj_i[0]), int(obj_i[1]), int(obj_i[2]), int(obj_i[3])
                        score, class_id = obj_i[4], int(obj_i[5])
                        if class_id == people_id:
                            detection.append([x1, y1, x2, y2, score])
                        # else:
                        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            if len(detection) > 0:
                tracker_people.update(frame, detection)

                for track in tracker_people.tracks:
                    bbox = track.bbox
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    track_id = track.track_id

                    col_i = (colors[track_id % len(colors)])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), col_i, 1)

                    cv2.circle(frame, get_centrmass((x1, y1), (x2, y2)), radius=5, color=col_i, thickness=-1)

                    cv2.putText(frame, f"id={track_id}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                                1, col_i, 2, cv2.LINE_AA)

            cv2.putText(frame, f"input: {inp_people} / output: {utp_people}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            # cv2.imwrite("frame_line.png", frame)
            cv2.line(frame, (x1line, y1line), (x2line, y2line), (0, 255, 0), 3)

            if GOFILE:
                cap_out.write(frame)
            else:
                cv2.imshow('frame', frame)
                cv2.waitKey(25)

        ret, frame = cap.read()

    if GOFILE:
        cap_out.release()
    else:
        cap.release()
        cv2.destroyAllWindows()