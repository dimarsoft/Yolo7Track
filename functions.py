import cv2
import torch
import numpy as np


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


def model_predict(model, frame, device):
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
    res = (int((p2[0] + p1[0]) / 2), int(p2[1] + 0.35 * (p1[1] - p2[1])))
    return res


def get_proj(p1, p2, m):
    k = (p2[0] - p1[0]) / (p2[1] - p1[1])
    f1 = m[1] + k * m[0]
    f2 = p1[1] - (1 / k) * p1[0]
    x_proj = (f1 - f2) / (k + 1 / k)
    y_proj = f1 - k * x_proj
    return [x_proj, y_proj]


def get_norm(p1, p2):
    pc = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
    xmin, xmax = pc[0] - 10, pc[0] + 10
    a = (p2[1] - p1[1]) / (p2[0] - p1[0])
    fnorm = lambda x: pc[1] - (1 / a) * (x - pc[0])
    line_norm = [[xmin, fnorm(xmin)], [xmax, fnorm(xmax)]]
    return line_norm


def crossing_bound(people_path, bound_line):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    if len(people_path) >= 4:
        p1 = [(people_path[0][0] + people_path[1][0]) / 2, (people_path[0][1] + people_path[1][1]) / 2]
        p2 = [(people_path[-2][0] + people_path[-1][0]) / 2, (people_path[-2][1] + people_path[-1][1]) / 2]
    else:
        p1 = [people_path[0][0], people_path[0][1]]
        p2 = [people_path[-1][0], people_path[-1][1]]

    direction = "up" if p2[1] - p1[1] < 0 else "down"

    line_norm = get_norm(*bound_line)
    p1_proj = get_proj(*line_norm, p1)
    p2_proj = get_proj(*line_norm, p2)

    intersect = intersect(p1_proj, p2_proj, bound_line[0], bound_line[1])
    return {"direction": direction, "intersect": intersect}


def calc_inp_outp_people(tracks_info):
    input_p = 0
    output_p = 0
    for track_i in tracks_info:
        if track_i["intersect"]:
            if track_i["direction"] == 'down':
                input_p += 1
            elif track_i["direction"] == 'up':
                output_p += 1
    return {"input": input_p, "output": output_p}


def process_filt(people_tracks):
    max_id = max([int(idv) for idv in people_tracks.keys()])
    max_id += 1
    res = {}
    max_delt = 5 # frame
    for pk in people_tracks.keys():
        path = people_tracks[pk]["path"]
        frid = people_tracks[pk]["frid"]
        bbox = people_tracks[pk]["bbox"]
        new_path = {"path": [path[0]], "frid": [frid[0]], "bbox": [bbox[0]]}
        for i in range(1, len(frid)):
            if frid[i] - frid[i-1] > max_delt and len(new_path) > 1:
                if str(pk) in res.keys():
                    new_id = str(max_id)
                    max_id += 1
                else:
                    new_id = str(pk)
                res.update({new_id: new_path})
                new_path = {"path": [path[i]], "frid": [frid[i]], "bbox": [bbox[i]]}
            else:
                new_path["path"].append(path[i])
                new_path["frid"].append(frid[i])
                new_path["bbox"].append(bbox[i])
        if len(new_path) > 1:
            if str(pk) in res.keys():
                new_id = str(max_id)
                max_id += 1
            else:
                new_id = str(pk)
            res.update({new_id: new_path})
    return res
