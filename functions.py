from copy import deepcopy

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


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def crossing_bound(people_path, bound_line):
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

    intersect_ = intersect(p1_proj, p2_proj, bound_line[0], bound_line[1])
    return {"direction": direction, "intersect": intersect_}


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
    max_delt = 5  # frame
    for pk in people_tracks.keys():
        path = people_tracks[pk]["path"]
        frid = people_tracks[pk]["frid"]
        bbox = people_tracks[pk]["bbox"]
        new_path = {"path": [path[0]], "frid": [frid[0]], "bbox": [bbox[0]]}
        for i in range(1, len(frid)):
            if frid[i] - frid[i - 1] > max_delt and len(new_path) > 1:
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


def remakedict(dict_m):
    base_x = []
    for k in dict_m.keys():
        base_x += dict_m[k]["frid"]
    base_x = list(set(base_x))
    res = {v: [] for v in base_x}

    for k in dict_m.keys():
        obj = dict_m[k]
        for i in range(len(obj["frid"])):
            res[obj["frid"][i]].append(obj["bbox"][i])
    return res


def remakedict_person(dict_m):
    base_x = []
    for k in dict_m.keys():
        base_x += dict_m[k]["frid"]
    base_x = list(set(base_x))
    res = {v: [] for v in base_x}

    for k in dict_m.keys():
        obj = dict_m[k]
        for i in range(len(obj["frid"])):
            res[obj["frid"][i]].append({
                "id": k,
                "bbox": obj["bbox"][i],
                "helmet": False,
                "vest": False
            })
    base_x = sorted(base_x)
    return res, base_x


def get_ind_list(mass, val, key):
    res = None
    for i in range(len(mass)):
        if mass[i][key] == val:
            res = i
            break
    return res


def get_iou(pred_box, gt_box):
    # функция для расчета метрики. Изначально думал делать через iou,
    # но сейчас считаю как отношение площади пересечения к площади объекта - жилет / каска
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)

    inters = iw * ih

    #     uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
    #            (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
    #            inters)
    uni = (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.)
    iou = inters / uni

    return iou


def calc_iou(bbox_peop, bbox_obj, objkey, upbbox=True):
    result = deepcopy(bbox_peop)
    flatten = lambda l: [item for sublist in l for item in sublist]
    for o_bbx in bbox_obj:
        vids = []
        viou = []
        for p_bbx in result:
            valbbx = flatten(p_bbx["bbox"])
            if upbbox:
                valbbx[3] = (valbbx[1] + valbbx[3]) / 2
            val = get_iou(valbbx, flatten(o_bbx))
            vids.append(p_bbx["id"])
            viou.append(val)
        if len(vids) > 0:
            indmax = np.nanargmax(viou)
            if viou[indmax] > 0.7:
                indres = get_ind_list(bbox_peop, vids[indmax], "id")
                if not indres is None:
                    bbox_peop[indres][objkey] = True
                    result.pop(indmax)
                else:
                    print("?!?!?!?!??!")
    return bbox_peop


def iou_prop(people, helmet, vest):
    for frid in people.keys():
        if not helmet.get(frid) is None:
            people[frid] = calc_iou(people[frid], helmet[frid], "helmet", upbbox=True)
        if not vest.get(frid) is None:
            people[frid] = calc_iou(people[frid], vest[frid], "vest", upbbox=False)


def get_deviations(people_tracks, helmet_tracks, vest_tracks, bound_line):
    people_info = get_people_info(people_tracks, helmet_tracks, vest_tracks)
    return find_deviations(people_info, bound_line, False)


def get_people_info(people_tracks, helmet_tracks, vest_tracks):
    helmet_tracks_id = remakedict(helmet_tracks)
    vest_tracks_id = remakedict(vest_tracks)

    people_tracks_id, base_x = remakedict_person(people_tracks)

    iou_prop(people_tracks_id, helmet_tracks_id, vest_tracks_id)

    for k in people_tracks.keys():
        n = len(people_tracks[k]["path"])
        people_tracks[k]["helmet"] = [None for i in range(n)]
        people_tracks[k]["vest"] = [None for i in range(n)]

    for xi in people_tracks_id.keys():
        for xobj in people_tracks_id[xi]:
            if xi in people_tracks[xobj["id"]]["frid"]:
                indx = people_tracks[xobj["id"]]["frid"].index(xi)
                people_tracks[xobj["id"]]["helmet"][indx] = xobj['helmet']
                people_tracks[xobj["id"]]["vest"][indx] = xobj['vest']

    return people_tracks


def search_frame(people_info, bound_line):
    n = len(people_info["path"])

    line_norm = get_norm(*bound_line)

    res = None
    for i in range(n - 1, 0, -1):
        p1_proj = get_proj(*line_norm, people_info["path"][i])
        p2_proj = get_proj(*line_norm, people_info["path"][i - 1])
        intersect_val = intersect(p1_proj, p2_proj, bound_line[0], bound_line[1])
        if intersect_val:
            res = i  # [i, int((i + n-1) / 2)]
            break
    return res


def find_deviations(people_tracks, bound_line, only_down: bool = True):
    tracks_info = []
    intrs_pid = []

    devs = []
    bool_print = False
    for p_id in people_tracks.keys():
        people_path = people_tracks[p_id]['path']
        tr_info = crossing_bound(people_path, bound_line)
        tracks_info.append(tr_info)

        if tr_info["intersect"]:
            intrs_pid.append(p_id)
            if only_down:
                process = tr_info["direction"] == 'down'
            else:
                process = True
        else:
            process = False

        if process:
            helmet = np.array(people_tracks[p_id]["helmet"], dtype=bool)
            vest = np.array(people_tracks[p_id]["vest"], dtype=bool)
            in_helm = len(helmet[helmet]) / len(helmet) >= 0.5
            in_vest = len(vest[vest]) / len(vest) >= 0.5

            truefalse = "Нет" if in_helm == True and in_vest == True else "Да"

            no_dev = in_helm and in_vest

            id_frame_ind = search_frame(people_tracks[p_id], bound_line)
            if not id_frame_ind is None:
                id_frame = people_tracks[p_id]["frid"][id_frame_ind]
                if not no_dev:
                    bbox = people_tracks[p_id]["bbox"][id_frame_ind]
                    devs.append({
                        "frame_id": id_frame,
                        "has_helmet": in_helm,
                        "has_uniform": in_vest,
                        "box": bbox})

                if in_vest == False or in_helm == False:
                    # print(file, p_id, "Тут есть нарушение? - ", truefalse, f"id_frame = {id_frame}")
                    print(p_id, "Тут есть нарушение? - ", truefalse, f"id_frame = {id_frame}")

                    print(f"Жилетка: {in_vest} / Каска: {in_helm}")
                    # save_frame(fn, id_frame, people_tracks[p_id]["bbox"][id_frame_ind], bound_line, folder_with_video)

            bool_print = True
    return devs
