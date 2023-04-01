import os
import cv2
import json
from random import randrange
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from functions import model_predict, get_centrmass, crossing_bound, calc_inp_outp_people, process_filt
from tracker import Tracker
import multiprocessing
# from tqdm import tqdm

def load_bound_line(cameras_path):
    with open(cameras_path, 'r') as f:
        bound_line = json.load(f)
    return bound_line


def model_init(model_path, config):
    # Load model
    model = attempt_load(model_path, map_location=config["device"])  # load FP32 model
    # stride = int(model.stride.max())  # model stride
    # imgsz = check_img_size(imgsz, s=stride)  # check img_size
    return model


def update_tracker(obj_track, frame, itt, detection_obj, myobj_tracks, colors, godraw = False):
    obj_track.update(frame, detection_obj)
    for track in obj_track.tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        track_id = track.track_id
        if track_id in myobj_tracks.keys():
            myobj_tracks[track_id]["path"].append(get_centrmass((x1, y1), (x2, y2)))
            myobj_tracks[track_id]["frid"].append(itt)
            myobj_tracks[track_id]["bbox"].append([(x1, y1), (x2, y2)])
        else:
            myobj_tracks.update({track_id: {
                "path": [get_centrmass((x1, y1), (x2, y2))],
                "bbox": [[(x1, y1), (x2, y2)]],
                "frid": [itt]
            }})
        if godraw:
            col_i = (colors[track_id % len(colors)])
            cv2.rectangle(frame, (x1, y1), (x2, y2), col_i, 1)

            cv2.circle(frame, get_centrmass((x1, y1), (x2, y2)), radius=5, color=col_i, thickness=-1)

            cv2.putText(frame, f"id={track_id}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                        1, col_i, 2, cv2.LINE_AA)
    return myobj_tracks

def video_analysis(model, config, video_path, video_out_path, bound_line = []):
    colors = [(randrange(255), randrange(255), randrange(255)) for i in range(10)]

    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, frame = cap.read()

    if config["GOFILE"]:
        cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                  cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))
    else:
        cap_out = None

    tracker_people = Tracker(config["track_model_path"])
    helmet_people  = Tracker(config["track_model_path"])
    vest_people    = Tracker(config["track_model_path"])

    itt = 0
    people_tracks = {}
    helmet_tracks = {}
    vest_tracks = {}

    while ret:
        # print(f"{itt} / {video_length}")
        itt += 1

        # if itt < 150:
        #     ret, frame = cap.read()
        #     continue

        pred, img_new = model_predict(model, frame, config["device"])
        pred = non_max_suppression(pred)

        detection_people = []
        detection_helmet = []
        detection_vest   = []
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img_new.shape[2:], det[:, :4], frame.shape).round()
                for obj_i in det:
                    x1, y1, x2, y2 = int(obj_i[0]), int(obj_i[1]), int(obj_i[2]), int(obj_i[3])
                    score, class_id = obj_i[4], int(obj_i[5])
                    if class_id == config["people_id"]:
                        detection_people.append([x1, y1, x2, y2, score])
                    elif class_id == config["helmet_id"]:
                        detection_helmet.append([x1, y1, x2, y2, score])
                    elif class_id == config["vest_id"]:
                        detection_vest.append([x1, y1, x2, y2, score])
                    # else:
                    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        if len(detection_people) > 0:
            people_tracks = update_tracker(tracker_people, frame, itt, detection_people, people_tracks, colors, godraw=True)
        if len(detection_helmet) > 0:
            helmet_tracks = update_tracker(helmet_people, frame, itt, detection_helmet, helmet_tracks, colors, godraw=True)
        if len(detection_vest) > 0:
            vest_tracks = update_tracker(vest_people, frame, itt, detection_vest, vest_tracks, colors, godraw=True)
        if len(bound_line) > 1:
            cv2.line(frame, *bound_line, (0, 255, 0), 3)

        if config["GOFILE"]:
            cap_out.write(frame)
        else:
            cv2.imshow('frame', frame)
            cv2.waitKey(25)

        ret, frame = cap.read()
    if config["GOFILE"]:
        cap_out.release()
    else:
        cap.release()
        cv2.destroyAllWindows()
    return people_tracks, helmet_tracks, vest_tracks

def proc_analysis(camera_mass):
    for camera_num in camera_mass:
        print(camera_num)
        bound_line = bound_line_cameras.get(camera_num)
        if bound_line is None:
            print(f"error! no camera {camera_num}")
            continue

        outfold = os.path.join("output", camera_num)
        if not os.path.exists(outfold):
            os.makedirs(outfold)
        camera_fold = os.path.join("data_test", camera_num)
        files = os.listdir(camera_fold)

        for fn in files:
            fn = fn.split(".")[0]
            video_path = os.path.join(camera_fold, f"{fn}.mp4")
            video_out_path = os.path.join(outfold, f"{fn}_ann.mp4")

            people_tracks, helmet_tracks, vest_tracks = video_analysis(model, config, video_path, video_out_path, bound_line)
            if len(people_tracks) == 0:
                print(f"null_video! {fn}")
                continue
            people_tracks = process_filt(people_tracks)

            if True:
                with open(f"{fn}_people_tracks_new.json", "w") as f:
                    json.dump(people_tracks, f, indent=4)
                with open(f"{fn}_helmet_tracks_new.json", "w") as f:
                    json.dump(helmet_tracks, f, indent=4)
                with open(f"{fn}_vest_tracks_new.json", "w") as f:
                    json.dump(vest_tracks, f, indent=4)

            tracks_info = []
            for p_id in people_tracks.keys():
                people_path = people_tracks[p_id]
                tr_info = crossing_bound(people_path['path'], bound_line)
                tracks_info.append(tr_info)
                print(f"{p_id}: {tr_info}")
            result = calc_inp_outp_people(tracks_info)
            print(result)
            print("______________")

            stat_out_path = os.path.join(outfold, f"{fn}_people_path.json")
            with open(stat_out_path, "w") as f:
                json.dump(result, f, indent=4)

def mainrun(pool, cpu_count, cams):
    col_wells = len(cams)
    list_cams = []
    step = col_wells // cpu_count + col_wells % cpu_count

    for i in range(0, col_wells - 1, step):
        list_cams.append(cams[i: i + step])
    # list_cams = ["18"]
    # p = proc_analysis(list_cams)
    p = pool.map(proc_analysis, list_cams)
    return p



config = {
        "device": "cpu",
        "GOFILE": True,
        "people_id": 0,
        "helmet_id": 1,
        "vest_id": 2,
        "model_path" : os.path.join("ann_mod", "best_b4e54.pt"),
        "track_model_path": os.path.join("ann_mod", "mars-small128.pb"),
        "cameras_path": os.path.join("configuration", "camera_config.json")
    }
bound_line_cameras = load_bound_line(config["cameras_path"])
model = model_init(config["model_path"], config)

if __name__ == '__main__':

    full_cameras = os.listdir("data_test")

    cpu_count = multiprocessing.cpu_count()
    cpu_count = max(3, cpu_count - 2)
    pool = multiprocessing.Pool(cpu_count)

    print("Strat multiprocessing")
    mainrun(pool, cpu_count, full_cameras)