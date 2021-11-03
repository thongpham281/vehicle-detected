import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models.experimental import attempt_load
from utils.general import check_suffix, is_ascii, non_max_suppression, scale_coords, set_logging
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox


def get_box_info(box):
    (x, y, w, h) = [int(v) for v in box]
    center_X = int((x + (x + w)) / 2.0)
    center_Y = int((y + (y + h)) / 2.0)
    return x, y, w, h, center_X, center_Y


@torch.no_grad()
def vehicle_detected_model(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='vehicle_detecteds/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        image='',
        ):
    # Setup
    max_distance = 30
    curr_trackers = []
    obj_cnt = 0
    car_number, truck_number, motorbike_number = 0, 0, 0
    CAR, TRUCK, MOTOR = "car", "truck", "motorcycle"
    object_needed = [CAR, TRUCK, MOTOR]

    def countAndUpdateTracker(car_number, laser_line_color, old_trackers):
        for car in old_trackers:
            # Update tracker
            tracker = car['tracker']
            (_, box) = tracker.update(im0)
            boxes.append(box)

            new_obj = dict()
            new_obj['tracker_id'] = car['tracker_id']
            new_obj['tracker'] = tracker

            # Tinh toan tam doi tuong
            x, y, xR, yR, center_X, center_Y = get_box_info(box)

            # So sanh tam doi tuong voi duong laser line
            if laser_line_up < center_Y <= laser_line:
                # Neu vuot qua thi khong track nua ma dem xe
                cv2.circle(im0, (center_X, center_Y), 4, (0, 255, 255), 5)
                laser_line_color = (0, 255, 255)
                car_number += 1
            else:
                # Con khong thi track tiep
                curr_trackers.append(new_obj)

        return car_number, laser_line_color

    def is_old(center_Xd, center_Yd, boxes):
        for box_tracker in boxes:
            (xt, yt, wt, ht) = [int(c) for c in box_tracker]
            center_Xt, center_Yt = int((xt + (xt + wt)) / 2.0), int((yt + (yt + ht)) / 2.0)
            distance = math.sqrt((center_Xt - center_Xd) ** 2 + (center_Yt - center_Yd) ** 2)

            if distance < max_distance - 20:
                return True
        return False

    def updateObjectTracker(obj_cnt):
        # Objects detected
        for box in boxes_d:
            # Get info box of each Object
            xd, yd, wd, hd, center_Xd, center_Yd = get_box_info(box)

            # If center width of Object > laser_line
            if center_Yd >= laser_line:
                if not is_old(center_Xd, center_Yd, boxes):
                    cv2.rectangle(im0, (xd, yd), ((xd + wd), (yd + hd)), (0, 255, 255), 2)

                    # Tao doi tuong tracker moi
                    # tracker = cv2.legacy.TrackerMOSSE_create()
                    tracker = cv2.legacy.TrackerCSRT_create()

                    obj_cnt += 1
                    new_obj = dict()
                    tracker.init(im0, tuple(box))

                    new_obj['tracker_id'] = obj_cnt
                    new_obj['tracker'] = tracker

                    curr_trackers.append(new_obj)
        return curr_trackers

    def getLabel(box):
        if box in box_label[CAR]:
            return CAR
        elif box in box_label[TRUCK]:
            return TRUCK
        elif box in box_label[MOTOR]:
            return MOTOR
        return "other"

    def drawBox(boxes_d, laser_line):
        for box in boxes_d:
            c1 = (int(box[0]), int(box[1]))
            c2 = (int(box[0]) + int(box[2]), int(box[1]) + int(box[3]))

            if (c1[1] + c2[1]) // 2 >= laser_line:
                annotator.box_label_new(box, getLabel(box), color=colors(10, True))

    def removeDuplicate(boxes_label):
        boxes_truck = boxes_label[TRUCK]
        boxes_car = boxes_label[CAR]
        boxes_motor = boxes_label[MOTOR]

        if len(boxes_truck) == 0:
            return boxes_car

        for box_truck in boxes_truck:
            xL_truck, yL_truck, xR_truck, yR_truck, center_X_truck, center_Y_truck = get_box_info(box_truck)
            for box_car in boxes_car:
                xL_car, yL_car, xR_car, yR_car, center_X_car, center_Y_car = get_box_info(box_car)
                distance = math.sqrt((center_X_truck - center_X_car) ** 2 + (center_Y_truck - center_Y_car) ** 2)
                if distance < max_distance:
                    boxes_car.remove(box_car)

        return boxes_truck + boxes_car + boxes_motor

    save_img = not nosave and not source.endswith('.txt')  # save inference images

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)

    # vehicle_detected inference
    dt, seen = [0.0, 0.0, 0.0], 0

    # Start Detecting
    # Padded resize
    im0s = image
    img = letterbox(im0s)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    t1 = time_sync()
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1

    # Inference
    if pt:
        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(img, augment=augment, visualize=visualize)[0]

    t3 = time_sync()
    dt[1] += t3 - t2

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    dt[2] += time_sync() - t3

    # Process predictions
    for i, det in enumerate(pred):  # per image
        laser_line_color = (0, 0, 255)
        # boxes detected
        boxes_d = []
        # boxes tracked
        boxes = []
        # mapping box vehicle -> label
        box_label = {}
        for label in object_needed:
            box_label[label] = []

        seen += 1
        s, im0 = '', im0s.copy()
        s += '%gx%g ' % img.shape[2:]  # print string

        annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class

                    # if object is 'bicycle', 'car', 'motor', 'bus', 'truck' then draw box
                    if names[c] in object_needed:
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                        # get box info
                        box = annotator.box_info(xyxy)

                        # add box to box_label
                        label_name = label.split(" ")[0]
                        box_label[label_name] += [box]

                        # add box to boxes_d
                        # box[2], box[3] = box[2] - box[0], box[3] - box[1]
                        boxes_d.append(box)

        # Init laser_line m30, urban1
        laser_line_up = im0.shape[0] * 0.30
        laser_line = im0.shape[0] * 0.55

        # remove duplicate
        boxes_d = removeDuplicate(box_label)
        drawBox(boxes_d, laser_line)

        # Loop through tracker
        old_trackers = curr_trackers
        curr_trackers = []

        # CASE VIDEO
        # if dataset.mode != "image":

        # Loop thought object tracker
        car_number, laser_line_color = countAndUpdateTracker(car_number, laser_line_color, old_trackers)

        # Detect object per frames
        # if frame_count % FRAMES == 0:
        curr_trackers = updateObjectTracker(obj_cnt)

        # Print time (inference-only)
        print(f'{s}Done. ({t3 - t2:.3f}s)')

        # Show total car
        # cv2.putText(im0, "Car number: " + str(car_number), (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    return im0, boxes_d