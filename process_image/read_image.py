import cv2
import numpy as np

from fastapi import UploadFile
from io import BytesIO
from starlette.responses import StreamingResponse

# Specify the paths for the 2 files
protoFile = "assets/pose_deploy_linevec.prototxt"
weightsFile = "assets/pose_iter_440000.caffemodel"

# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


def cosine(poseVector1, poseVector2):
    result = []
    pose_pair_temp = [[1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
    pose_pair = [[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[8,9],[9,10],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
    for i, j in pose_pair:
        if (poseVector1[i] == None) or (poseVector1[j] == None) or (poseVector2[i] == None) or (poseVector2[j] == None):
            continue
        vectorX = (poseVector1[i][0]-poseVector1[j][0], poseVector1[i][1]-poseVector1[j][1])
        vectorY = (poseVector2[i][0]-poseVector2[j][0], poseVector2[i][1]-poseVector2[j][1])
        if (vectorX[0] == 0 and vectorX[1] == 0) or (vectorY[0] == 0 and vectorY[1] == 0):
            continue
        cosSim = (vectorX[0]*vectorY[0]+vectorX[1]*vectorY[1])/(((vectorX[0]**2+vectorX[1]**2)**0.5)*((vectorY[0]**2+vectorY[1]**2)**0.5))
        result.append((cosSim+1)*50)
    if len(result) == 0:
        return 0
    return round(sum(result)/len(result))


def draw_keypoints(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (320, 320), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []
    for i in range(18):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > 0.1 :
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)

    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3)

    return frame, points


def read_image(file: UploadFile):
    # Yolo 로드
    net = cv2.dnn.readNet("assets/yolov3.weights", "assets/yolov3.cfg")
    classes = []
    with open("assets/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    contents = file.file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            subframe = img[max(0,y-20):min(y+h+20,height), max(0,x-20):min(x+w+20,width)]
            img[max(0,y-20):min(y+h+20,height), max(0,x-20):min(x+w+20,width)], pts = draw_keypoints(subframe)
            standing = [(61, 57), (49, 108), (19, 120), None, (95, 183), (87, 103), None, (95, 183), None, None, None, None, None, None, (53, 51), (68, 51), (133, 206), (80, 62)]
            print(f"finished {i}th box:", pts)
            point = cosine(pts, standing)

    is_success, buffer = cv2.imencode(".jpg", img)
    buf = BytesIO(buffer)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg")


def give_points(file: UploadFile):
    # Yolo 로드
    net = cv2.dnn.readNet("assets/yolov3.weights", "assets/yolov3.cfg")
    classes = []
    with open("assets/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    contents = file.file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    points = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            subframe = img[max(0,y-20):min(y+h+20,height), max(0,x-20):min(x+w+20,width)]
            img[max(0,y-20):min(y+h+20,height), max(0,x-20):min(x+w+20,width)], pts = draw_keypoints(subframe)
            standing = [(61, 57), (49, 108), (19, 120), None, (95, 183), (87, 103), None, (95, 183), None, None, None, None, None, None, (53, 51), (68, 51), (133, 206), (80, 62)]
            print(f"finished {i}th box:", pts)
            points.append(cosine(pts, standing))

    return points