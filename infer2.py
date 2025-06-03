import cv2
import numpy as np
import tensorflow as tf

# ========== CONFIG ==========
TFLITE_MODEL_PATH = "best_float32.tflite"
IMAGE_PATH = "a.jpg"
CONF_THRESH = 0.0
IOU_THRESH = 0.45
INPUT_SIZE = 640  # assume model input is (640, 640)
# ============================

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize with unchanged aspect ratio using padding"""
    shape = img.shape[:2]  # current shape [height, width]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_padded, ratio, dw, dh

def nms(boxes, scores, iou_threshold):
    """Simple NMS using IoU"""
    indices = np.argsort(scores)[::-1]
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        if len(indices) == 1:
            break
        ious = compute_iou(boxes[current], boxes[indices[1:]])
        indices = indices[1:][ious < iou_threshold]
    return keep

def compute_iou(box, boxes):
    """Compute IoU between a box and a list of boxes"""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union_area = box_area + boxes_area - inter_area
    return inter_area / (union_area + 1e-6)

# === Load model ===
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Load and preprocess image ===
image = cv2.imread(IMAGE_PATH)
img_input, ratio, dw, dh = letterbox(image, (INPUT_SIZE, INPUT_SIZE))
img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
img_input = np.expand_dims(img_input, axis=0).astype(input_details[0]['dtype'])

# === Inference ===
interpreter.set_tensor(input_details[0]['index'], img_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])[0]

# === Decode + NMS ===
boxes = []
scores = []
classes = []

for pred in output:
    x, y, w, h = pred[:4]
    obj_conf = pred[4]
    cls_scores = pred[5:]
    cls_id = np.argmax(cls_scores)
    cls_conf = cls_scores[cls_id]
    conf = obj_conf * cls_conf

    if conf < CONF_THRESH:
        continue

    # Convert to xyxy and scale back
    x1 = (x - w / 2 - dw) / ratio
    y1 = (y - h / 2 - dh) / ratio
    x2 = (x + w / 2 - dw) / ratio
    y2 = (y + h / 2 - dh) / ratio
    boxes.append([x1, y1, x2, y2])
    scores.append(conf)
    classes.append(cls_id)

boxes = np.array(boxes)
scores = np.array(scores)
classes = np.array(classes)

if len(boxes) > 0:
    keep_idxs = nms(boxes, scores, IOU_THRESH)
    print("📦 Detected objects:")
    for i in keep_idxs:
        box = boxes[i].astype(int)
        print(f"🟩 Class {classes[i]} | Conf: {scores[i]:.2f} | Box: ({box[0]}, {box[1]}, {box[2]}, {box[3]})")
else:
    print("🚫 No objects detected.")
##1122