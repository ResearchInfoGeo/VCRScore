
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import struct
import numpy as np
from skimage.io import imread
from skimage.color import gray2rgb
from tensorflow.image import resize
class YOLO:
    def __init__(self, file):
        self.model = load_model(file)
        self.labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
                       "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                       "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                       "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                       "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                       "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                       "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                       "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                       "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                       "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


    def get_label(self, idx):
        return self.labels[idx]
    

    def detect(self, image_file):
        im=imread(image_file)
        if len(im.shape) == 2:
            im = gray2rgb(im)
        im_h, im_w, _ = im.shape
        image_or = im
        image_h, image_w, _ = image_or.shape
        image_r = resize(image_or, (416,416)) / 255

        img = img_to_array(image_r)
        img = tf.expand_dims(img, 0)
        input_w, input_h = 416, 416
        yhat = self.model(img, training=False)
        anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
        # define the probability threshold for detected objects
        class_threshold = 0.6
        boxes = list()
        for i in range(len(yhat)):
            # decode the output of the network
            boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
        correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
        do_nms(boxes, 0.5)
        return get_boxes(boxes, self.labels, class_threshold)



# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
                # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores
        

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = tf.reshape(netout, (grid_h, grid_w, nb_box, -1)) # (h,w,3,85)
    nb_class = netout.shape[-1] - 5

    boxes = []

    netout_t1  = tf.math.sigmoid(netout[..., :2])
    netout_t2  = tf.math.sigmoid(netout[..., 4:])
    netout = tf.concat([netout_t1, netout[..., 2:4], netout_t2], axis=-1)
    netout_t2  = netout[..., 4][..., tf.newaxis] * netout[..., 5:]
    netout = tf.concat([netout[..., 0:5], netout_t2], axis=-1)
    mask = tf.cast(netout[..., 5:] > obj_thresh, tf.float32)
    netout_t2 = mask * netout[..., 5:]
    netout = tf.concat([netout[..., 0:5], netout_t2], axis=-1)
    
    
    objs = tf.where(netout[:,:,:,4] > obj_thresh)

    for row, col, b in objs:
        objectness = netout[row, col, b, 4]
        
        x, y, w, h = netout[row, col, b, :4]
        
        x = (tf.cast(col, tf.float32) + x) / grid_w # center position, unit: image width
        y = (tf.cast(row, tf.float32) + y) / grid_h # center position, unit: image height
        w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
        h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height  

        # last elements are class probabilities
        classes = netout[row, col, b, 5:]

        box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness.numpy(), classes.numpy())

        boxes.append(box)
    return boxes


def _sigmoid(x):
    return tf.math.sigmoid(x)


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.objness = objness
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score
    def __str__(self):
        return "(x:%i y:%i w:%i h:%i score:%f, class:%i)"%(self.xmin, self.ymin, self.xmax-self.xmin, self.ymax-self.ymin, 
                                                           self.get_score(), self.get_label())
    
def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h*net_w)/image_w
    else:
        new_h = net_w
        new_w = (image_w*net_h)/image_h
        
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
        

def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
        
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3     
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()