#!/usr/bin/env python3

import os
import numpy as np
import time
import cv2
import dlr
import threading



ms = lambda: int(round(time.time() * 1000))

test_image = "dog.jpg"
dshape = (1, 3, 512, 512)
dtype = "float32"
w = 1280
h = 720
result = []




# Preprocess image
def open_and_norm_image(frame):
    
    #orig_img = cv2.imread(f)
    orig_img = frame
    img = cv2.resize(orig_img, (dshape[2], dshape[3]))
    img = img[:, :, (2, 1, 0)].astype(np.float32)
    img -= np.array([123, 117, 104])
    img = np.transpose(np.array(img), (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return orig_img, img


model_path = "models/mxnet-ssd-mobilenet-512"


class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

######################################################################
# Create TVM runtime and do inference

# Build TVM runtime
device = 'opencl'
m = dlr.DLRModel(model_path, device)



cap = cv2.VideoCapture(8)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

def thread_job():
    device = 'opencl'
    m = dlr.DLRModel(model_path, device)
    while True:
        ret, test_image = cap.read()
        if(ret == False):
            continue
        orig_img, img_data = open_and_norm_image(test_image)
        input_data = img_data.astype(dtype)

        m_out =  m.run(input_data)
        out = m_out[0][0]
        i = 0
        print('------------')
        for det in out:
            cid = int(det[0])
            if cid < 0:
                continue
            score = det[1]
            if score < 0.5:
                continue
            i += 1
            if(i>10):
                break
            print(i, class_names[cid], det)
        print('---end---------')
        if(i>10):
            continue
        global result
        result = out
        #cv2.imshow('Single-Threaded Detection',test_image)

def display(frame, out, thresh=0.5):
    pens = dict()
    for det in out:
        cid = int(det[0])
        if cid < 0:
            continue
        score = det[1]
        if score < thresh:
            continue
        scales = [frame.shape[1], frame.shape[0]] * 1
        (left, right, top, bottom) = (det[2] * w, det[4] * w,det[3] * h, det[5] * h)
        p1 = (int(left), int(top))
        p2 = (int(right), int(bottom))
        cv2.rectangle(frame, p1, p2, (77, 255, 9), 3, 1)
        cv2.putText(frame, class_names[cid], (int(left + 10), int((top+bottom)/2)), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
        #frame = cv2.resize(frame,((int)(w*0.6),(int)(h*0.6)))
    cv2.imshow('Single-Threaded Detection',frame)


added_thread = threading.Thread(target=thread_job)
    # 執行 thread
added_thread.start() 


while True:
    ret, test_image = cap.read()
    if(ret == False):
        continue
    #print(ret)
    #orig_img, img_data = open_and_norm_image(test_image)
    #cv2.imshow('Single-Threaded Detection',test_image)
    global result
    display(test_image, result, 0.5)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cv2.waitKey(0)
cv2.destroyAllWindows()



