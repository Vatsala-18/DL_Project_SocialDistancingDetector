import sklearn
import numpy as np
import streamlit as st
from configs import config
from configs.detection import detect_people
from scipy.spatial import distance as dist 
import argparse
import imutils
import cv2
import os

def streamlit_interface(myVid):
    st.title("My_Distance_Detection_App")
    st.video("show.mp4")

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, default="", help="path to (optional) input video file")
    ap.add_argument("-o", "--output", type=str, default="", help="path to (optional) output video file")
    ap.add_argument("-d", "--display", type=int, default=1, help="whether or not output frame should be displayed")
    args = vars(ap.parse_args())

    labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
    configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    ln = net.getUnconnectedOutLayersNames()

    print("[INFO] accessing video stream...")
    
    vs = cv2.VideoCapture(myVid)
    writer = None
    
    stframe = st.empty()

    while vs.isOpened():
        print("hello")
        ret, frame = vs.read()
     
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stframe.image(gray)
        
        while True:
            ret, frame = vs.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            stframe.image(gray)
           
            frame = imutils.resize(frame, width=700)
            results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
                
            violate = set()
            
            if len(results) >= 2:
                centroids = np.array([r[2] for r in results])
                D = dist.cdist(centroids, centroids, metric="euclidean")

                for i in range(0, D.shape[0]):
                    for j in range(i+1, D.shape[1]):
                        if D[i, j] < config.MIN_DISTANCE:
                            violate.add(i)
                            violate.add(j)

            for (i, (prob, bbox, centroid)) in enumerate(results):
                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid
                color = (0, 255, 0)

                if i in violate:
                    color = (0, 0, 255)

                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.circle(frame, (cX, cY), 5, color, 1)

            text = "Social Distancing Violations: {}".format(len(violate))
            cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

            if args["display"] > 0:
                cv2.imshow("Output", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

            if args["output"] != "" and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)

            if writer is not None:
                print("[INFO] writing stream to output")
                writer.write(frame)    

if __name__=="__main__":

    my_dataset = st.sidebar.selectbox("Select Video",("sample1.mp4","sample2.mp4","sample3.mp4"))
    
    if my_dataset == "sample1.mp4":
        streamlit_interface("sample1.mp4")
    elif my_dataset == "sample2.mp4":
        streamlit_interface("sample2.mp4")
    elif my_dataset == "sample3.mp4":
        streamlit_interface("sample3.mp4")