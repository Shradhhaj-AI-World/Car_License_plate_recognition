import numpy as np
import argparse
import time
import cv2
import os
from google.cloud import vision
from google.cloud.vision_v1 import types
from PIL import Image
import io
# import pytesseract

confthres=0.5
nmsthres=0.1
yolo_path="./"

class License_plate():
    def __init__(self,labelsPath,cfgpath,wpath):

        self.labelsPath=labelsPath
        self.cfgpath=cfgpath
        self.wpath=wpath
        self.Lables=self.get_labels(self.labelsPath)
        self.CFG=self.get_config(self.cfgpath)
        self.Weights=self.get_weights(self.wpath)
        self.nets=self.load_model(self.CFG,self.Weights)
        self.Colors=self.get_colors(self.Lables)

    def get_labels(self,labels_path):
        lpath=os.path.sep.join([yolo_path, labels_path])
        LABELS = open(lpath).read().strip().split("\n")
        return LABELS

    def get_colors(self,LABELS):
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
        return COLORS

    def get_weights(self,weights_path):
        weightsPath = os.path.sep.join([yolo_path, weights_path])
        return weightsPath

    def get_config(self,config_path):
        configPath = os.path.sep.join([yolo_path, config_path])
        return configPath

    def load_model(self,configpath,weightspath):
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
        return net

    def get_predection(self,image,net,LABELS,COLORS):
        (H, W) = image.shape[:2]

        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                    swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        print(layerOutputs)
        end = time.time()

        print("[INFO] YOLO took {:.6f} seconds".format(end - start))
        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
            for detection in output:               
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]               
                if confidence > confthres:                    
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")                  
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
            
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,nmsthres)
        if len(idxs) > 0:            
            for i in idxs.flatten():                
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                cv2.rectangle(image, (x, y), (x + w, y + h), [int(c) for c in COLORS[classIDs[i]]], 2)
                subImg = image[y : y + h, x : x + w, :]
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                print(boxes)
                print(classIDs)
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, [int(c) for c in COLORS[classIDs[i]]], 2)
        return image,subImg

    def google(self,content):
        try:
            client = vision.ImageAnnotatorClient()
            image = types.Image(content=content)
            # Performs label detection on the image file
            response = client.text_detection(image=image,image_context={"language_hints": ["en"]})
            labels = response.text_annotations[0].description
            print(labels)
            return labels
        except Exception as e:
            labels='Not Detected'
            return labels

    def main(self,image_path):
        image = cv2.imread(image_path)
        detected_image,croped_image=self.get_predection(image,self.nets,self.Lables,self.Colors)
      
        pilImage = Image.fromarray(croped_image)
        ext = image_path.split('.')[-1]
        b = io.BytesIO()
        if ext.lower() in ('png'):
            save_ext = 'PNG'
        elif ext.lower() in ('jpg', 'jpeg'):
            save_ext = 'JPEG'
        pilImage.save(b, save_ext) 
        text = self.google(b.getvalue())
        # cv2.imshow("croped_image", croped_image)
        # cv2.imwrite("a11.jpg",croped_image)
        # cv2.waitKey()
        # text = pytesseract.image_to_string(pilImage, lang = 'eng')
        return detected_image,text


if __name__ == "__main__":
    lp=License_plate()
    path="static/a11.jpg"
    image,text=lp.main(path)
    print(text)
    cv2.imshow("Image", image)
    cv2.waitKey()