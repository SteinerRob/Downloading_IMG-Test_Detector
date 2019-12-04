import cv2
import sys
import numpy as np
import os
from pascal_voc_writer import Writer
import multiprocessing as multi
import xml.etree.ElementTree as ET
import time

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4  #Non-maximum suppression threshold

inpWidth = 416  #608     #Width of network's input image
inpHeight = 416 #608     #Height of network's input image

base_folder = '.'
models_folder = os.path.join(base_folder, 'models')
target_folder = os.path.join(base_folder, 'target')
anns_folder = os.path.join(base_folder, 'annotations')
vis_folder = os.path.join(base_folder, 'visualizations')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = os.path.join(base_folder, 'darknet-yolov3.cfg')

# Load names of classes
classesFile = os.path.join(base_folder, 'classes.names')

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

nets = []

for classID in classes:
    modelWeights = os.path.join(models_folder, '{}/final.weights'.format(classID))
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    nets.append(net)

try:
    os.remove(os.path.join(target_folder, '.DS_Store'))
except:
    pass
try:
    os.remove(os.path.join(anns_folder, '.DS_Store'))
except:
    pass

dataset_classes = [*classes]

def find_all_classes(anns):
    for ann in anns:
        ann_path = os.path.join(anns_folder, ann)
        tree = ET.parse(ann_path)
        root = tree.getroot()
        for ann in root.findall('object'):
            bbx_class = ann.find('name').text
            if bbx_class not in dataset_classes:
                dataset_classes.append(bbx_class)
    with open('dataset_classes.txt', 'w') as f:
        for dataset_class in dataset_classes:
            f.write('\'' + dataset_class + '\'\n')

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(bbxes, confidences, classes, img_id):
    # Draw a bounding box.
    #    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    frame = cv2.imread(os.path.join(target_folder, img_id))
    for i in range(len(bbxes)):
        left = bbxes[i][0]
        top = bbxes[i][1]
        right = bbxes[i][2]
        bottom = bbxes[i][3]
        classId = classes[i]
        conf = confidences[i]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if dataset_classes:
            assert(classId < len(dataset_classes))
            label = '%s:%s' % (dataset_classes[classId], label)

        #Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv2.FILLED)
        #cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
    
    return frame

def load_existing_anns(img_id):
    ann_path = os.path.join(anns_folder, img_id.split('.')[0] + '.xml')
    tree = ET.parse(ann_path)
    root = tree.getroot()       
    anns = []
    counter = 0
    bbxes = []
    confidences = []
    bbx_classes = []
    filename = root.find('filename').text   
    for ann in root.findall('object'):
        bbox = []
        bbx_class = ann.find('name').text
        for bbx in ann.find('bndbox'):
            bbox.append(int(bbx.text))
        # bbox[3] = bbox[3] - bbox[1]
        # bbox[2] = bbox[2] - bbox[0]
        bbxes.append(bbox)
        confidences.append(1)
        bbx_classes.append(dataset_classes.index(bbx_class))
    return bbxes, confidences, bbx_classes, filename

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, img_id):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    net_counter = 0
    for net_outs in outs:
        for out in net_outs:
            # print("out.shape : ", out.shape)
            for detection in out:
                #if detection[4]>0.001:
                detection = detection.clip(0, 1)

                scores = detection[5:]
                classId = np.argmax(scores)
                #if scores[classId]>confThreshold:
                confidence = scores[classId]
                if detection[4]>confThreshold:
                    pass
                    # print(detection[4], " - ", scores[classId], " - th : ", confThreshold)
                    # print(detection)
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId+net_counter)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
                    print(left, top, width, height)
        net_counter += 1

    img_path = os.path.join(target_folder, img_id)
    xml_file = os.path.join(anns_folder, img_id.split('.')[0] + '.xml')
    writer = Writer(img_path, frameWidth, frameHeight)

    ann_path = os.path.join(anns_folder, img_id.split('.')[0] + '.xml')
    if os.path.isfile(ann_path):
        existing_bbxes, existing_confidences, existing_classes, filename = load_existing_anns(img_id)
        boxes = [*boxes, *existing_bbxes]
        confidences = [*confidences, *existing_confidences]
        classIds = [*classIds, *existing_classes]


    # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        right = left+width if left+width <= frameWidth else frameWidth
        bottom = top+height if top+height <= frameHeight else frameHeight
        writer.addObject(dataset_classes[classIds[i]], left, top, right, bottom)
        # drawPred(classIds[i], confidences[i], left, top, right, bottom)
    writer.save(xml_file)

def create_ann(img_list):
    for img_id in img_list:
        img = cv2.imread(os.path.join(target_folder, img_id))
        # Create a 4D blob from a frame.
        outs = []
        blob = cv2.dnn.blobFromImage(img, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
        net_counter = 0
        for net in nets:
            # Sets the input to the network
            net.setInput(blob)
            # Runs the forward pass to get output of the output layers
            new_outs = net.forward(getOutputsNames(net))
            outs.append(new_outs)

        # Remove the bounding boxes with low confidence
        postprocess(img, outs, img_id)

def multi_create_ann(entries):
    cpus = multi.cpu_count()
    entries_parts = np.array_split(entries, cpus)

    workers = []
    for cpu in range(cpus):
        worker = multi.Process(name=str(cpu),
                               target=create_ann,
                               args=[entries_parts[cpu]],
                               )
        worker.start()
        workers.append(worker)

    for worker in workers:
        worker.join()

def visualize_annotations():
    for ann in os.listdir(anns_folder):
        bbxo, confo, klass, filename = load_existing_anns(ann)
        cv2.imwrite(os.path.join(vis_folder, filename), drawPred(bbxo, confo, klass, filename))

find_all_classes(os.listdir(anns_folder))
multi_create_ann(os.listdir(target_folder))
visualize_annotations()