# import library
from datetime import datetime
import numpy as np
import cv2
import os
import time
import RPi.GPIO as GPIO
import mysql.connector
import time

mydb = mysql.connector.connect(
    host = 'localhost',
    user = 'absensi',
    password = '1234',
    database = 'Absensi'
)


        
#Disable warnings (optional)
GPIO.setwarnings(False)

GPIO.setmode(GPIO.BCM)
#Set servo - pin 17
servo_pin = 17
GPIO.setup(servo_pin,GPIO.OUT)
#Set buzzer - pin 27
buzzer = 27
GPIO.setup(buzzer,GPIO.OUT)

# setup PWM process
pwm = GPIO.PWM(servo_pin,50) # 50 Hz (20 ms PWM period)
pwm.start(2.5) # start PWM by rotating to 90 degrees
list_deteksi =["",""]
def extract_boxes_confidences_classids(outputs, confidence, width, height):
    boxes = []
    confidences = []
    classIDs = []
    for output in outputs:
        for detection in output:            
            # Extract the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classID = np.argmax(scores)
            conf = scores[classID]
            
            # Consider only the predictions that are above the confidence threshold
            if conf > confidence:
                # Scale the bounding box back to the size of the image
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, w, h = box.astype('int')
                # Use the center coordinates, width and height to get the coordinates of the top left corner
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(conf))
                classIDs.append(classID)
    return boxes, confidences, classIDs

def draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors):
    cv2.putText(image, 'Proses Deteksi', (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    list_name =['Donny','Ghel','Tony','Fredy']
    
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            # draw the bounding box and label on the image
            color = [int(c) for c in colors[classIDs[i]]]
            print(labels[classIDs[i]])
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            list_deteksi.append(labels[classIDs[i]])
            print(list_deteksi)
            try:
                if labels[classIDs[i]] in list_name and list_deteksi[-2]=='Use Mask':
                    cv2.putText(image, 'Silahkan Masuk dan Terapkan 3M', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    while 1:
                        mycursor = mydb.cursor()
                        mycursor.execute("Update Karyawan SET Waktu=NOW() Where Nama='"+str(labels[classIDs[i]]+"';"))
                        mydb.commit()
                        GPIO.output(buzzer,GPIO.HIGH)
                        time.sleep(0.1)
                        GPIO.output(buzzer,GPIO.LOW)
                        pwm.ChangeDutyCycle(2.5) # rotate to 0 degrees
                        time.sleep(5)
                        pwm.ChangeDutyCycle(7.5) # rotate to 90 degrees
                        time.sleep(5)
                        list_deteksi.clear()
                        break
                    pwm.ChangeDutyCycle(0) # this prevents jitter 
                else:
                    cv2.putText(image, 'Recognizing', (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            except IndexError:
                pass
                
                
                
    return image

def make_prediction(net, layer_names, labels, image, confidence, threshold):
    height, width = image.shape[:2]
    # Create a blob and pass it through the model
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)
    
    # Extract bounding boxes, confidences and classIDs
    boxes, confidences, classIDs = extract_boxes_confidences_classids(outputs, confidence, width, height)
    # Apply Non-Max Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)
    return boxes, confidences, classIDs, idxs


if __name__ == '__main__':
    tgl = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    labels = '/home/pi/Downloads/go/obj.names'
    config = '/home/pi/Downloads/go/yolov3-tiny1.cfg'
    weights = '/home/pi/Downloads/go/yolov3-tiny_final.weights'
    # Get the labels
    labels = open(labels).read().strip().split('\n')
    # Create a list of colors for the labels
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
    # Load weights using OpenCV
    net = cv2.dnn.readNetFromDarknet(config, weights)
    #net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    #net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    # Get the ouput layer names
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    tglname = tgl.replace(":","-")
    out = cv2.VideoWriter(tglname+'.avi',fourcc, 10.0, (640, 480))
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    
    while True:
	start_time = time.time()
        image = cap.read()[1]
        t1 = cv2.getTickCount()
        boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, labels, image, 0.5, 0.3)
        image = draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors)
        out.write(image)
        cv2.putText(image,'FPS: {0:.1f}'.format(frame_rate_calc),(20,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,215,255),2,cv2.LINE_AA)
        cv2.imshow('YOLO Object Detection', image)
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1
	print("--- %s seconds --- " % (time.time() - start_time))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
