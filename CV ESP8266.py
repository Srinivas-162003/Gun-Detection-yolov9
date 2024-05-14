from ultralytics import YOLO
import cv2
import cvzone
import math
import urllib.request
import urllib.error
import http.client

root_url = "http://192.168.4.1"


def sendRequest(url):
    try:
        n = urllib.request.urlopen(url) # send request to ESP
    except urllib.error.URLError as e:
        print("Error: Failed to connect to the ESP8266:", e)
    except http.client.RemoteDisconnected as e:
        print("Error: Connection to the ESP8266 was unexpectedly closed:", e)

def ON():
    sendRequest(root_url+"/LEDON?")
    print("Led is on")

def OFF():
    sendRequest(root_url+"/LEDOFF?")
    print("Led is off")

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("C:\\Users\\Srinivas\\Downloads\\yolov9-20240427T104314Z-001\\best.pt")

classNames = ["person", "gun"]

while True:
    success, img  = cap.read()
    results = model(img, stream=True)
    name = "temp"
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0]*100))/100

            cls = box.cls[0]
            name = classNames[int(cls)]

            cvzone.putTextRect(img, f'{name} 'f'{conf}', (max(0,x1), max(35,y1)), scale = 0.5)
            
        if name == "gun":
            ON()
        else:
            OFF()


    cv2.imshow("Image", img)
    
    cv2.waitKey(1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release() 

cv2.destroyAllWindows() 