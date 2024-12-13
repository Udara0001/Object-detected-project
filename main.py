import cv2

# Initialize camera
cap = cv2.VideoCapture(0)  # Use 0 if you have one camera
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height
cap.set(10, 70)   # Brightness

# Load class names

classnames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')

# Load model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Real-time object detection
while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classnames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
