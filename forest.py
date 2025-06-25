from ultralytics import YOLO
import cv2

model = YOLO('best.pt')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("🔴 Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, imgsz=640, conf=0.3, verbose=False)
    annotated_frame = results[0].plot()
    cv2.namedWindow('Forest Fire Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Forest Fire Detection', 640, 480)
    cv2.imshow('Forest Fire Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()