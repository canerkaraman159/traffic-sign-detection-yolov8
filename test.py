import cv2
from ultralytics import YOLO

# modeli yukle
model = YOLO("runs/detect/traffic-sign-model-4/weights/best.pt")

# test edilecek gorselin yuklenmesi
image_path = "test1.jpg"
image = cv2.imread(image_path)

# image tahmini
results = model(image_path)[0]
print(results)

# kutu cizimi
for box in results.boxes:

    # koordinatlar
    x1, y1, x2, y2 = map(int, box.xyxy[0]) # kose koordinatlar
    cls_id = int(box.cls[0]) # classification id
    confidence = float(box.conf[0]) # guven seviye
    label = f"{model.names[cls_id]} conf: {confidence:.2f}" # detection label

    # kutu ciz
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # etiketi image uzerine ekle
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# kaydet
cv2.imwrite("prediction_result.jpg", image)