from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="traffic-sign-detection/data.yaml",
        epochs=5,
        imgsz=640,
        batch=16,
        name="traffic-sign-model",
        device=0
    )

if __name__ == '__main__':
    main()