from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)

# # Train the model
results = model.train(data="Image Processing\config.yaml", epochs=1, imgsz=640)
