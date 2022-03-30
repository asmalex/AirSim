import torch

model = torch.hub.load('ultralytics/yolov3', 'custom', 'D:\\Redei\\yolov3\\runs\\train\\exp12\\weights\\best.pt')  # custom trained model

im = 'D:\\Redei\\AirSim\\DetectionLibraries\\data\\images\\validation\\2022_03_22_21_28_13_20.jpeg'  # or file, Path, URL, PIL, OpenCV, numpy, list

# Inference
results = model(im)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

print(results.xyxy[0])  # im predictions (tensor)
print(results.pandas().xyxy[0])  # im predictions (pandas)