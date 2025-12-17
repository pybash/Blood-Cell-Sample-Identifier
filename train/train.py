from io import TextIOWrapper
import os
from datetime import datetime
from ultralytics import YOLO

# Load the YOLOv11n model 
model = YOLO("yolov11n.pt")

##############################
### Data validation checks ###
##############################

# Check if "./datasets" directory exists and create it if not
if not os.path.exists("./datasets"):
    os.makedirs("./datasets")
    print("Created './datasets' directory.")

# Check if the data root directory exists
data_root = "./current_dataset"
if not os.path.exists(data_root):
    raise FileNotFoundError(f"The specified data root directory '{data_root}' does not exist.")
    exit(-1)

# Check if the images directory exists
images_dir = os.path.join(data_root, "images")
if not os.path.exists(images_dir):
    raise FileNotFoundError(f"The images directory '{images_dir}' does not exist.")
    exit(-1)

# Check if the train images are present
train_images_dir = os.path.join(images_dir, "train")
if not os.path.exists(train_images_dir) or len(os.listdir(train_images_dir)) == 0:
    raise FileNotFoundError(f"No training images found in '{train_images_dir}'.")
    exit(-1)

# Check if the value images are present
val_images_dir = os.path.join(images_dir, "val")
if not os.path.exists(val_images_dir) or len(os.listdir(val_images_dir)) == 0:
    raise FileNotFoundError(f"No validation images found in '{val_images_dir}'.")
    exit(-1)

###############################
##### Copy dataset files ######
###############################

# Copy data into a dated folder inside "./datasets"
date_now = datetime.now().strftime("%Y%m%d_%H%M%S")
dataset_dest = os.path.join("./datasets", f"dataset_{date_now}")
os.makedirs(dataset_dest, exist_ok=True)

with open("config.yaml", "r") as src_file:
    io = src_file.read()
    io = io.replace("<<SETPATH>>", dataset_dest)
    with open(os.path.join(dataset_dest, "config.yaml"), "w") as dest_file:
        dest_file.write(io)

# Train the model using a custom dataset
model.train(data=os.path.join(dataset_dest, "config.yaml"), epochs=100, imgsz=640)