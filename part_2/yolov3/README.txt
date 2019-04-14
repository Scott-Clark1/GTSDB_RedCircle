# BEFORE BEGINNING
  - Ensure all Python (3.6) dependencies in requirements.txt are satisfied
  - Edit all files in the folder "./rc", replacing text in curly brackets with absolute paths to training, validation and test data
  - Ensure that images and their labels are stored in the exact same parent directory, with images in {PARENT_DIR}/images and labels in {PARENT_DIR}/labels

# TO TEST THE MODEL DESCRIBED IN THE PAPER:

  - Edit "cfg/rc.data" such that the "valid" variable is set to "rc/val.txt" or "rc/test.txt", depending on your goals
  - Run "python test.py --weights weights/yolov3-tiny-best.pt"



# TO VISUALIZE RESULTS ON VALIDATION/TEST SET

  - Copy validation or test data into data/val or data/test, respectively
  - Run "python detect.py --images data/{val OR test} --weights weights/yolov3-tiny-best.pt
  - View corresponding images with bounding boxes



# TO TRAIN THE MODEL FROM PRETRAINED WEIGHTS (to replicate paper results):

python train.py --resume --epochs 150


For more information, please e-mail sclark78@gatech.edu or reference the Ultralytics GitHub, which served as the codebase for this project https://github.com/ultralytics/yolov3



Note:
Full credit to Ultralytics for providing this implementation of YOLOv3 https://github.com/ultralytics/yolov3
