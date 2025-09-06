from roboflow import Roboflow
rf = Roboflow(api_key="8gkTq7qdcLZKgzEMBx6U")
project = rf.workspace("majorproject-ymwew").project("monkey_bison_detection-5cx3s")
version = project.version(13)
dataset = version.download("yolov8")
                