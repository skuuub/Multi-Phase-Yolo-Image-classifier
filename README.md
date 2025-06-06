# Multi-Phase-Yolo-Image-classifier
A two-stage, real-time video pipeline that first uses YOLOv8 to detect and crop persons, then applies Deep SORT for persistent tracking so each individual is only processed once, and finally runs a second YOLO model to classify their clothing—saving each new person’s image along with a detailed breakdown of apparel counts and confidence scores.
