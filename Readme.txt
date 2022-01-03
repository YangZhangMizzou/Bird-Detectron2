Required environment:
numpy,
pytorch,
detectron,
pandas,
cv2

Folders:
images: images for inference. ground truth json file can also be saved here.
models: detection models(FasterRCNN-FPN) trained by different dataset.
pretrained_weight: Pretrained_weight of coco dataset.

Example:
python image_inference.py -p /home/yangzhang/Desktop/image_inference_software/images/label_by_model/ -t 0.5
-p: path of image folder
-t confidence threshold

output:
Labeled images and prediction json file will be saved in a seperated folder in image folder. If ground truth file is provided, Evaluation result will be saved as a txt file.
