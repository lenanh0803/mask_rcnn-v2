# Mask_rcnn-v2
Instance Segmentation is a confluence of two issues: Object Detection and Semantic Segmentation. In this project, I will investigate the Mask-RCNN object detector using Pytorch. The pretrained Mask-RCNN model with Resnet50 as the backbone will be used.

# Recognizing model inputs and outputs
The pretrained Mask-RCNN ResNet-50 that I will use wants the input image tensor to be of the type [n, c, h, w] with:
- n is the number of images
- c is the number of channels , for RGB images it is 3
- h is the height of the image
- w is the widht of the image
The model will return:
- boxes (Tensor[N, 4])
- labels (Tensor[N])
- scores (Tensor[N])
- masks (Tensor[N, H, W])

# Instance segmentation pipeline
I define three util functions used for model inference:
- get_colored_mask get the colored mask for a specific class label in the image
- get_prediction take the img_path, and confidence as input, and returns predicted bounding boxes, classes, and masks.
- segment_instance uses the get_prediction function and gives the visualization result.

# The predictions
As we can see from the figure, the Mask-RCNN model shares the same structure with Fast-RCNN but Faster R-CNN has 2 outputs for each candidate object, a class label and a bounding-box offset, Mask R-CNN is the addition of a third branch that outputs the object mask. 
![image](https://user-images.githubusercontent.com/70872369/202607121-93bed954-f137-475a-877a-142d7f9b57cc.png)
