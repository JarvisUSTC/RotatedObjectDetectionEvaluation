# RotatedObjectDetectionEvaluation
## Summary

This repo is evaluation codes for rotated object detection which uses polygon representation (x1,y1,x2,y2...,x4,y4) for object, mainly based on Deformable DETR.

## Why I create this repo

I know in detectron2 RotatedCOCOEvaluator can satisfy almost all cases for rotated object detection. But it has two limitations. **First, it only supports angle representation for rotated rectangle**, like (cx,cy,w,h,\theta). But in some cases, the groundtruth is not rotated rectangle and we need polygon representation. **Secondly, in some times we need to discuss the testing data using data augmentations,** such as rotated data augmentation to explore the model generality. But these evaluation methods I can find are for no change testing dataset. So I rewrite the COCO evaluator to support polygon representation and rewrite the evaluation method of DeformableDetr to support data augmentation for testing data.

## Usage

- dataset/Rotatedcoco.py rewrite **COCO**
- dataset/rotated_coco_eval.py rewrite two class — **CocoEvaluator and CocoEval**
- dataset/transforms.py is for data augmentation
- dataset/coco.py needs to be inited for your own data format.
- In engine.py, First store all testing data after augmentations and all prediction results. Then create RotatedCOCOEvaluator using the stored testing data and prediction results to evaluate.

## Link

https://github.com/fundamentalvision/Deformable-DETR

https://github.com/cocodataset/cocoapi

![Evaluator.png](https://github.com/JarvisUSTC/RotatedObjectDetectionEvaluation/blob/main/Evaluator.png?raw=true)