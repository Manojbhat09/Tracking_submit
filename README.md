# Argoverse Tracking Challenge submission

### Detection module:(Vanilla PointRCNN only available here) 

PointRCNN with Centerness loss (FCOS:Fully Convolutional One-Stage Object Detection on 3D) for robust 3D object detection. (mAP 95.12% on KITTI-Easy vs 85.95%)[To be released]  
Along with Argoverse dataloader.  
Trained seperate for Pedestrain and Vehicle classes.  

### Tracker module:

Vanilla AB3DMOT modified for PointRCNN output.  
Used mahalanobis distance and feature.  
__Note__: The repo incudes Tracker with MLP refinement in _run_ab3dmot_mod.py_ (Needs to be complete)
Pipline is Detection -> Tracker -> MLP-refine -> IDs, Locations, Size  

Without Groundtuth (Colored Trackers)  &  With Groundtuth (white + Colored Trackers):

<img src="https://github.com/Manojbhat09/Tracking_submit/blob/master/with_gt_555.gif" width="400"><img src="https://github.com/Manojbhat09/Tracking_submit/blob/master/without_gt.gif" width="400">




## In-Progress:
* PointRCNN with Centerness loss + Non-NMS regression 
* PointRCNN with Autoregressive Transformer regression
* PointRCNN with PointCNN w. knn-graph Backbone (Performs better in RPN, more backbones can be tried)
* PointRCNN with MeteorNet Tracker
* Tracker with MLP
* Tracker with LSTM 
* Tracker with PointNet local features (Points inside BBOX)
* Fusion with stereo-images(360) and then using Frustum pointnet with 2D+3D ground-truth (mAP 96.48% on KITTI-Easy)
* Fusion with range-image and PointGNN






