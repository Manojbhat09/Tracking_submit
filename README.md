# Argoverse Tracking Challenge submission

### Detection module:

PointRCNN with Centerness loss (FCOS on 3D) for robust 3D object detection. (mAP 95.12% on KITTI-Easy vs 85.95%)  
Along with Argoverse dataloader.  
Trained seperate for Pedestrain and Vehicle classes.  

### Tracker module:

Vanilla AB3DMOT modified for PointRCNN output.  
Used mahalanobis distance and feature.  
__Note__: The repo incudes Tracker with MLP refinement in _run_ab3dmot_mod.py_  
Pipline is Detection -> Tracker -> MLP-refine -> IDs, Locations, Size  

Checkout the demo video  

