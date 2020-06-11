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

Useful commands:  
CUDA_VISIBLE_DEVICES=2,3 python train_rcnn.py --cfg_file cfgs/argo_config_sampling_trainfull_peds.yaml --batch_size 16 --train_mode rpn --epochs 200 --mgpus  
CUDA_VISIBLE_DEVICES=0 python train_rcnn.py --cfg_file cfgs/argo_config_sampling_trainfull_peds.yaml --batch_size 1 --train_mode rpn --epochs 200  
CUDA_VISIBLE_DEVICES=3 python eval_rcnn.py --eval_mode=rpn --rpn_ckpt="/home/jupyter/peds_prcnn/PointRCNN-Argoverse/output/rpn/peds1single/ckpt/checkpoint_epoch_1.pth" --save_result --save_rpn_feature  
python run_ab3dmot_dl.py --dets_dataroot=../PointRCNN-Argoverse/test_detects/ --pose_dir=../argoverse-tracking --split=test --tracks_dump_dir=test_prcnn  --plot --mine  
python run_ab3dmot_mod.py --dets_dataroot=../PointRCNN-Argoverse/test_detects/ --pose_dir=../argoverse-tracking --split=test --tracks_dump_dir=test_prcnn --tag="train_1" --plot --mine  
python run_ab3dmot_dl_peds.py --dets_dataroot=../test_detects --dets_peds_dataroot=../test_detects_peds --pose_dir=../argoverse-tracking --split=test --tracks_dump_dir=test_prcnn --tag="train_2" --plot --mine  

