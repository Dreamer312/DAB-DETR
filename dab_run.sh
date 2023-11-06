#1.
#======================================================================
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -m torch.distributed.launch --nproc_per_node=4 \
#   --use_env main.py \
#   -m dab_detr \
#   --output_dir logs/DABDETR/R50 \
#   --batch_size 4 \
#   --epochs 50 \
#   --lr_drop 40 \
#   --coco_path ../coco2017 \
#   --amp 


export CUDA_VISIBLE_DEVICES=1,2,4,5
torchrun --nproc_per_node=4 \
  --standalone \
  --nnodes=1 \
  main.py \
  -m dab_detr \
  --output_dir logs/DABDETR/R50 \
  --batch_size 4 \
  --epochs 50 \
  --lr_drop 40 \
  --random_refpoints_xy \
  --coco_path ../coco2017



#======================================================================