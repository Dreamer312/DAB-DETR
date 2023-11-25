
#==================================================
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# torchrun --nproc_per_node=4 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_vg_main.py \
#             --dataset vg \
#             --img_folder /home/cmh/cmh/projects/detrs/RelTR/data/vg/images/ \
#             --ann_path /home/cmh/cmh/projects/detrs/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=4 \
#             --output_dir ./checkpoint_dab_vg \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \

export CUDA_VISIBLE_DEVICES=5
torchrun --nproc_per_node=1 \
         --standalone \
         --nnodes=1 \
         cmh_dab_vg_main.py \
            --dataset vg \
            --img_folder /home/cmh/cmh/projects/detrs/RelTR/data/vg/images/ \
            --ann_path /home/cmh/cmh/projects/detrs/RelTR/data/vg/ \
            --modelname dab_detr \
            --batch_size=1 \
            --output_dir ./checkpoint_dab_vg \
            --epochs 50 \
            --lr_drop 40 \
            --random_refpoints_xy \
            --eval \
            --resume /home/cmh/cmh/projects/detrs/DAB_DETR/checkpoint_dab_vg/checkpoint0039.pth
#==================================================