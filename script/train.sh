export ROBOTFLOW_VERSION=11
export DATADIR=./data/Mirror-Glass_Segmentation.v${ROBOTFLOW_VERSION}i.coco-segmentation

# mirror3dnet on raw sensor depth 
python mirror3d/mirror3dnet/run_mirror3dnet.py \
--config mirror3d/mirror3dnet/config/mirror3dnet_config.yml \
--resume_checkpoint_path ./checkpoint/R-50.pkl \
--coco_train ${DATADIR}/train_annot.json \
--coco_val ${DATADIR}/valid_annot.json \
--coco_train_root ${DATADIR} \
--coco_val_root ${DATADIR} \
--coco_focal_len 524 \
--depth_shift 1000 \
--input_height 360 \
--input_width 640 \
--batch_size 16 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--anchor_normal_npy ${DATADIR}/kmeans_normal_10.npy \
--log_directory ./output/mirror_glass_segm_v${ROBOTFLOW_VERSION} \
--eval_save_depth \
--ref_mode rawD_border