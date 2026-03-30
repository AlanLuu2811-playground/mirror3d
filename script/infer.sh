export ROBOTFLOW_VERSION=11
export DATADIR=./data/Mirror-Glass_Segmentation.v${ROBOTFLOW_VERSION}i.coco-segmentation

# mirror3dnet on raw sensor depth 
python mirror3d/mirror3dnet/run_mirror3dnet.py \
--eval \
--refined_depth \
--config mirror3d/mirror3dnet/config/mirror3dnet_config.yml \
--resume_checkpoint_path ./output/mirror_glass_segm_v11/m3n_full_rawD_resume_2026-03-28-19-07-19/model_0001999.pth \
--coco_val ${DATADIR}/valid_annot.json \
--coco_val_root ${DATADIR} \
--coco_focal_len 524 \
--mesh_depth \
--depth_shift 1000 \
--input_height 360 \
--input_width 640 \
--batch_size 1 \
--anchor_normal_npy ${DATADIR}/kmeans_normal_10.npy \
--log_directory ./output/eval/mirror_glass_segm_v${ROBOTFLOW_VERSION} \
--ref_mode rawD_border