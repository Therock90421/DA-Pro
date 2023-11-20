# RN50
python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/PascalVOC-Detection/da_pro_faster_rcnn_R_50_C4_s2c.yaml \
MODEL.WEIGHTS ./put/your/fine-tuned/model/here \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS .put/your/offline/rpn/model/here \
MODEL.CLIP.TEXT_EMB_PATH .put/your/class/embedding/here \
MODEL.ROI_HEADS.SOFT_NMS_ENABLED True \
LEARNABLE_PROMPT.CTX_SIZE 8 \
LEARNABLE_PROMPT.TUNING True


