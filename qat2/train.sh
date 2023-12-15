#! /bin/sh


#PARAMS_OVERRIDE="task.quantization.pretrained_original_checkpoint=/tmp/qat/mobilenetv2_ssd_i256_ckpt"
#TPU_NAME="<tpu-name>"  # The name assigned while creating a Cloud TPU.
python3 train.py \
  --experiment=retinanet_mobile_coco_qat \
  --config_file=coco_mobilenetv2_qat_tpu_e2e.yaml \
  --model_dir=model_dir \
  --mode=train \
#  --params_override="task.quantization.pretrained_original_checkpoint=mobilenetv2_ssd_i256_ckpt/ckpt-277200" \
#  --use_tpu=false \

#--tpu=$TPU_NAME \
