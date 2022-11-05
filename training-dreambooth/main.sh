#!/bin/bash
echo $INSTANCE_PROMPT
python3 train.py --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=images \
    --output_dir=/tmp/sd-model-output \
    --resolution=$RESOLUTION \
    --train_batch_size=$BATCH_SIZE \
    --learning_rate=$LEARNING_RATE \
    --max_train_steps=$MAX_TRAIN_STEPS \
    --class_data_dir=class_images \
    --with_prior_preservation \
    --num_class_images=$NUM_CLASS_IMAGES \
    --prior_loss_weight=$PRIOR_LOSS_WEIGHT \
    --mixed_precision=bf16 \
    --hub_token=$HF_TOKEN