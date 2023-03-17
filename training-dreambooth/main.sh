#!/bin/bash

echo "model name: $MODEL_NAME"
echo "resolution: $RESOLUTION"
echo "batch size : $BATCH_SIZE"
echo "learning rate : $LEARNING_RATE"
echo "max train steps : $MAX_TRAIN_STEPS"
echo "num class images : $NUM_CLASS_IMAGES"
echo "prior loss weight: $PRIOR_LOSS_WEIGHT"
echo "HF TOKEN: $HF_TOKEN"
echo "class-prompt : $CLASS_PROMPT"
echo "instance-prompt : $INSTANCE_PROMPT"
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
    --hub_token=$HF_TOKEN \
    --train_text_encoder