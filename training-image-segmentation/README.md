# Training Image Segmentation Models

## Intro

Create an image segmentation model using DeepLabV3+. This model is lightweight and works well. This code is based on the (Keras tutorial)[https://keras.io/examples/vision/deeplabv3_plus/] for image segmentation.

### Setup

1. Clone repo if you haven't. Navigate to the `training-image-segmentation` folder.
1. Install dependencies. The file `mask.py` will be used at the end to remove a background from an image and inpaint it with another background using stable diffusion.

    ```bash
    pip install gdown opencv-python scipy matplotlib tensorflow diffusers transformers
    ```

1. Download the data.

    ```bash
    gdown "1B9A9UCJYMwTL4oBEo4RZfbMZMaZhKJaz"
    unzip -q instance-level-human-parsing.zip
    ```

1. Copy the data to gcs. We'll set up some environment variables to use in the next steps. Replace the project values with yours.

    ```bash
    PROJECT_BUCKET_NAME=<your-bucket-name> #Ex: jfacevedo-demos-bucket
    PROJECT_ID=<your-project-id>
    REGION=us-central1
    gsutil -m cp -r instance-level_human_parsing gs://$PROJECT_BUCKET_NAME/datasets/segmentation_data/
    ```

1. Build the training image and push it.

    ```bash
    docker build . -t gcr.io/$PROJECT_ID/image_segmentation_train:latest
    docker push gcr.io/$PROJECT_ID/image_segmentation_train:latest
    ```

1. Run the training job. Here we will use 1/3 of the data to train and 4 T4 GPUs. This will take about 2.5 hours to train and gets good results.

    ```bash
    python gcp_deploy.py --project-id $PROJECT_ID --accelerator-count 4 --image-uri gcr.io/$PROJECT_ID/image_segmentation_train:latest --gcs-datadir /gcs/$PROJECT_BUCKET_NAME/datasets/segmentation_dataset/instance-level_human_parsing/instance-level_human_parsing/Training --num-train-images 10000 --num-eval-images 500 --model-output-dir gs://$PROJECT_BUCKET_NAME/models/segmentation/ --batch-size 32
    ```

    The final metrics should look like `"
312/312 [==============================] - 351s 1s/step - loss: 0.2066 - accuracy: 0.9360 - val_loss: 0.6632 - val_accuracy: 0.8335
"`

1. After the job completes, copy the final model to this directory.

    ```bash
    gsutil -m cp -r gs://$PROJECT_BUCKET_NAME/models/segmentation .
    ```

1. Run the inference script. The script creates a black and white mask to isolate humans (white) from anything else (black). If you don't have a GPU you can comment out lines `80` downwards which runs stable diffusion inpainting.

    ```bash
    python mask.py
    ```

1. Mask.py runs the segmentation model and then uses cv2's findContours function to try to fill in the parts of the human the model didn't mask. As we can see, the mask is not perfect, but it works. 

    <center>
        <image src="./images/merged.png" width="256px">
        <p>Original Image</p>
        <image src="./images/mask.png" width="256px">
        <p>Mask</p>
        <image src="./images/final.png" width="256px">
        <p>prompt: RAW photo, a photograph of a beach, ocean, sunset, highly detailed, close up shot.</p>
    </center>