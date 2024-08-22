import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import time
import numpy as np
import pandas as pd
from google.cloud import storage
import tensorflow as tf
import segmentation_models as sm
from report_generator import process_and_generate_reports
from io import BytesIO
from PIL import Image, UnidentifiedImageError

def hello_gcs(event, context):
    """Triggered by a change to a Cloud Storage bucket.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    file = event
    print(f"Processing file: {file['name']}.")

    client = storage.Client()
    source_bucket_name = event['bucket']
    file_name = event['name']
    source_bucket = client.bucket(source_bucket_name)
    model_bucket_name = 'brain-tumour-detection'
    model_bucket = client.bucket(model_bucket_name)
    os.environ["SM_FRAMEWORK"] = "tf.keras"
    
    image_path = '/tmp/image.tif'
    blob = source_bucket.blob(file_name)
    blob.download_to_filename(image_path)
    print(f"Image downloaded to {image_path}")
    model_path = '/tmp/my_model.h5'
    model_blob = model_bucket.blob('model/my_model.h5')
    model_blob.download_to_filename(model_path)
    print(f"Model downloaded to {model_path}")
    if not os.path.exists(model_path):
        print(f"Failed to download the model file {model_path}")
        return

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'binary_focal_loss_plus_dice_loss': sm.losses.binary_focal_dice_loss,
            'iou_score': sm.metrics.IOUScore,
            'f1-score': sm.metrics.FScore
        }
    )
    print("Model loaded successfully")

    try:
        process_and_generate_reports(image_path, model, 
                                     docx_report_path='/tmp/report.docx', 
                                     pdf_report_path='/tmp/report.pdf')
        print("Reports generated successfully")
    except Exception as e:
        print(f"Failed to generate reports: {str(e)}")
        return
    
    if not os.path.exists('/tmp/report.docx') or not os.path.exists('/tmp/report.pdf'):
        print("Reports not found after generation")
        return
    
    try:
        docx_blob = model_bucket.blob(f'reports/{file_name}_report.docx')
        pdf_blob = model_bucket.blob(f'reports/{file_name}_report.pdf')
        docx_blob.upload_from_filename('/tmp/report.docx')
        pdf_blob.upload_from_filename('/tmp/report.pdf')
        print(f"Uploaded report files for: {file_name}")
    except Exception as e:
        print(f"Failed to upload report files: {str(e)}")
        return

    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        archive_file_name = f'{file_name}_{timestamp}.tif'
        archive_blob = model_bucket.blob(f'archive/{archive_file_name}')
        archive_blob.upload_from_filename(image_path)
        print(f"Archived the processed file: {archive_file_name}")
    except Exception as e:
        print(f"Failed to archive the file: {str(e)}")
        return
    
    try:
        blob.delete()
        print(f"Deleted original file: {file_name}")
    except Exception as e:
        print(f"Failed to delete original file: {file_name}, error: {str(e)}")
    os.remove('/tmp/report.docx')
    os.remove('/tmp/report.pdf')
    os.remove(model_path)
    os.remove(image_path)
    print("Cleaned up temporary files")