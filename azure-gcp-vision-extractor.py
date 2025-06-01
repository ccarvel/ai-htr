# pip install google-cloud-vision azure-cognitiveservices-vision-computervision Pillow pdf2image

# Linux: sudo apt-get install poppler-utils
# macOS: brew install poppler

# Linux: sudo apt-get install libwebp-dev
# macOS: brew install webp

import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime # Added for timestamping

# Google Cloud Vision
from google.cloud import vision
from google.protobuf.json_format import MessageToDict

# Azure Computer Vision
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

# PDF to Image and Image Handling
from pdf2image import convert_from_path # For PDF to image conversion
from PIL import Image # For image operations
import io

# --- Configuration ---
SUPPORTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.webp']
SUPPORTED_PDF_EXTENSION = '.pdf'

# --- Helper Functions ---
def ensure_dir(directory):
    """Ensures a directory exists, creates it if not."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def save_results(output_base_filepath_stem, text_content, json_content): # Renamed arg for clarity
    """Saves text to .txt and JSON data to .json, using the provided stem."""
    txt_path = f"{output_base_filepath_stem}.txt"
    json_path = f"{output_base_filepath_stem}.json"

    with open(txt_path, 'w', encoding='utf-8') as f_txt:
        f_txt.write(text_content)
    print(f"Saved text to: {txt_path}")

    with open(json_path, 'w', encoding='utf-8') as f_json:
        json.dump(json_content, f_json, indent=4, ensure_ascii=False)
    print(f"Saved JSON to: {json_path}")

def convert_pil_image_to_bytes(pil_image, img_format="PNG"):
    """Converts a PIL Image object to bytes. Used by pdf_to_images flow."""
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format=img_format)
    return img_byte_arr.getvalue()

# --- Google Cloud Vision Functions ---
def get_google_vision_client(credentials_path):
    """Initializes and returns a Google Vision client."""
    try:
        client = vision.ImageAnnotatorClient.from_service_account_json(credentials_path)
        print("Google Vision client initialized successfully.")
        return client
    except Exception as e:
        print(f"Error initializing Google Vision client: {e}")
        print("Ensure your GOOGLE_APPLICATION_CREDENTIALS environment variable is set or the path is correct.")
        return None

def ocr_google_image_bytes(client, image_bytes, original_filename="image"):
    """Performs OCR on image bytes using Google Vision."""
    if not client: return None, None
    try:
        image = vision.Image(content=image_bytes)
        response = client.document_text_detection(image=image)
        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))
        text = response.full_text_annotation.text if response.full_text_annotation else ""
        return text, MessageToDict(response._pb)
    except Exception as e:
        print(f"Google Vision API error processing {original_filename}: {e}")
        return None, None

def ocr_google_pdf(client, pdf_path):
    """Performs OCR on a PDF using Google Vision (direct PDF processing)."""
    if not client: return None, None
    try:
        with open(pdf_path, 'rb') as f:
            pdf_content = f.read()
        input_config = vision.InputConfig(content=pdf_content, mime_type='application/pdf')
        features = [vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)]
        request = vision.AnnotateFileRequest(input_config=input_config, features=features)
        operation = client.async_batch_annotate_files(requests=[request])
        pdf_filename = Path(pdf_path).name
        print(f"Waiting for Google PDF processing operation to complete for {pdf_filename}...")
        operation_result = operation.result(timeout=600)
        all_text = ""
        full_json_response_pages = []
        if operation_result.responses and len(operation_result.responses) > 0:
            file_response = operation_result.responses[0]
            for i, image_response in enumerate(file_response.responses):
                page_text = ""
                if image_response.full_text_annotation:
                    page_text = image_response.full_text_annotation.text
                    all_text += page_text + "\n\n--- Page Break ---\n\n"
                page_json_dict = MessageToDict(image_response._pb)
                page_json_dict['page_number_from_ocr_iteration'] = i + 1
                if hasattr(image_response, 'context') and image_response.context and hasattr(image_response.context, 'page_number'):
                     page_json_dict['page_number_from_api'] = image_response.context.page_number
                full_json_response_pages.append(page_json_dict)
        else:
            print(f"No responses found in Google PDF processing result for {pdf_filename}.")
            return None, {"error": f"No responses from Google PDF processing for {pdf_filename}"}
        if all_text: # Remove trailing page break if text was added
            all_text = all_text.strip()
            if all_text.endswith("--- Page Break ---"): # Be more precise
                 all_text = all_text[:-len("--- Page Break ---")].strip()

        return all_text, {"pages": full_json_response_pages, "source_file": pdf_filename}
    except Exception as e:
        print(f"Google Vision API error (PDF '{Path(pdf_path).name}'): {e}")
        return None, None

# --- Azure Computer Vision Functions ---
def get_azure_vision_client(credentials_path):
    """Initializes and returns an Azure Computer Vision client."""
    try:
        with open(credentials_path, 'r') as f:
            creds = json.load(f)
        endpoint = creds['endpoint']
        key = creds['key']
        client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))
        print("Azure Computer Vision client initialized successfully.")
        return client
    except FileNotFoundError:
        print(f"Azure credentials file not found at {credentials_path}")
        return None
    except KeyError:
        print(f"Azure credentials file {credentials_path} is missing 'endpoint' or 'key'.")
        return None
    except Exception as e:
        print(f"Error initializing Azure client: {e}")
        return None

def ocr_azure_read_stream(client, stream, original_filename="stream"):
    """Performs OCR on an image/PDF stream using Azure's Read API."""
    if not client: return None, None
    try:
        if hasattr(stream, 'seek') and callable(stream.seek):
            stream.seek(0)
        read_response = client.read_in_stream(stream, raw=True)
        operation_location = read_response.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]
        print(f"Azure Read operation submitted for {original_filename}. Waiting for results...")
        while True:
            read_result = client.get_read_result(operation_id)
            if read_result.status not in ['notStarted', 'running']:
                break
            time.sleep(2)
        text_content = ""
        if read_result.status == OperationStatusCodes.succeeded:
            for text_result in read_result.analyze_result.read_results:
                for line in text_result.lines:
                    text_content += line.text + "\n"
                text_content += "\n--- Page Break ---\n\n"
            if text_content: # Remove trailing page break
                text_content = text_content[:-len("\n\n--- Page Break ---\n\n")].strip()
            return text_content, read_result.as_dict()
        else:
            error_message = f"Azure Read API failed for {original_filename}: {read_result.status}"
            error_details = read_result.as_dict()
            if hasattr(read_result, 'message') and read_result.message:
                 error_message += f" - Message: {read_result.message}"
            if read_result.analyze_result and hasattr(read_result.analyze_result, 'errors') and read_result.analyze_result.errors:
                error_message += f" - API Errors: {read_result.analyze_result.errors}"
            print(error_message)
            return None, {"error": error_message, "details": error_details}
    except Exception as e:
        print(f"Azure Read API error for {original_filename}: {e}")
        return None, None

# --- Main Processing Logic ---
def process_file(file_path_obj, output_dir, api_choice, google_client, azure_client, force_pdf_to_images=False):
    """Processes a single file (image or PDF) for OCR."""
    file_path_str = str(file_path_obj) # Keep string version for some operations
    print(f"\nProcessing file: {file_path_str}")
    
    original_filename_stem = file_path_obj.stem
    file_ext = file_path_obj.suffix.lower()
    
    is_pdf = file_ext == SUPPORTED_PDF_EXTENSION
    is_image = file_ext in SUPPORTED_IMAGE_EXTENSIONS

    if not (is_pdf or is_image):
        print(f"Unsupported file type: {file_ext} for {file_path_str}. Skipping.")
        return

    text_content = None
    json_response = None

    if is_pdf and not force_pdf_to_images:
        print(f"Processing PDF directly: {file_path_obj.name}")
        if api_choice == "google":
            if google_client:
                text_content, json_response = ocr_google_pdf(google_client, file_path_str)
        elif api_choice == "azure":
            if azure_client:
                with open(file_path_str, "rb") as f_pdf:
                    text_content, json_response = ocr_azure_read_stream(azure_client, f_pdf, original_filename=file_path_obj.name)
        else:
            print(f"Unknown API choice: {api_choice}")
            return
    
    elif is_pdf and force_pdf_to_images:
        print(f"Converting PDF to images first: {file_path_obj.name}")
        try:
            images_from_path = convert_from_path(file_path_str)
            all_text_parts = []
            all_json_responses_parts = []
            for i, pil_image in enumerate(images_from_path):
                page_num = i + 1
                print(f"Processing page {page_num} of PDF {file_path_obj.name} as image...")
                img_bytes = convert_pil_image_to_bytes(pil_image, img_format="PNG") 
                page_text, page_json = None, None
                page_filename_for_api = f"{original_filename_stem}_page_{page_num}.png"
                if api_choice == "google":
                    if google_client:
                        page_text, page_json = ocr_google_image_bytes(google_client, img_bytes, original_filename=page_filename_for_api)
                elif api_choice == "azure":
                    if azure_client:
                        page_text, page_json = ocr_azure_read_stream(azure_client, io.BytesIO(img_bytes), original_filename=page_filename_for_api)
                if page_text: all_text_parts.append(page_text)
                if page_json: all_json_responses_parts.append({"page": page_num, "response": page_json})
            
            text_content = "\n\n--- Page Break ---\n\n".join(all_text_parts)
            if text_content: # Remove trailing page break
                text_content = text_content.strip()
                if text_content.endswith("--- Page Break ---"):
                     text_content = text_content[:-len("--- Page Break ---")].strip()
            json_response = {"pages": all_json_responses_parts, "source_file": file_path_obj.name, "method": "pdf_converted_to_images"}
        except Exception as e:
            print(f"Error converting PDF {file_path_str} to images or processing pages: {e}")
            return

    elif is_image:
        print(f"Processing image file: {file_path_obj.name}")
        try:
            with open(file_path_str, "rb") as f_img:
                image_bytes = f_img.read()
            if api_choice == "google":
                if google_client:
                    text_content, json_response = ocr_google_image_bytes(google_client, image_bytes, original_filename=file_path_obj.name)
            elif api_choice == "azure":
                if azure_client:
                    text_content, json_response = ocr_azure_read_stream(azure_client, io.BytesIO(image_bytes), original_filename=file_path_obj.name)
            else:
                print(f"Unknown API choice: {api_choice}")
                return
        except FileNotFoundError:
            print(f"Image file not found: {file_path_str}")
            return
        except Exception as e:
            print(f"Error reading or processing image {file_path_str}: {e}")
            return

    if text_content is not None and json_response is not None:
        # Generate timestamp string: YYYYMMDD-HHMMSS
        timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Construct new output filename stem
        output_filename_base = f"{original_filename_stem}_{api_choice}_{timestamp_str}"
        
        # Create full path for saving (without extension, save_results adds it)
        output_path_stem_for_saving = Path(output_dir) / output_filename_base
        
        save_results(str(output_path_stem_for_saving), text_content, json_response)
    else:
        print(f"Failed to OCR {file_path_str} with {api_choice} API. No results to save.")


def main():
    parser = argparse.ArgumentParser(
        description="OCR/HTR PDFs and images (PNG, JPG, BMP, TIFF, GIF, WebP) using Google or Azure Vision APIs.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_path", help="Path to an input image/PDF file or a directory of files.")
    parser.add_argument("output_dir", help="Directory to save the .txt and .json results.")
    parser.add_argument("--api", choices=["google", "azure"], required=True, help="Choose API: 'google' or 'azure'.")
    parser.add_argument("--google_creds", default="google_credentials.json", help="Path to Google Cloud credentials JSON file (default: google_credentials.json).")
    parser.add_argument("--azure_creds", default="azure_credentials.json", help="Path to Azure credentials JSON file (default: azure_credentials.json).")
    parser.add_argument("--force_pdf_to_images", action="store_true",
                        help="Force PDF files to be converted to images page-by-page before OCR,\n"
                             "instead of using the API's direct PDF processing capabilities.\n"
                             "Note: This requires 'poppler' for pdf2image.")
    args = parser.parse_args()
    ensure_dir(args.output_dir)
    google_client = None
    azure_client = None

    if args.api == "google":
        google_client = get_google_vision_client(args.google_creds)
        if not google_client: print("Exiting due to Google client initialization failure."); return
    elif args.api == "azure":
        azure_client = get_azure_vision_client(args.azure_creds)
        if not azure_client: print("Exiting due to Azure client initialization failure."); return

    input_path_obj = Path(args.input_path)
    files_to_process = []
    if input_path_obj.is_file():
        if input_path_obj.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS or input_path_obj.suffix.lower() == SUPPORTED_PDF_EXTENSION:
            files_to_process.append(input_path_obj)
        else:
            print(f"Input file {args.input_path} is not a supported type. Supported: {SUPPORTED_IMAGE_EXTENSIONS + [SUPPORTED_PDF_EXTENSION]}")
    elif input_path_obj.is_dir():
        print(f"Scanning directory: {args.input_path}")
        for p in input_path_obj.rglob('*'):
            if p.is_file() and (p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS or p.suffix.lower() == SUPPORTED_PDF_EXTENSION):
                files_to_process.append(p)
    else:
        print(f"Error: Input path {args.input_path} is not a valid file or directory."); return

    if not files_to_process:
        print(f"No supported files (PDFs or {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}) found in {args.input_path}."); return
    
    print(f"Found {len(files_to_process)} file(s) to process.")
    for file_to_process_obj in files_to_process:
        process_file(file_to_process_obj, args.output_dir, args.api, google_client, azure_client, args.force_pdf_to_images)
    print("\nProcessing complete.")

if __name__ == "__main__":
    # Optional: Check for Pillow features at startup (informational)
    try:
        from PIL import features
        for feature_name, check_func_name in [('webp', 'webp'), ('gif', 'gif')]:
            if not features.check(check_func_name):
                print(f"Note: Pillow {feature_name.upper()} support might not be fully available or configured (e.g., libwebp for WebP). "
                      "This script sends raw bytes for images, so API support is key. "
                      "Pillow's capabilities are more relevant for --force_pdf_to_images.")
    except ImportError:
        print("Warning: Pillow library not found. It is required for --force_pdf_to_images and some image operations.")
        pass 
    main()
 


# import argparse
# import json
# import os
# import time
# from pathlib import Path

# # Google Cloud Vision
# from google.cloud import vision
# from google.protobuf.json_format import MessageToDict

# # Azure Computer Vision
# from azure.cognitiveservices.vision.computervision import ComputerVisionClient
# from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
# from msrest.authentication import CognitiveServicesCredentials

# # PDF to Image and Image Handling
# from pdf2image import convert_from_path # For PDF to image conversion
# from PIL import Image # For image operations, including handling WebP/GIF if needed via pdf2image
# import io

# # --- Configuration ---
# SUPPORTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.webp']
# SUPPORTED_PDF_EXTENSION = '.pdf'

# # --- Helper Functions ---
# def ensure_dir(directory):
#     """Ensures a directory exists, creates it if not."""
#     Path(directory).mkdir(parents=True, exist_ok=True)

# def save_results(output_base_path, text_content, json_content):
#     """Saves text to .txt and JSON data to .json."""
#     txt_path = f"{output_base_path}.txt"
#     json_path = f"{output_base_path}.json"

#     with open(txt_path, 'w', encoding='utf-8') as f_txt:
#         f_txt.write(text_content)
#     print(f"Saved text to: {txt_path}")

#     with open(json_path, 'w', encoding='utf-8') as f_json:
#         json.dump(json_content, f_json, indent=4, ensure_ascii=False)
#     print(f"Saved JSON to: {json_path}")

# def convert_pil_image_to_bytes(pil_image, img_format="PNG"):
#     """Converts a PIL Image object to bytes. Used by pdf_to_images flow."""
#     img_byte_arr = io.BytesIO()
#     pil_image.save(img_byte_arr, format=img_format)
#     return img_byte_arr.getvalue()

# # --- Google Cloud Vision Functions ---
# def get_google_vision_client(credentials_path):
#     """Initializes and returns a Google Vision client."""
#     try:
#         client = vision.ImageAnnotatorClient.from_service_account_json(credentials_path)
#         print("Google Vision client initialized successfully.")
#         return client
#     except Exception as e:
#         print(f"Error initializing Google Vision client: {e}")
#         print("Ensure your GOOGLE_APPLICATION_CREDENTIALS environment variable is set or the path is correct.")
#         return None

# def ocr_google_image_bytes(client, image_bytes, original_filename="image"):
#     """Performs OCR on image bytes using Google Vision."""
#     if not client: return None, None
#     try:
#         image = vision.Image(content=image_bytes)
#         response = client.document_text_detection(image=image) # document_text_detection is good for dense text
#         # response = client.text_detection(image=image) # For sparse text

#         if response.error.message:
#             raise Exception(
#                 '{}\nFor more info on error messages, check: '
#                 'https://cloud.google.com/apis/design/errors'.format(
#                     response.error.message))
        
#         text = response.full_text_annotation.text if response.full_text_annotation else ""
#         return text, MessageToDict(response._pb)
#     except Exception as e:
#         print(f"Google Vision API error processing {original_filename}: {e}")
#         return None, None

# def ocr_google_pdf(client, pdf_path):
#     """Performs OCR on a PDF using Google Vision (direct PDF processing)."""
#     if not client: return None, None
#     try:
#         with open(pdf_path, 'rb') as f:
#             pdf_content = f.read()

#         input_config = vision.InputConfig(
#             content=pdf_content, mime_type='application/pdf'
#         )
#         features = [vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)]
        
#         request = vision.AnnotateFileRequest(
#             input_config=input_config,
#             features=features,
#         )
        
#         operation = client.async_batch_annotate_files(requests=[request])
#         pdf_filename = Path(pdf_path).name
#         print(f"Waiting for Google PDF processing operation to complete for {pdf_filename}...")
        
#         # Timeout needs to be generous for large PDFs
#         # You might need to increase this for very large or many-paged PDFs
#         operation_result = operation.result(timeout=600) 

#         all_text = ""
#         full_json_response_pages = []

#         # Correctly access responses after operation.result()
#         # The structure is operation_result.responses[0].responses (for a single file request)
#         if operation_result.responses and len(operation_result.responses) > 0:
#             file_response = operation_result.responses[0]
#             for i, image_response in enumerate(file_response.responses):
#                 page_text = ""
#                 if image_response.full_text_annotation:
#                     page_text = image_response.full_text_annotation.text
#                     all_text += page_text + "\n\n--- Page Break ---\n\n"
                
#                 # Convert each page's protobuf response to dict
#                 page_json_dict = MessageToDict(image_response._pb)
#                 page_json_dict['page_number_from_ocr'] = i + 1 # Add page number if available (usually context.page_number)
#                 if hasattr(image_response, 'context') and image_response.context and hasattr(image_response.context, 'page_number'):
#                      page_json_dict['page_number_from_api'] = image_response.context.page_number
#                 full_json_response_pages.append(page_json_dict)
#         else:
#             print(f"No responses found in Google PDF processing result for {pdf_filename}.")
#             return None, {"error": f"No responses from Google PDF processing for {pdf_filename}"}

#         return all_text.strip(), {"pages": full_json_response_pages, "source_file": pdf_filename}

#     except Exception as e:
#         print(f"Google Vision API error (PDF '{Path(pdf_path).name}'): {e}")
#         return None, None


# # --- Azure Computer Vision Functions ---
# def get_azure_vision_client(credentials_path):
#     """Initializes and returns an Azure Computer Vision client."""
#     try:
#         with open(credentials_path, 'r') as f:
#             creds = json.load(f)
#         endpoint = creds['endpoint']
#         key = creds['key']
#         client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))
#         print("Azure Computer Vision client initialized successfully.")
#         return client
#     except FileNotFoundError:
#         print(f"Azure credentials file not found at {credentials_path}")
#         return None
#     except KeyError:
#         print(f"Azure credentials file {credentials_path} is missing 'endpoint' or 'key'.")
#         return None
#     except Exception as e:
#         print(f"Error initializing Azure client: {e}")
#         return None

# def ocr_azure_read_stream(client, stream, original_filename="stream"):
#     """Performs OCR on an image/PDF stream using Azure's Read API."""
#     if not client: return None, None
#     try:
#         if hasattr(stream, 'seek') and callable(stream.seek):
#             stream.seek(0)

#         read_response = client.read_in_stream(stream, raw=True)
#         operation_location = read_response.headers["Operation-Location"]
#         operation_id = operation_location.split("/")[-1]

#         print(f"Azure Read operation submitted for {original_filename}. Waiting for results...")
#         while True:
#             read_result = client.get_read_result(operation_id)
#             if read_result.status not in ['notStarted', 'running']:
#                 break
#             time.sleep(2)

#         text_content = ""
#         if read_result.status == OperationStatusCodes.succeeded:
#             for text_result in read_result.analyze_result.read_results: # Iterates over pages
#                 for line in text_result.lines:
#                     text_content += line.text + "\n"
#                 text_content += "\n--- Page Break ---\n\n" # Add page break after each page's content
#             # Remove last page break if text_content is not empty
#             if text_content:
#                 text_content = text_content[:-len("\n\n--- Page Break ---\n\n")]

#             return text_content.strip(), read_result.as_dict()
#         else:
#             error_message = f"Azure Read API failed for {original_filename}: {read_result.status}"
#             error_details = read_result.as_dict()
#             if hasattr(read_result, 'message') and read_result.message: # Older SDK versions might have this
#                  error_message += f" - Message: {read_result.message}"
#             # For newer SDKs, error details might be in analyze_result if status is 'failed'
#             if read_result.analyze_result and hasattr(read_result.analyze_result, 'errors') and read_result.analyze_result.errors:
#                 error_message += f" - API Errors: {read_result.analyze_result.errors}"

#             print(error_message)
#             return None, {"error": error_message, "details": error_details}

#     except Exception as e:
#         print(f"Azure Read API error for {original_filename}: {e}")
#         return None, None


# # --- Main Processing Logic ---
# def process_file(file_path_obj, output_dir, api_choice, google_client, azure_client, force_pdf_to_images=False):
#     """Processes a single file (image or PDF) for OCR."""
#     file_path = str(file_path_obj)
#     print(f"\nProcessing file: {file_path}")
#     base_name = file_path_obj.stem
#     output_base = Path(output_dir) / base_name

#     file_ext = file_path_obj.suffix.lower()
    
#     is_pdf = file_ext == SUPPORTED_PDF_EXTENSION
#     is_image = file_ext in SUPPORTED_IMAGE_EXTENSIONS

#     if not (is_pdf or is_image):
#         print(f"Unsupported file type: {file_ext} for {file_path}. Skipping.")
#         return

#     text_content = None
#     json_response = None

#     if is_pdf and not force_pdf_to_images:
#         print(f"Processing PDF directly: {file_path_obj.name}")
#         if api_choice == "google":
#             if google_client:
#                 text_content, json_response = ocr_google_pdf(google_client, file_path)
#         elif api_choice == "azure":
#             if azure_client:
#                 with open(file_path, "rb") as f_pdf:
#                     text_content, json_response = ocr_azure_read_stream(azure_client, f_pdf, original_filename=file_path_obj.name)
#         else:
#             print(f"Unknown API choice: {api_choice}")
#             return
    
#     elif is_pdf and force_pdf_to_images:
#         print(f"Converting PDF to images first: {file_path_obj.name}")
#         try:
#             # Note: For WebP support with Pillow, ensure libwebp is installed.
#             # `sudo apt-get install libwebp-dev` or `brew install webp`
#             images_from_path = convert_from_path(file_path)
#             all_text_parts = []
#             all_json_responses_parts = []

#             for i, pil_image in enumerate(images_from_path):
#                 page_num = i + 1
#                 print(f"Processing page {page_num} of PDF {file_path_obj.name} as image...")
#                 # Convert PIL image to bytes (e.g., PNG, as it's lossless and widely supported)
#                 img_bytes = convert_pil_image_to_bytes(pil_image, img_format="PNG") 
#                 page_text, page_json = None, None
#                 page_filename_for_api = f"{file_path_obj.stem}_page_{page_num}.png"

#                 if api_choice == "google":
#                     if google_client:
#                         page_text, page_json = ocr_google_image_bytes(google_client, img_bytes, original_filename=page_filename_for_api)
#                 elif api_choice == "azure":
#                     if azure_client:
#                         page_text, page_json = ocr_azure_read_stream(azure_client, io.BytesIO(img_bytes), original_filename=page_filename_for_api)
                
#                 if page_text:
#                     all_text_parts.append(page_text)
#                 if page_json:
#                     all_json_responses_parts.append({"page": page_num, "response": page_json})
            
#             text_content = "\n\n--- Page Break ---\n\n".join(all_text_parts)
#             json_response = {"pages": all_json_responses_parts, "source_file": file_path_obj.name, "method": "pdf_converted_to_images"}

#         except Exception as e:
#             print(f"Error converting PDF {file_path} to images or processing pages: {e}")
#             return

#     elif is_image:
#         print(f"Processing image file: {file_path_obj.name}")
#         # For GIF, this will typically process the first frame.
#         # For WebP, ensure Pillow (if used implicitly by libraries) or the direct API call supports it.
#         # Both Google and Azure Vision APIs support GIF and WebP natively when sent as bytes.
#         try:
#             with open(file_path, "rb") as f_img:
#                 image_bytes = f_img.read()
            
#             if api_choice == "google":
#                 if google_client:
#                     text_content, json_response = ocr_google_image_bytes(google_client, image_bytes, original_filename=file_path_obj.name)
#             elif api_choice == "azure":
#                 if azure_client:
#                     text_content, json_response = ocr_azure_read_stream(azure_client, io.BytesIO(image_bytes), original_filename=file_path_obj.name)
#             else:
#                 print(f"Unknown API choice: {api_choice}")
#                 return
#         except FileNotFoundError:
#             print(f"Image file not found: {file_path}")
#             return
#         except Exception as e:
#             print(f"Error reading or processing image {file_path}: {e}")
#             return


#     if text_content is not None and json_response is not None:
#         save_results(output_base, text_content, json_response)
#     else:
#         print(f"Failed to OCR {file_path} with {api_choice} API. No results to save.")


# def main():
#     parser = argparse.ArgumentParser(
#         description="OCR/HTR PDFs and images (PNG, JPG, BMP, TIFF, GIF, WebP) using Google or Azure Vision APIs.",
#         formatter_class=argparse.RawTextHelpFormatter # Allows for better formatting of help text
#     )
#     parser.add_argument("input_path", help="Path to an input image/PDF file or a directory of files.")
#     parser.add_argument("output_dir", help="Directory to save the .txt and .json results.")
#     parser.add_argument("--api", choices=["google", "azure"], required=True, help="Choose API: 'google' or 'azure'.")
#     parser.add_argument("--google_creds", default="google_credentials.json", help="Path to Google Cloud credentials JSON file (default: google_credentials.json).")
#     parser.add_argument("--azure_creds", default="azure_credentials.json", help="Path to Azure credentials JSON file (default: azure_credentials.json).")
#     parser.add_argument("--force_pdf_to_images", action="store_true",
#                         help="Force PDF files to be converted to images page-by-page before OCR,\n"
#                              "instead of using the API's direct PDF processing capabilities.\n"
#                              "Note: This requires 'poppler' for pdf2image.\n"
#                              "For WebP support via Pillow (used in this mode), ensure 'libwebp' is installed \n"
#                              "(e.g., 'sudo apt-get install libwebp-dev' or 'brew install webp').")

#     args = parser.parse_args()

#     # Friendly reminder for Pillow WebP support if --force_pdf_to_images is used (though not directly for WebP input)
#     # This is more relevant if pdf2image tried to output to WebP, which it isn't here.
#     # The main point is that Pillow needs libwebp to *open* WebP files if it were to do so,
#     # but here we send bytes directly to the APIs for input .webp files.
#     # The note in help text for --force_pdf_to_images is more about general Pillow dependencies.

#     ensure_dir(args.output_dir)

#     google_client = None
#     azure_client = None

#     if args.api == "google":
#         google_client = get_google_vision_client(args.google_creds)
#         if not google_client:
#             print("Exiting due to Google client initialization failure.")
#             return
#     elif args.api == "azure":
#         azure_client = get_azure_vision_client(args.azure_creds)
#         if not azure_client:
#             print("Exiting due to Azure client initialization failure.")
#             return

#     input_path_obj = Path(args.input_path)
#     files_to_process = []

#     if input_path_obj.is_file():
#         if input_path_obj.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS or input_path_obj.suffix.lower() == SUPPORTED_PDF_EXTENSION:
#             files_to_process.append(input_path_obj)
#         else:
#             print(f"Input file {args.input_path} is not a supported type. Supported: {SUPPORTED_IMAGE_EXTENSIONS + [SUPPORTED_PDF_EXTENSION]}")
#     elif input_path_obj.is_dir():
#         print(f"Scanning directory: {args.input_path}")
#         for p in input_path_obj.rglob('*'):
#             if p.is_file() and (p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS or p.suffix.lower() == SUPPORTED_PDF_EXTENSION):
#                 files_to_process.append(p)
#     else:
#         print(f"Error: Input path {args.input_path} is not a valid file or directory.")
#         return

#     if not files_to_process:
#         print(f"No supported files (PDFs or {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}) found in {args.input_path}.")
#         return
    
#     print(f"Found {len(files_to_process)} file(s) to process.")

#     for file_to_process_obj in files_to_process:
#         process_file(file_to_process_obj, args.output_dir, args.api, google_client, azure_client, args.force_pdf_to_images)

#     print("\nProcessing complete.")

# if __name__ == "__main__":
#     # Add a note about Pillow's WebP dependency if it's relevant upon import
#     try:
#         from PIL import features
#         if not features.check('webp'):
#             print("Note: Pillow WebP support (libwebp) might not be installed. "
#                   "This is needed if Pillow needs to open/process WebP files (e.g., for resizing before OCR, "
#                   "or if pdf2image were to output WebP, which it doesn't here). "
#                   "For direct WebP file input to APIs, this script sends raw bytes, which should work if APIs support it.")
#         if not features.check('gif'):
#             print("Note: Pillow GIF support might not be fully available.")
#     except ImportError:
#         pass # Pillow not installed, script will fail earlier for pdf2image

#     main()

# python azure-gcp-vision-extractor.py my_input_folder my_output_folder --api google
# python azure-gcp-vision-extractor.py my_image.webp my_output_folder --api azure