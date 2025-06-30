import os
import mimetypes
import argparse
from google.cloud import documentai_v1 as documentai
from google.api_core.exceptions import InvalidArgument

# --- Configuration ---
# Replace with your project ID, location (region of your processor), and processor ID
# You can get these from the Google Cloud Console after creating a processor.
DEFAULT_PROJECT_ID = "superb-runXXXX"  # REQUIRED
DEFAULT_LOCATION = "us"  # e.g., 'us' or 'eu' - MUST match your processor's region
DEFAULT_PROCESSOR_ID = "a00897XXXX"  # REQUIRED

# Supported MIME types by this script (and common for Document AI)
# Document AI might support more, but these are the ones requested.
SUPPORTED_MIME_TYPES = {
    '.pdf': 'application/pdf',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.tiff': 'image/tiff',
    '.tif': 'image/tiff',
    '.gif': 'image/gif',  # Document AI usually processes the first frame
    '.png': 'image/png',
    '.webp': 'image/webp',
    '.bmp': 'image/bmp',
}

def get_mime_type(file_path):
    """Guesses the MIME type of a file."""
    extension = os.path.splitext(file_path)[1].lower()
    if extension in SUPPORTED_MIME_TYPES:
        return SUPPORTED_MIME_TYPES[extension]
    else:
        # Fallback to mimetypes library if not in our explicit list
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and mime_type.split('/')[0] in ['image', 'application']: # Basic check
            return mime_type
    return None

def process_document_sample(
    project_id: str,
    location: str,
    processor_id: str,
    file_path: str,
    mime_type: str,
    skip_human_review: bool = True # Set to False if you have human review configured
) -> documentai.Document | None:
    """
    Processes a document using the Document AI API.
    """
    print(f"Processing document: {file_path} with MIME type: {mime_type}")
    print(f"Using Project ID: {project_id}, Location: {location}, Processor ID: {processor_id}")

    # You must set the `api_endpoint` if you use a location other than "us".
    opts = {}
    if location != "us":
        opts["api_endpoint"] = f"{location}-documentai.googleapis.com"

    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    # The full resource name of the processor version, e.g.:
    # `projects/{project_id}/locations/{location}/processors/{processor_id}/processorVersions/{processor_version_id}`
    # If `processor_version_id` is not specified, the default version is used.
    # For this script, we'll use the default (latest active) version of the processor.
    name = client.processor_path(project_id, location, processor_id)

    # Read the file into memory
    with open(file_path, "rb") as image:
        image_content = image.read()

    # Configure the process request
    raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)
    request = documentai.ProcessRequest(
        name=name,
        raw_document=raw_document,
        skip_human_review=skip_human_review # Set to False if human review is configured
    )

    try:
        result = client.process_document(request=request)
        # We are interested in the full text, which is a top-level property of the Document object.
        # For more structured data (forms, tables), you would parse other fields of result.document
        return result.document
    except InvalidArgument as e:
        print(f"Error processing document: {e}")
        if "mime_type" in str(e).lower():
            print(f"  Hint: The MIME type '{mime_type}' might not be supported by the processor or is incorrect.")
        elif "file type" in str(e).lower():
            print(f"  Hint: The file content might not match the declared MIME type '{mime_type}'.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR/HTR a document using Google Document AI.")
    parser.add_argument("file_path", help="Path to the document file (PDF, JPG, TIFF, GIF, PNG, WEBP, BMP).")
    parser.add_argument("--project_id", default=os.environ.get("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT_ID),
                        help="Google Cloud Project ID.")
    parser.add_argument("--location", default=DEFAULT_LOCATION,
                        help="Location/Region of the Document AI processor (e.g., 'us', 'eu').")
    parser.add_argument("--processor_id", default=DEFAULT_PROCESSOR_ID,
                        help="Document AI Processor ID.")
    parser.add_argument("--output_file", help="Optional: Path to save the extracted text.")

    args = parser.parse_args()

    if args.project_id == "your-gcp-project-id" or not args.project_id:
        print("ERROR: Please set your Google Cloud Project ID in the script or via the --project_id argument.")
        exit(1)
    if args.processor_id == "your-processor-id" or not args.processor_id:
        print("ERROR: Please set your Document AI Processor ID in the script or via the --processor_id argument.")
        exit(1)

    if not os.path.exists(args.file_path):
        print(f"Error: File not found at {args.file_path}")
        exit(1)

    file_mime_type = get_mime_type(args.file_path)
    if not file_mime_type:
        print(f"Error: Could not determine a supported MIME type for {args.file_path}")
        print(f"Supported extensions: {', '.join(SUPPORTED_MIME_TYPES.keys())}")
        exit(1)

    print(f"Attempting to process: {args.file_path} (MIME: {file_mime_type})")

    document_data = process_document_sample(
        project_id=args.project_id,
        location=args.location,
        processor_id=args.processor_id,
        file_path=args.file_path,
        mime_type=file_mime_type
    )

    if document_data and document_data.text:
        print("\n--- Extracted Text ---")
        print(document_data.text)
        print("--- End of Text ---")

        if args.output_file:
            try:
                with open(args.output_file, "w", encoding="utf-8") as f:
                    f.write(document_data.text)
                print(f"\nExtracted text saved to: {args.output_file}")
            except IOError as e:
                print(f"\nError saving text to file: {e}")
    elif document_data and not document_data.text:
        print("\nDocument processed, but no text was extracted. The document might be blank or non-textual.")
    else:
        print("\nFailed to process document or extract text.")
