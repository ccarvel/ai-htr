import os
import argparse
import base64
import io
import mimetypes
from datetime import datetime 
from dotenv import load_dotenv
from PIL import Image
from pdf2image import convert_from_path

# Load environment variables from .env file
load_dotenv()

# --- Provider Definitions ---
PROVIDERS = {
    "gemini": {
        "api_key_env": "GOOGLE_API_KEY",
        "default_model": "gemini-2.0-flash", # Or "gemini-1.5-flash-latest"
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o", # Or "gpt-4o-2024-08-06", "gpt-4-turbo"
    },
    "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-3-5-sonnet-20241022", # Or other Claude 3.5/Opus/Haiku model
    }
}

# --- Helper Functions (Common - from previous script, slightly adapted) ---
def image_bytes_to_pil(image_bytes):
    return Image.open(io.BytesIO(image_bytes))

def pil_to_image_bytes(pil_image, image_format='PNG'):
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format=image_format)
    return img_byte_arr.getvalue()

def get_mime_type(image_format_ext):
    mime_type, _ = mimetypes.guess_type(f"dummy.{image_format_ext.lower()}")
    if not mime_type:
        ext_upper = image_format_ext.upper()
        if ext_upper in ['JPEG', 'JPG']: mime_type = 'image/jpeg'
        elif ext_upper == 'PNG': mime_type = 'image/png'
        elif ext_upper == 'WEBP': mime_type = 'image/webp'
        elif ext_upper == 'GIF': mime_type = 'image/gif'
        elif ext_upper in ['TIFF', 'TIF']: mime_type = 'image/tiff'
        else: mime_type = 'application/octet-stream'
    return mime_type

# --- Gemini Specific Functions ---
def get_gemini_model_instance(api_key, model_name): # Renamed for clarity
    import google.generativeai as genai
    if not api_key: raise ValueError("GOOGLE_API_KEY not found for Gemini.")
    genai.configure(api_key=api_key)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    generation_config = genai.types.GenerationConfig(temperature=0.1)
    return genai.GenerativeModel(model_name, safety_settings=safety_settings, generation_config=generation_config)

def extract_text_gemini(image_pil_or_bytes, prompt_text, model_instance):
    try:
        img_pil = image_bytes_to_pil(image_pil_or_bytes) if isinstance(image_pil_or_bytes, bytes) else image_pil_or_bytes
        response = model_instance.generate_content([prompt_text, img_pil], stream=False)
        response.resolve()
        return response.text.strip() if response.parts else "No text found (Gemini)."
    except Exception as e: return f"Gemini API Error: {e}"

# --- OpenAI Specific Functions ---
def get_openai_client_instance(api_key): # Renamed
    from openai import OpenAI as OpenAIClient
    if not api_key: raise ValueError("OPENAI_API_KEY not found for OpenAI.")
    return OpenAIClient(api_key=api_key)

def image_to_base64_data_uri(image_bytes, image_format_ext='PNG'):
    return f"data:{get_mime_type(image_format_ext)};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

def extract_text_openai(image_bytes, prompt_text, client_instance, model_name, image_format_ext='PNG'):
    base64_image = image_to_base64_data_uri(image_bytes, image_format_ext)
    try:
        response = client_instance.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {"url": base64_image, "detail": "high"}}]}],
            max_tokens=4000, temperature=0.1
        )
        return response.choices[0].message.content.strip() if response.choices and response.choices[0].message else "No text found (OpenAI)."
    except Exception as e: return f"OpenAI API Error: {e}"

# --- Anthropic Specific Functions ---
def get_anthropic_client_instance(api_key): # Renamed
    from anthropic import Anthropic as AnthropicClient
    if not api_key: raise ValueError("ANTHROPIC_API_KEY not found for Anthropic.")
    return AnthropicClient(api_key=api_key)

def image_to_anthropic_image_source(image_bytes, image_format_ext='PNG'):
    return {"type": "base64", "media_type": get_mime_type(image_format_ext), "data": base64.b64encode(image_bytes).decode('utf-8')}

def extract_text_anthropic(image_bytes, prompt_text, client_instance, model_name, image_format_ext='PNG'):
    image_source = image_to_anthropic_image_source(image_bytes, image_format_ext)
    try:
        message = client_instance.messages.create(
            model=model_name, max_tokens=4000, temperature=0.1,
            messages=[{"role": "user", "content": [{"type": "image", "source": image_source}, {"type": "text", "text": prompt_text}]}]
        )
        return message.content[0].text.strip() if message.content and message.content[0].type == "text" else "No text found (Anthropic)."
    except Exception as e: return f"Anthropic API Error: {e}"

# --- Generic File Processors ---
def process_image_file_with_provider(image_path, prompt, provider_name, client_or_model_obj, model_name_for_api):
    try:
        with open(image_path, "rb") as f: image_bytes = f.read()
        _, ext = os.path.splitext(image_path)
        image_format_ext = ext.lstrip('.').upper()
        if image_format_ext == "JPG": image_format_ext = "JPEG"
        if image_format_ext == "TIF": image_format_ext = "TIFF"

        if provider_name == "gemini":
            return extract_text_gemini(image_bytes, prompt, client_or_model_obj)
        elif provider_name == "openai":
            return extract_text_openai(image_bytes, prompt, client_or_model_obj, model_name_for_api, image_format_ext)
        elif provider_name == "anthropic":
            return extract_text_anthropic(image_bytes, prompt, client_or_model_obj, model_name_for_api, image_format_ext)
        return "Invalid provider for image extraction."
    except FileNotFoundError: return f"Error: Image file not found at {image_path}"
    except Image.UnidentifiedImageError: return f"Error: Cannot identify image file: {image_path}"
    except Exception as e: return f"Error processing image file {image_path} with {provider_name}: {e}"

def process_pdf_file_with_provider(pdf_path, prompt, provider_name, client_or_model_obj, model_name_for_api, poppler_path=None):
    all_page_texts = []
    pdf_page_img_fmt = 'PNG'
    try:
        if os.name == 'nt' and poppler_path:
            page_pil_images = convert_from_path(pdf_path, poppler_path=poppler_path, fmt=pdf_page_img_fmt.lower())
        else:
            page_pil_images = convert_from_path(pdf_path, fmt=pdf_page_img_fmt.lower())
        if not page_pil_images: return "Could not convert PDF to images."

        for i, page_pil_image in enumerate(page_pil_images):
            print(f"  Processing PDF page {i + 1}/{len(page_pil_images)} with {provider_name}...")
            page_text = ""
            if provider_name == "gemini":
                page_text = extract_text_gemini(page_pil_image, prompt, client_or_model_obj)
            elif provider_name in ["openai", "anthropic"]:
                img_bytes = pil_to_image_bytes(page_pil_image, format=pdf_page_img_fmt)
                if provider_name == "openai":
                    page_text = extract_text_openai(img_bytes, prompt, client_or_model_obj, model_name_for_api, pdf_page_img_fmt)
                elif provider_name == "anthropic":
                    page_text = extract_text_anthropic(img_bytes, prompt, client_or_model_obj, model_name_for_api, pdf_page_img_fmt)
            all_page_texts.append(f"--- Page {i + 1} ---\n{page_text}")
        return "\n\n".join(all_page_texts)
    except FileNotFoundError: return f"Error: PDF file not found at {pdf_path}"
    except Exception as e: return f"Error processing PDF {pdf_path} with {provider_name}: {e}"

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract text from an image or PDF file using available AI providers based on .env configuration, or a specified provider."
    )
    parser.add_argument("file_path", help="Path to the image (e.g., .png, .jpg, .tif) or PDF (.pdf) file.")
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()),
        default=None, # No default, will auto-detect
        help="Specify a single AI provider. If not set, uses all configured providers."
    )
    parser.add_argument(
        "--prompt",
        default="Extract all text content from this document. Preserve formatting like line breaks where possible. Provide only the extracted text.",
        help="The prompt to use for text extraction."
    )
    parser.add_argument(
        "--model",
        default=None,
        help=("Name of the model. Applies if --provider is set, or if only one provider is auto-detected. "
              "Ignored if multiple providers are auto-detected (uses defaults).")
    )
    parser.add_argument(
        "--poppler_path",
        default=None,
        help="[Windows Only] Path to the Poppler 'bin' directory if not in system PATH."
    )
    args = parser.parse_args()

    # Determine active providers and models to use
    providers_to_use_config = {} # Stores {provider_name: {"api_key": key, "model": model_name, "client_or_model_obj": None}}

    if args.provider: # User specified a single provider
        provider_name = args.provider
        config = PROVIDERS.get(provider_name)
        api_key = os.getenv(config["api_key_env"])
        if not api_key:
            print(f"Error: API key for specified provider '{provider_name}' ({config['api_key_env']}) not found in .env or environment.")
            exit(1)
        providers_to_use_config[provider_name] = {
            "api_key": api_key,
            "model": args.model if args.model else config["default_model"],
            "client_or_model_obj": None
        }
        print(f"Using specified provider: {provider_name} with model {providers_to_use_config[provider_name]['model']}")
    else: # Auto-detect providers
        active_auto_providers = []
        for name, config in PROVIDERS.items():
            api_key = os.getenv(config["api_key_env"])
            if api_key:
                active_auto_providers.append(name)
                providers_to_use_config[name] = {
                    "api_key": api_key,
                    "model": config["default_model"], # Start with default
                    "client_or_model_obj": None
                }
        
        if not active_auto_providers:
            print("Error: No API keys found in .env for any provider (GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY).")
            print("Please configure at least one or use --provider to specify one if its key is set elsewhere.")
            exit(1)

        # Handle --model argument in auto-detect mode
        if args.model:
            if len(active_auto_providers) == 1:
                provider_name = active_auto_providers[0]
                providers_to_use_config[provider_name]["model"] = args.model
                print(f"Auto-detected single provider: {provider_name}. Using specified model: {args.model}")
            else:
                print(f"Auto-detected multiple providers: {', '.join(active_auto_providers)}. "
                      "The --model argument ('{args.model}') is ignored; using default models for each. "
                      "To specify a model, use --provider <name> --model <model_name>.")
        else:
            if active_auto_providers:
                 print(f"Auto-detected providers: {', '.join(active_auto_providers)}. Using default models.")


    if not providers_to_use_config:
        print("Error: No providers selected or configured. Exiting.")
        exit(1)

    # Initialize clients/models for selected providers
    for name, config_data in providers_to_use_config.items():
        try:
            if name == "gemini":
                config_data["client_or_model_obj"] = get_gemini_model_instance(config_data["api_key"], config_data["model"])
            elif name == "openai":
                config_data["client_or_model_obj"] = get_openai_client_instance(config_data["api_key"])
            elif name == "anthropic":
                config_data["client_or_model_obj"] = get_anthropic_client_instance(config_data["api_key"])
            if not config_data["client_or_model_obj"]:
                raise Exception("Client/model object is None after initialization attempt.")
        except ImportError as ie:
            print(f"Import Error for {name}: {ie}. Is the library '{name if name != 'gemini' else 'google-generativeai'}' installed?")
            # Remove provider if client init fails due to import error
            providers_to_use_config.pop(name)
            continue # to next provider initialization
        except Exception as e:
            print(f"Error initializing client/model for provider '{name}' with model '{config_data['model']}': {e}")
            print(f"Provider '{name}' will be skipped.")
            providers_to_use_config.pop(name) # Remove provider if client init fails
            continue

    if not providers_to_use_config: # All providers failed to initialize
        print("No providers could be initialized successfully. Exiting.")
        exit(1)

    if not os.path.exists(args.file_path):
        print(f"Error: File not found at {args.file_path}")
        exit(1)

    file_ext = os.path.splitext(args.file_path)[1].lower()
    supported_image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tif', '.tiff']
    all_results = {} # Store results: {provider_name: extracted_text}

    # Process file with each configured and initialized provider
    for provider_name, config_data in providers_to_use_config.items():
        print(f"\n--- Processing with: {provider_name.upper()} (Model: {config_data['model']}) ---")
        client_or_model_obj = config_data["client_or_model_obj"]
        model_name_for_api = config_data["model"] # This is the actual model string passed to API
        extracted_text = ""

        if file_ext in supported_image_extensions:
            # ... (image processing logic) ...
            extracted_text = process_image_file_with_provider(
                args.file_path, args.prompt, provider_name, client_or_model_obj, model_name_for_api
            )
        elif file_ext == '.pdf':
            # ... (pdf processing logic) ...
            extracted_text = process_pdf_file_with_provider(
                args.file_path, args.prompt, provider_name, client_or_model_obj, model_name_for_api, args.poppler_path
            )
        else:
            print(f"Unsupported file type for provider {provider_name}: {file_ext}. Skipping.")
            all_results[provider_name] = f"Unsupported file type: {file_ext}"
            continue
        
        all_results[provider_name] = extracted_text
        print(f"\n--- Extracted Text (using {provider_name.upper()}) ---")
        print(extracted_text)

        # --- MODIFIED SECTION FOR FILENAME ---
        current_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_name = os.path.splitext(os.path.basename(args.file_path))[0]
        output_filename = f"{base_name}_{provider_name}_extracted_{current_time_str}.txt"
        # --- END OF MODIFIED SECTION ---
        
        try:
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(extracted_text)
            print(f"Extracted text from {provider_name.upper()} saved to: {output_filename}")
        except Exception as e:
            print(f"Error saving extracted text from {provider_name.upper()} to file {output_filename}: {e}")

    print("\n--- Processing Complete ---")
    if not all_results:
        print("No results were generated.")