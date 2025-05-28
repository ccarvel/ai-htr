# AI HTR TEXT EXTRACTOR

# Multi-Provider AI Text Extractor

A Python utility that extracts text from images and PDF files using multiple AI providers (Google Gemini, OpenAI GPT, and Anthropic Claude).

## Features

- **Multi-Provider Support**: Automatically detects and uses available AI providers based on your API keys
- **Flexible File Support**: Handles images (PNG, JPG, JPEG, WEBP, BMP, GIF, TIF, TIFF) and PDF files
- **Customizable Prompts**: Specify custom extraction prompts for different use cases
- **Model Selection**: Choose specific models for each provider
- **Auto-Save**: Automatically saves extracted text to timestamped files
- **PDF Multi-Page**: Processes all pages in PDF files with per-page extraction

## Prerequisites

### Required Python Packages

```bash
pip install python-dotenv pillow pdf2image google-generativeai openai anthropic
```

### Additional Requirements

- **For PDF processing**: Install Poppler utilities
  - **Windows**: Download Poppler and set `--poppler_path` or add to PATH
  - **macOS**: `brew install poppler`
  - **Linux**: `sudo apt-get install poppler-utils`

### API Keys Setup

Create a `.env` file in the same directory with your API keys:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

You only need to provide keys for the providers you want to use.

## Usage

### Basic Usage

```bash
# Extract text using all available providers
python multi_extractor2.py document.pdf

# Extract text from an image
python multi_extractor2.py image.png
```

### Advanced Usage

```bash
# Use a specific provider
python multi_extractor2.py document.pdf --provider openai

# Use a specific model
python multi_extractor2.py image.jpg --provider gemini --model gemini-1.5-flash-latest

# Custom extraction prompt
python multi_extractor2.py receipt.png --prompt "Extract all prices and item names from this receipt"

# Windows with custom Poppler path
python multi_extractor2.py document.pdf --poppler_path "C:\poppler\bin"
```

## Command Line Arguments

- `file_path`: Path to the image or PDF file (required)
- `--provider`: Specify AI provider (`gemini`, `openai`, `anthropic`)
- `--prompt`: Custom extraction prompt (default: general text extraction)
- `--model`: Specific model name (when using single provider)
- `--poppler_path`: Path to Poppler bin directory (Windows only)

## Default Models

- **Gemini**: `gemini-2.0-flash`
- **OpenAI**: `gpt-4o`
- **Anthropic**: `claude-3-5-sonnet-20241022`

## Output

The tool creates timestamped text files for each provider:
- Format: `{filename}_{provider}_extracted_{timestamp}.txt`
- Example: `document_gemini_extracted_20241225-143022.txt`

## Supported File Types

### Images
- PNG, JPG/JPEG, WEBP, BMP, GIF, TIF/TIFF

### Documents
- PDF (multi-page support)

## Error Handling

- Gracefully handles missing API keys
- Continues processing with available providers if one fails
- Provides detailed error messages for troubleshooting
- Skips unsupported file types

## Example Output

```
Auto-detected providers: gemini, openai. Using default models.

--- Processing with: GEMINI (Model: gemini-2.0-flash) ---
  Processing PDF page 1/3 with gemini...
  Processing PDF page 2/3 with gemini...
  Processing PDF page 3/3 with gemini...

--- Extracted Text (using GEMINI) ---
[Extracted text content here]

Extracted text from GEMINI saved to: document_gemini_extracted_20241225-143022.txt

--- Processing Complete ---
```

## Tips

1. **For receipts/invoices**: Use custom prompts like "Extract all line items, prices, and totals"
2. **For forms**: Try "Extract all field labels and their values"
3. **For handwritten text**: Some providers perform better than others; try multiple
4. **Large PDFs**: Processing may take time as each page is analyzed separately

## Troubleshooting

- **Import errors**: Ensure all required packages are installed
- **PDF conversion fails**: Check Poppler installation
- **API errors**: Verify API keys and rate limits
- **No providers detected**: Check `.env` file configuration

----
This software is freely distributed under the BSD 3-clause OSI license. Please see the [LICENSE](https://github.com/ccarvel/ai-htr/tree/main?tab=BSD-3-Clause-1-ov-file#) file for more information.
