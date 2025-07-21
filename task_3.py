import os
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any
from PIL import Image
import pytesseract
import whisper
from scipy.io.wavfile import write
import numpy as np
import json

# --- 0. Configuration and API Key Loading ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini Model
GEMINI_MODEL_NAME = 'gemini-1.5-flash'  # Switched to a more robust model
gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

# Load Whisper model
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded.")

# --- 1. Pydantic Data Models ---
class ExtractedAttribute(BaseModel):
    name: str = Field(description="The name of the attribute (e.g., 'price', 'email').")
    value: Any = Field(description="The value of the attribute.")
    data_type: Optional[str] = Field(description="Inferred data type (e.g., 'string', 'number').", default=None)

class ExtractedBlock(BaseModel):
    type: str = Field(description="Entity type (e.g., 'Company', 'Product').")
    name: Optional[str] = Field(description="Primary name or identifier.", default=None)
    attributes: List[ExtractedAttribute] = Field(default_factory=list, description="List of attributes.")
    raw_text_span: Optional[str] = Field(description="Original text snippet.", default=None)

class ExtractedRelationship(BaseModel):
    source_block_name: str = Field(description="Source block name.")
    source_block_type: str = Field(description="Source block type.")
    relationship_type: str = Field(description="Type of relationship.")
    target_block_name: str = Field(description="Target block name.")
    target_block_type: str = Field(description="Target block type.")
    description: Optional[str] = Field(description="Relationship context.", default=None)

class ExtractedKnowledge(BaseModel):
    blocks: List[ExtractedBlock] = Field(default_factory=list, description="List of extracted entities.")
    relationships: List[ExtractedRelationship] = Field(default_factory=list, description="List of relationships.")

# --- Gemini Tool Schema ---
extract_knowledge_tool = {
    "name": "extract_knowledge",
    "description": "Extracts entities, attributes, and relationships from text for industrial companies.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "blocks": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "type": {"type": "STRING", "description": "Entity type"},
                        "name": {"type": "STRING", "description": "Entity name"},
                        "attributes": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "name": {"type": "STRING", "description": "Attribute name"},
                                    "value": {"type": "STRING", "description": "Attribute value"},
                                    "data_type": {"type": "STRING", "description": "Attribute data type"}
                                },
                                "required": ["name", "value"]
                            }
                        },
                        "raw_text_span": {"type": "STRING", "description": "Original text snippet"}
                    },
                    "required": ["type"]
                }
            },
            "relationships": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "source_block_name": {"type": "STRING"},
                        "source_block_type": {"type": "STRING"},
                        "relationship_type": {"type": "STRING"},
                        "target_block_name": {"type": "STRING"},
                        "target_block_type": {"type": "STRING"},
                        "description": {"type": "STRING", "description": "Relationship context"}
                    },
                    "required": ["source_block_name", "source_block_type", "relationship_type", "target_block_name", "target_block_type"]
                }
            }
        },
        "required": ["blocks", "relationships"]
    }
}

# --- 2. Input Conversion Functions ---
def extract_text_from_image(image_path: str) -> Optional[str]:
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip() if text else None
    except Exception as e:
        print(f"Error in OCR for {image_path}: {e}")
        return None

def transcribe_audio(audio_path: str) -> Optional[str]:
    try:
        result = whisper_model.transcribe(audio_path)
        return result['text'].strip() if result and 'text' in result else None
    except Exception as e:
        print(f"Error in audio transcription: {e}")
        return None

def load_text_from_file(file_path: str) -> Optional[str]:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def extract_knowledge_with_gemini(text_input: str) -> Optional[ExtractedKnowledge]:
    if not text_input or not text_input.strip():
        print("Warning: Empty text input provided for extraction.")
        return None
    
    prompt_message = (
        f"Extract entities (e.g., Company, Product, Material, ExternalFactor), their attributes, "
        f"and relationships from the following text using the `extract_knowledge` tool. "
        f"Focus on industrial operations context for a company like STK Produktion GmbH. "
        f"Identify block types, attributes (with data types like 'string', 'number', 'percentage'), "
        f"and relationships (e.g., 'produces', 'affects'). Return empty lists if no data is found.\n\n"
        f"Text: {text_input}\n"
    )
    
    try:
        response = gemini_model.generate_content(
            prompt_message,
            tools=[extract_knowledge_tool],
            tool_config={"function_calling_config": {"mode": "ANY"}}
        )
        
        print(f"Raw Gemini API Response: {response}")  # Debug: Log full response
        
        if not response.candidates or not response.candidates[0].content.parts:
            print("Gemini's response did not contain any content or tool calls.")
            return None
            
        tool_calls = response.candidates[0].content.parts
        
        for part in tool_calls:
            if hasattr(part, 'function_call') and part.function_call.name == extract_knowledge_tool['name']:
                try:
                    # Get the function call arguments
                    extracted_json_args = part.function_call.args
                    print(f"Raw function_call.args: {extracted_json_args}")
                    print(f"Args type: {type(extracted_json_args)}")
                    
                    # Convert to dictionary if needed
                    if hasattr(extracted_json_args, '_pb'):
                        # This is a protobuf object, convert to dict
                        args_dict = {}
                        for key, value in extracted_json_args.items():
                            args_dict[key] = value
                        extracted_json_args = args_dict
                    elif isinstance(extracted_json_args, str):
                        try:
                            extracted_json_args = json.loads(extracted_json_args)
                        except json.JSONDecodeError as e:
                            print(f"Error: function_call.args is a string but not valid JSON: {e}")
                            print(f"Raw string: {extracted_json_args}")
                            return None
                    
                    print(f"Processed args: {extracted_json_args}")
                    
                    # Validate the structure
                    if not isinstance(extracted_json_args, dict):
                        print(f"Error: Processed args is not a dictionary: {type(extracted_json_args)}")
                        return None
                    
                    # Ensure required keys exist
                    if 'blocks' not in extracted_json_args:
                        extracted_json_args['blocks'] = []
                    if 'relationships' not in extracted_json_args:
                        extracted_json_args['relationships'] = []
                    
                    # Validate with Pydantic
                    try:
                        validated_data = ExtractedKnowledge(**extracted_json_args)
                        return validated_data
                    except ValidationError as e:
                        print(f"Validation Error: Extracted data does not match Pydantic schema: {e}")
                        print(f"Raw args: {extracted_json_args}")
                        
                        # Try to fix common validation issues
                        try:
                            # Ensure all required fields are present in blocks
                            for block in extracted_json_args.get('blocks', []):
                                if 'attributes' not in block:
                                    block['attributes'] = []
                                if 'type' not in block:
                                    block['type'] = 'Unknown'
                            
                            # Try validation again
                            validated_data = ExtractedKnowledge(**extracted_json_args)
                            return validated_data
                        except ValidationError as e2:
                            print(f"Second validation attempt failed: {e2}")
                            return None
                            
                except Exception as e:
                    print(f"Unexpected error during argument processing: {e}")
                    print(f"Exception type: {type(e)}")
                    import traceback
                    traceback.print_exc()
                    return None
            else:
                print(f"Warning: Expected tool call '{extract_knowledge_tool['name']}' but received different part type")
                if hasattr(part, 'text'):
                    print(f"Part contains text: {part.text}")
        
        print("No matching function call found in response parts")
        return None
        
    except genai.types.BlockedPromptException as e:
        print(f"Gemini API Blocked Prompt: {e.prompt_feedback}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during Gemini API call: {e}")
        print(f"Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        return None

def visualize_knowledge_graph(knowledge: ExtractedKnowledge):
    if not knowledge or (not knowledge.blocks and not knowledge.relationships):
        print("\n--- No Knowledge Graph to Display ---")
        return
    print("\n" + "="*80)
    print("--- Extracted Knowledge Graph ---")
    print("="*80)
    print("\nBlocks (Entities):\n")
    for i, block in enumerate(knowledge.blocks):
        print(f"  [{i+1}] {block.name or 'Unnamed'} (Type: {block.type})")
        if block.attributes:
            for attr in block.attributes:
                print(f"    - {attr.name}: {attr.value} (Type: {attr.data_type or 'N/A'})")
        if block.raw_text_span:
            print(f"    (From: '{block.raw_text_span}')")
        print("-" * 20)
    print("\nRelationships (Connections):\n")
    if knowledge.relationships:
        for rel in knowledge.relationships:
            print(f"  ({rel.source_block_name} [{rel.source_block_type}]) "
                  f"--[{rel.relationship_type}]--> "
                  f"({rel.target_block_name} [{rel.target_block_type}])")
            if rel.description:
                print(f"    Description: {rel.description}")
        print("-" * 80)
    else:
        print("  No explicit relationships extracted.")
    print("="*80 + "\n")

if __name__ == "__main__":
    print(f"--- Running Knowledge Extraction for STK Produktion GmbH ---")
    print(f"Using Gemini Model: {GEMINI_MODEL_NAME}\n")
    os.makedirs("test_inputs", exist_ok=True)
    
    # Create dummy text file
    with open("test_inputs/stk_text_report.txt", "w") as f:
        f.write("STK Produktion GmbH specializes in manufacturing plastic and metal components. "
                "Their new 'Eco-Polymer Part X' (Product ID: EP-100) is in high demand. "
                "The current energy prices (ExternalFactor) have increased by 15% affecting the Production Process for metal components. "
                "Supply Chain for raw materials is managed by Operations Department. "
                "Tariffs on imported steel (RegulatoryFramework) impact the Cost Structure of metal components. "
                "The CEO, Dr. Anna Schmidt, is overseeing Project Green, aimed at reducing CO2 emissions by 20% by 2026.")

    # Create dummy image
    try:
        from PIL import ImageDraw, ImageFont
        img = Image.new('RGB', (400, 100), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        try:
            fnt = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            try:
                fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except IOError:
                fnt = ImageFont.load_default()
        d.text((10,10), "Production capacity for 'Plastic Housing Y' is 5000 units/day.", fill=(0,0,0), font=fnt)
        img.save("test_inputs/stk_production_capacity.png")
        print("Dummy image 'stk_production_capacity.png' created.")
    except Exception as e:
        print(f"Could not create dummy image: {e}. Skipping image test.")
        if not os.path.exists("test_inputs/stk_production_capacity.png"):
            with open("test_inputs/stk_production_capacity.png", "w") as f: f.write("")

    # Create dummy audio
    dummy_audio_path = "test_inputs/stk_audio_report.wav"
    if not os.path.exists(dummy_audio_path):
        samplerate = 16000
        duration = 2
        data = np.zeros(int(samplerate * duration)).astype(np.int16)
        write(dummy_audio_path, samplerate, data)
        print("Created a silent dummy audio file.")

    # Process text input
    print("\n" + "="*100)
    print("--- Processing Text Report ---")
    text_report_path = "test_inputs/stk_text_report.txt"
    stk_text_content = load_text_from_file(text_report_path)
    if stk_text_content:
        print(f"\nSource: Text File '{text_report_path}'")
        extracted_data_text = extract_knowledge_with_gemini(stk_text_content)
        if extracted_data_text:
            visualize_knowledge_graph(extracted_data_text)
        else:
            print("No structured knowledge extracted from the text report.")
    print("="*100 + "\n")

    # Process image input
    image_path = "test_inputs/stk_production_capacity.png"
    if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
        print("\n" + "="*100)
        print("--- Processing Image Input ---")
        image_text_content = extract_text_from_image(image_path)
        if image_text_content:
            print(f"\nSource: Image File '{image_path}' (OCR Text: '{image_text_content}')")
            extracted_data_image = extract_knowledge_with_gemini(image_text_content)
            if extracted_data_image:
                visualize_knowledge_graph(extracted_data_image)
            else:
                print("No structured knowledge extracted from the image.")
        else:
            print(f"Could not extract text from image '{image_path}'. Skipping image processing.")
        print("="*100 + "\n")
    else:
        print(f"\nSkipping Image Input Test: Image '{image_path}' not available or empty.")

    # Process audio input
    if os.path.exists(dummy_audio_path) and os.path.getsize(dummy_audio_path) > 0:
        print("\n" + "="*100)
        print("--- Processing Audio Input ---")
        audio_transcript_content = transcribe_audio(dummy_audio_path)
        if audio_transcript_content:
            print(f"\nSource: Audio File '{dummy_audio_path}' (Transcript: '{audio_transcript_content}')")
            extracted_data_audio = extract_knowledge_with_gemini(audio_transcript_content)
            if extracted_data_audio:
                visualize_knowledge_graph(extracted_data_audio)
            else:
                print("No structured knowledge extracted from the audio transcript.")
        else:
            print(f"Could not transcribe audio from '{dummy_audio_path}'. Skipping audio processing.")
        print("="*100 + "\n")
    else:
        print(f"\nSkipping Audio Input Test: Audio file '{dummy_audio_path}' not available or empty.")

    # Clean up
    for f_name in ["stk_text_report.txt", "stk_production_capacity.png", "stk_audio_report.wav"]:
        if os.path.exists(os.path.join("test_inputs", f_name)):
            try:
                os.remove(os.path.join("test_inputs", f_name))
            except Exception as e:
                print(f"Could not clean up dummy file {f_name}: {e}")
    if os.path.exists("test_inputs"):
        try:
            os.rmdir("test_inputs")
        except OSError:
            pass