# {
#   "free_space_raw": [
#     [v1, v2, v3, ..., v1536],  // Embedding 1
#     [v1, v2, v3, ..., v1536],  // Embedding 2
#     [v1, v2, v3, ..., v1536]   // Embedding 3
#   ],
#   "free_space": [v1, v2, v3, ..., v1536],  // Normalized centroid
#   "wall_raw": [...],
#   "wall": [...],
#   "metadata": {...}
# }









import os
import json
import argparse
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor
import torch

# Categories (same as in ijepa_integration_predictor.py)
CATEGORIES = ['free_space', 'wall', 'door', 'corridor', 'clutter']

# Paths - relative to main project root
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
# IMAGE_BASE_DIR = os.path.join(PROJECT_ROOT, 'main/src/prediction_pipeline/prototype_images_for_ijepa')
# OUTPUT_JSON = os.path.join(PROJECT_ROOT, 'main/src/prediction_pipeline/ijepa_prototypes.json')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../..'))
print(f"📁 PROJECT_ROOT: {PROJECT_ROOT}")
print(f"📁 SCRIPT_DIR: {SCRIPT_DIR}")

# Image directory: should be in the same folder as this script
IMAGE_BASE_DIR = os.path.join(SCRIPT_DIR, 'prototype_images_for_ijepa')
print(f"📁 IMAGE_BASE_DIR: {IMAGE_BASE_DIR}")

# Output JSON: save in the same folder as this script
OUTPUT_JSON = os.path.join(SCRIPT_DIR, 'ijepa_prototypes.json')
print(f"📁 OUTPUT_JSON: {OUTPUT_JSON}")





def load_ijepa_model():
    """EXACT same logic as ijepa_integration_predictor.py"""
    print("🔮 Loading I-JEPA model (same as predictor)...")
    
    model_id = "facebook/ijepa_vith14_1k"
    print(f"🔮 Loading {model_id}...")
    
    # EXACT same as predictor
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    
    # EXACT same as predictor
    model.eval()
    print(f"✅ Model loaded: {model_id}")
    print(f"✅ Embedding dim: {model.config.hidden_size}")
    
    return model, processor


def l2_normalize(embeddings):
    """L2 normalize embeddings along the last dimension"""
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / norms


def process_category(category, model, processor, json_data):
    """Process a single category: load images, get embeddings, compute centroid"""
    print(f"\n📂 Processing category: {category}")
    
    category_path = os.path.join(IMAGE_BASE_DIR, category)
    
    # Check if directory exists
    if not os.path.exists(category_path):
        print(f"⚠️  Directory not found: {category_path}")
        return False
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(category_path) 
                          if f.lower().endswith(ext)])
    
    if not image_files:
        print(f"⚠️  No images found in {category_path}")
        return False
    
    print(f"📸 Found {len(image_files)} images")
    
    # Process each image
    all_embeddings = []
    
    for img_file in image_files:
        img_path = os.path.join(category_path, img_file)
        
        try:
            # Load and preprocess image
            image = Image.open(img_path).convert('RGB')
            inputs = processor(images=image, return_tensors="pt")
            
            # Get embedding
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            all_embeddings.append(embedding)
            print(f"  ✓ Processed: {img_file}")
            
        except Exception as e:
            print(f"  ✗ Failed to process {img_file}: {e}")
    
    if not all_embeddings:
        print(f"⚠️  No embeddings generated for {category}")
        return False
    
    # Convert to numpy array
    all_embeddings = np.array(all_embeddings)
    print(f"✅ Generated {all_embeddings.shape[0]} embeddings, dimension {all_embeddings.shape[1]}")
    
    # ✅ Step 1: L2-normalize all raw embeddings
    normalized_embeddings = l2_normalize(all_embeddings)
    
    # Store raw embeddings (already normalized)
    raw_key = f"{category}_raw"
    json_data[raw_key] = normalized_embeddings.tolist()
    print(f"💾 Saved normalized raw embeddings: {raw_key}")
    print(f"   DEBUG: json_data[{raw_key}] type: {type(json_data[raw_key])}, len: {len(json_data[raw_key])}")
    
    # ✅ Step 2: Compute centroid (mean vector)
    centroid = np.mean(normalized_embeddings, axis=0)
    print(f"📊 Computed centroid for {category}")
    print(f"   DEBUG: Centroid shape: {centroid.shape}")
    
    # ✅ Step 3: L2-normalize the centroid
    normalized_centroid = l2_normalize(centroid.reshape(1, -1)).squeeze()
    print(f"📐 Normalized centroid (L2 length = {np.linalg.norm(normalized_centroid):.6f})")
    print(f"   DEBUG: Normalized centroid shape: {normalized_centroid.shape}")
    
    # Store normalized centroid
    centroid_key = category
    json_data[centroid_key] = normalized_centroid.tolist()
    print(f"💾 Saved normalized centroid: {centroid_key}")
    print(f"   DEBUG: json_data[{centroid_key}] type: {type(json_data[centroid_key])}, len: {len(json_data[centroid_key])}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Collect I-JEPA prototypes')
    parser.add_argument('--category', type=str, required=False,
                       choices=CATEGORIES + ['all'], default='all',
                       help=f'Category to process. Use "all" for all categories. Choices: {CATEGORIES}')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='Device to use (if GPU available)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"🔄 I-JEPA Prototype Embedding Generator")
    print(f"📁 Category: {args.category}")
    print("=" * 60)
    
    # Set device if GPU available
    if args.device == 'cuda' and torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 Using CPU")
    
    # Load existing JSON if it exists
    json_data = {}
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, 'r') as f:
                json_data = json.load(f)
            print(f"📖 Loaded existing JSON with {len(json_data.keys())} keys")
        except Exception as e:
            print(f"⚠️  Failed to load existing JSON: {e}")
    
    # Load I-JEPA model
    model, processor = load_ijepa_model()
    
    # Move model to device if needed
    if args.device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
    
    # Determine which categories to process
    if args.category == 'all':
        categories_to_process = CATEGORIES
    else:
        categories_to_process = [args.category]
    
    # Process categories
    success_count = 0  # ⬅️ INITIALIZE HERE!
    for category in categories_to_process:
        print(f"\n🚀 Starting to process {category}")
        print(f"   Current json_data keys: {list(json_data.keys())}")
        
        if process_category(category, model, processor, json_data):
            success_count += 1
            print(f"   ✅ {category} processed successfully")
            print(f"   Updated json_data keys: {list(json_data.keys())}")
        else:
            print(f"   ❌ {category} failed to process")
    
    # DEBUG: Print all keys before saving
    print(f"\n🔍 DEBUG: Final json_data keys before save: {list(json_data.keys())}")
    for key in json_data:
        if key.endswith('_raw'):
            print(f"   {key}: {len(json_data[key])} embeddings")
        else:
            print(f"   {key}: centroid vector ({len(json_data[key])} dims)")
    
    # Save to JSON file
    print(f"\n💾 Saving to {OUTPUT_JSON}")
    try:
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"✅ Successfully saved JSON with {len(json_data.keys())} keys")
        print(f"✅ Processed {success_count}/{len(categories_to_process)} categories successfully")
        
        # Verify the saved file
        with open(OUTPUT_JSON, 'r') as f:
            saved_data = json.load(f)
            print(f"📖 Verified saved file keys: {list(saved_data.keys())}")
            
    except Exception as e:
        print(f"❌ Failed to save JSON: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 JSON structure preview:")
    for key in json_data:
        if key.endswith('_raw'):
            print(f"  {key}: {len(json_data[key])} embeddings (normalized)")
        else:
            print(f"  {key}: centroid vector (normalized)")
    
    print("\n✅ Processing complete!")


if __name__ == "__main__":
    main()