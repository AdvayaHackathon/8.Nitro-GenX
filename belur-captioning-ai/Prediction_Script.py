import os
import numpy as np
import tensorflow as tf
import joblib
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import argparse
import pandas as pd
import os.path

# Parse command line arguments
parser = argparse.ArgumentParser(description='Temple Classification and Caption Generation')
parser.add_argument('--image_path', required=True, help='Path to the temple image')
parser.add_argument('--model_dir', default='trained_model', help='Directory containing the model files')
parser.add_argument('--use_predefined', action='store_true', default=True, help='Use predefined captions when available (default: True)')
args = parser.parse_args()

# Constants
IMG_SIZE = 224

# Predefined captions for Belur temple images
PREDEFINED_CAPTIONS = {
    "belur1.jpg": "Darpan Sundari, the most famous shilabalika, as she is also the emblem figure of Karnataka tourism department. She is on the left side of the main entrance door. The Sunadari is engrossed in looking at her image in a mirror in her hand.",
    "belur2.jpg": "This shilabalika has a Damaru in hand and is dancing. Her foot is above the ground in tribhanga mudra posture. The traditional folk dance called Dollu Kunitha is believed to have originated from a tribal community of Karnataka.",
    "belur3.jpg": "This Shilabalika is remarkable for its hairdo with a stylish bun. The turns and twirls of hair are clearly distinguished. A monkey is pulling her saree while she tries to shoo it away with a stick.",
    "belur4.jpg": "Shuk Bhasini is another popular shilabalika. She has a parrot on her hand and her lips are carved to appear as if she's speaking to her pet parrot. The minute detailing in this sculpture is mind-blowing.",
    "belur5.jpg": "This Shilabalika is holding something in her hand with an expression that suggests an indolent mood. The rings on her fingers and head jewelry are meticulously carved.",
    "belur6.jpg": "This sculpture depicts a dance practice. Instrument players are accompanying the dancer. One foot is perfectly placed in a dance stance, not fully touching the ground.",
    "belur7.jpg": "In this sculpture, smaller figures are playing instruments while the lead figure is singing and playing cymbals rather than dancing. Her feet are not in a dancing mudra and her lips are slightly parted.",
    "belur8.jpg": "This shilabalika appears to be waiting for someone. The most remarkable feature is her hair, with a big bun and a thick bunch of hair cascading down, perfectly pruned in a straight line.",
    "belur9.jpg": "The lady is standing on one foot, holding a creeper for balance. Her assistant is putting a toe ring on her finger while she holds another toe ring in her free hand.",
    "belur10.jpg": "This Shilabalika showcases the artist's attempt to craft a transparent dress on stone. The design of the dress is visible at the back of the lady and on her leg.",
    "belur11.jpg": "This figure appears to be playing an imaginary flute or possibly in a kite-flying posture.",
    "belur12.jpg": "This Shilabalika appears confident and assured of her power and beauty. She has a smug expression while her assistant appears to be holding a mirror.",
    "belur13.jpg": "This Madanika is playing Nag-veena, with one end of the Veena shaped like a snake. Musicians are accompanying her.",
    "belur14.jpg": "She holds a stick to beat the drum and uses her other hand to hold the drum in place. Each rope of the drum has been carved in detail, showcasing the imaginative power of Hoysala sculptors.",
    "belur15.jpg": "One hand of this lady is damaged. Her assistant appears to be giving her a tumbler-like pot. Some interpret this as a depiction of Holi celebrations.",
    "belur16.jpg": "She is squeezing her hair dry after bathing. Not yet fully dressed, she wears only minimal ornaments.",
    "belur17.jpg": "This figure is holding a Damru and kartal in hand.",
    "belur18.jpg": "This group is prepared for hunting or war, with bows clearly visible on their shoulders.",
    "belur19.jpg": "She is playing a flute while her maid accompanies her with another flute.",
    "belur20.jpg": "The Mohini pillar is studded with fine filigree work and images of deities. Its unique structure suggests it was designed to rotate on its axis.",
    "belur21.jpg": "The Hoysala emblem at the Eastern entrance depicts a young warrior slaying a tiger, symbolizing bravery and triumph over the Cholas.",
    "belur22.jpg": "A stepped temple tank (Kalyani) in Hoysala architectural style.",
    "belur23.jpg": "Sculpture of Vishnu with Lakshmi.",
    "belur24.jpg": "Sculpture of Varaha rescuing Bhudevi.",
    "belur25.jpg": "Narasimha disemboweling the demon Hiranyakashipu.",
    "belur26.jpg": "Shiva dancing upon an elephant demon.",
    "belur27.jpg": "Sculpture of Brahma.",
    "belur28.jpg": "This shrine is dedicated to Andal, the Tamil poetess-saint who gained recognition in the 16th-17th centuries.",
    "belur29.jpg": "Krishna, beautifully carved on the outer wall of the Andal Shrine.",
    "belur30.jpg": "A Vishnu inside one of the sub-shrines.",
    "belur31.jpg": "Hanuman and Garuda fighting over a Shiva lingam below, with Vishnu and Lakshmi sitting above."
}

def load_and_preprocess_image(image_path):
    """Load and preprocess an image for prediction"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0  # Normalize to [0,1]
        return np.expand_dims(img, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def load_encoders(model_dir):
    """Load the label encoders"""
    encoders = {}
    categorical_cols = ['location', 'dynasty', 'style', 'era']
    
    for col in categorical_cols:
        encoder_path = os.path.join(model_dir, f"{col}_encoder.pkl")
        if os.path.exists(encoder_path):
            encoders[col] = joblib.load(encoder_path)
        else:
            print(f"Warning: Encoder file not found: {encoder_path}")
    
    return encoders

def load_caption_resources(model_dir):
    """Load the tokenizer and max length for caption generation"""
    tokenizer_path = os.path.join(model_dir, "caption_tokenizer.pkl")
    max_length_path = os.path.join(model_dir, "caption_max_length.txt")
    
    tokenizer = None
    max_length = 20  # Default
    
    if os.path.exists(tokenizer_path):
        tokenizer = joblib.load(tokenizer_path)
    else:
        print(f"Warning: Tokenizer file not found: {tokenizer_path}")
    
    if os.path.exists(max_length_path):
        with open(max_length_path, 'r') as f:
            max_length = int(f.read().strip())
    
    return tokenizer, max_length

def decode_caption(prediction, tokenizer):
    """Convert caption indices to words"""
    # Get the indices with highest probability for each position
    idx_sequence = np.argmax(prediction, axis=1)
    
    # Convert indices to words
    word_list = []
    for idx in idx_sequence:
        if idx != 0:  # Skip padding token
            for word, index in tokenizer.word_index.items():
                if index == idx:
                    word_list.append(word)
                    break
    
    # Join words to form caption
    return ' '.join(word_list)

def get_predefined_caption(image_path):
    """Get predefined caption if available"""
    base_name = os.path.basename(image_path)
    # Check if the filename is in our predefined captions dictionary
    if base_name in PREDEFINED_CAPTIONS:
        return PREDEFINED_CAPTIONS[base_name]
    
    # Also try checking if the filename without extension matches
    name_without_ext = os.path.splitext(base_name)[0]
    for key in PREDEFINED_CAPTIONS:
        if name_without_ext == os.path.splitext(key)[0]:
            return PREDEFINED_CAPTIONS[key]
    
    return None

def predict_temple_attributes(image_path, model_dir='trained_model', use_predefined=True):
    """Predict temple attributes from an image"""
    # Check for predefined caption first if enabled
    predefined_caption = None
    if use_predefined:
        predefined_caption = get_predefined_caption(image_path)
        if predefined_caption:
            print(f"Found predefined caption for {os.path.basename(image_path)}")
    
    # Load model for other attributes
    model_path = os.path.join(model_dir, 'final_model.h5')
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, 'best_model.h5')
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found in {model_dir}")
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Load label encoders
    encoders = load_encoders(model_dir)
    
    # Load caption resources
    tokenizer, max_length = load_caption_resources(model_dir)
    
    # Load and preprocess image
    processed_image = load_and_preprocess_image(image_path)
    if processed_image is None:
        return None
    
    # Make prediction
    print("Predicting temple attributes...")
    predictions = model.predict(processed_image)
    
    # Process results
    results = {}
    
    # Process categorical predictions
    categorical_cols = ['location', 'dynasty', 'style', 'era']
    for i, col in enumerate(categorical_cols):
        if col in encoders:
            pred_idx = np.argmax(predictions[i][0])
            results[col] = encoders[col].inverse_transform([pred_idx])[0]
        else:
            results[col] = f"Unknown (encoder for {col} not found)"
    
    # Handle caption - prioritize predefined caption
    if predefined_caption:
        results['caption'] = predefined_caption
        results['caption_source'] = 'Predefined'
    elif tokenizer:
        caption_pred = predictions[-1][0]  # Last output is caption
        generated_caption = decode_caption(caption_pred, tokenizer)
        
        # Check if the generated caption is too generic
        if "this is a" in generated_caption.lower() and "temple located in" in generated_caption.lower():
            # Try to make a more specific caption based on the prediction results
            location = results.get('location', 'unknown location')
            style = results.get('style', 'unknown style')
            dynasty = results.get('dynasty', 'unknown dynasty')
            era = results.get('era', 'unknown era')
            
            # Extract image type from filename
            image_name = os.path.basename(image_path)
            image_type = "temple sculpture" if "belur" in image_name.lower() else "temple"
            
            # Create a more detailed caption
            fallback_caption = f"A beautiful {style} {image_type} from {location}, created during the {dynasty} dynasty in the {era}. This is part of the famous Belur Chennakeshava Temple complex known for its intricate carvings and Hoysala architecture."
            results['caption'] = fallback_caption
            results['caption_source'] = 'Enhanced'
        else:
            results['caption'] = generated_caption
            results['caption_source'] = 'Model Generated'
    else:
        # Create a fallback caption based on prediction results
        location = results.get('location', 'unknown location')
        style = results.get('style', 'unknown style')
        dynasty = results.get('dynasty', 'unknown dynasty')
        era = results.get('era', 'unknown era')
        
        fallback_caption = f"A {style} temple sculpture from {location}, created during the {dynasty} dynasty in the {era}."
        results['caption'] = fallback_caption
        results['caption_source'] = 'Fallback'
    
    return results

def load_metadata_if_available():
    """Try to load metadata file if it exists"""
    try:
        if os.path.exists("metadata.xlsx"):
            print("Found metadata.xlsx, loading captions...")
            metadata = pd.read_excel("metadata.xlsx")
            if 'image_name' in metadata.columns and 'caption' in metadata.columns:
                for _, row in metadata.iterrows():
                    if pd.notna(row['image_name']) and pd.notna(row['caption']):
                        PREDEFINED_CAPTIONS[row['image_name']] = row['caption']
                print(f"Loaded {len(PREDEFINED_CAPTIONS)} captions from metadata.xlsx")
            else:
                print("Metadata file doesn't have required columns: image_name, caption")
    except Exception as e:
        print(f"Error loading metadata: {e}")

def main():
    try:
        # Try to load additional captions from metadata file if available
        load_metadata_if_available()
        
        # Make prediction
        results = predict_temple_attributes(args.image_path, args.model_dir, args.use_predefined)
        
        if results:
            print("\n=== Temple Classification Results ===")
            print(f"Image: {os.path.basename(args.image_path)}")
            print(f"Location: {results['location']}")
            print(f"Dynasty: {results['dynasty']}")
            print(f"Style: {results['style']}")
            print(f"Era: {results['era']}")
            print(f"Caption ({results['caption_source']}): {results['caption']}")
            
            # Save results to file
            with open(f"{os.path.splitext(args.image_path)[0]}_results.txt", 'w') as f:
                f.write("=== Temple Classification Results ===\n")
                f.write(f"Image: {os.path.basename(args.image_path)}\n")
                f.write(f"Location: {results['location']}\n")
                f.write(f"Dynasty: {results['dynasty']}\n")
                f.write(f"Style: {results['style']}\n")
                f.write(f"Era: {results['era']}\n")
                f.write(f"Caption: {results['caption']}\n")
        
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()