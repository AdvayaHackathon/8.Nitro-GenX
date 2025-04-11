import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib  # Add this import
from tensorflow.keras import layers, Model, applications, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2

# ====================================================
# Step 1: Prepare Metadata
# ====================================================
metadata = pd.read_excel("metadata.xlsx")

# Verify data is loaded correctly
print(f"Loaded {len(metadata)} records from metadata.xlsx")
print(f"Columns found: {metadata.columns.tolist()}")

# Ensure required columns exist
required_cols = ['image_name', 'location', 'dynasty', 'style', 'era']
for col in required_cols:
    if col not in metadata.columns:
        print(f"Warning: '{col}' column not found in metadata. Please ensure your Excel file has this column.")

# Create a caption column if it doesn't exist
if 'caption' not in metadata.columns:
    print("Caption column not found. Creating descriptive captions from attributes...")
    metadata['caption'] = metadata.apply(
        lambda row: f"This is a {row['style']} temple located in {row['location']}, built during the {row['dynasty']} dynasty in the {row['era']}.", 
        axis=1
    )

# Encode categorical labels
label_encoders = {}
categorical_cols = ['location', 'dynasty', 'style', 'era']

for col in categorical_cols:
    le = LabelEncoder()
    metadata[col] = le.fit_transform(metadata[col])
    label_encoders[col] = le

# ====================================================
# Step 2: Image Processing
# ====================================================
# Create image paths with verification
image_paths = []
for fname in metadata['image_name']:
    path = os.path.join("data", fname)
    if os.path.exists(path):
        image_paths.append(path)
    else:
        print(f"Warning: Image file not found: {path}")

if len(image_paths) == 0:
    raise ValueError("No valid image paths found. Please check your data directory and image filenames.")

print(f"Found {len(image_paths)} valid images for training.")
IMG_SIZE = 224
BATCH_SIZE = 4  # Small batch size due to limited data

def load_and_preprocess(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalize to [0,1]
    return img

# Create image paths
image_paths = [os.path.join("data", fname) for fname in metadata['image_name']]

# ====================================================
# Step 3: Data Augmentation
# ====================================================
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# ====================================================
# Step 4: Build Multi-Output Model
# ====================================================
# Add these imports at the top
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string

# After loading metadata, add this code to process captions
# ====================================================
# Process caption data
# ====================================================
# Assuming metadata has a 'caption' column with text descriptions
if 'caption' in metadata.columns:
    # Clean captions
    metadata['clean_caption'] = metadata['caption'].apply(lambda x: x.lower().translate(
        str.maketrans('', '', string.punctuation)))
    
    # Create tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(metadata['clean_caption'])
    vocab_size = len(tokenizer.word_index) + 1
    
    # Convert captions to sequences
    caption_sequences = tokenizer.texts_to_sequences(metadata['clean_caption'])
    max_length = max(len(seq) for seq in caption_sequences)
    padded_sequences = pad_sequences(caption_sequences, maxlen=max_length, padding='post')
    
    # Save tokenizer
    joblib.dump(tokenizer, 'trained_model/caption_tokenizer.pkl')
    
    # Save max_length
    with open('trained_model/caption_max_length.txt', 'w') as f:
        f.write(str(max_length))
else:
    # If no caption data, create dummy values
    vocab_size = 1000  # Arbitrary
    max_length = 20    # Arbitrary
    padded_sequences = np.zeros((len(metadata), max_length))

# Modify the create_model function to include caption generation
def create_model():
    # Base model
    base_model = applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Make some layers trainable for better feature extraction
    base_model.trainable = True
    for layer in base_model.layers[:-30]:  # Freeze early layers
        layer.trainable = False

    # Inputs
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=True)  # Set training=True
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)  # Increased units
    x = layers.BatchNormalization()(x)  # Add batch normalization
    x = layers.Dropout(0.3)(x)  # Reduced dropout

    # Classification Heads with improved architecture
    outputs = []
    for col in categorical_cols:
        num_class = len(label_encoders[col].classes_)
        feat = layers.Dense(256, activation='relu')(x)
        feat = layers.BatchNormalization()(feat)
        out = layers.Dense(num_class, activation='softmax', name=col)(feat)
        outputs.append(out)

    # Caption Head
    if 'caption' in metadata.columns:
        # Caption Head
        if 'caption' in metadata.columns:
            # Create a proper sequence model for captions
            caption_features = layers.Dense(512, activation='relu')(x)
            caption_features = layers.BatchNormalization()(caption_features)
            
            # Reshape to sequence format and repeat to match max_length
            caption_features = layers.RepeatVector(max_length)(caption_features)
            
            # Add LSTM layers for sequence processing
            caption_features = layers.LSTM(512, return_sequences=True)(caption_features)
            caption_features = layers.TimeDistributed(layers.Dense(256, activation='relu'))(caption_features)
            
            # Output layer with vocab_size units for each timestep
            caption_out = layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax'), name='caption')(caption_features)
        else:
            # Fallback for when no caption data is available
            caption_out = layers.Dense(vocab_size, activation='softmax', name='caption')(x)
    else:
        caption_out = layers.Dense(vocab_size, activation='softmax', name='caption')(x)
    
    return Model(inputs=inputs, outputs=outputs + [caption_out])

# Define data_generator before using it
def data_generator(paths, batch_size):
    indices = np.arange(len(paths))
    while True:
        np.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_paths = [paths[idx] for idx in batch_indices]
            
            if len(batch_paths) == 0:
                continue
            
            # Load and preprocess images with error handling
            batch_images = []
            valid_paths = []
            for p in batch_paths:
                try:
                    img = load_and_preprocess(p)
                    batch_images.append(img)
                    valid_paths.append(p)
                except Exception as e:
                    print(f"Error processing image {p}: {e}")
            
            if len(batch_images) == 0:
                continue
                
            batch_images = np.array(batch_images)
            
            # Get corresponding metadata
            batch_filenames = [os.path.basename(p) for p in valid_paths]
            batch_data = metadata[metadata['image_name'].isin(batch_filenames)]
            
            if len(batch_data) == 0:
                continue
                
            # Prepare outputs
            y = {
                'location': batch_data['location'].values,
                'dynasty': batch_data['dynasty'].values,
                'style': batch_data['style'].values,
                'era': batch_data['era'].values,
            }
            
            # Add caption data if available
            if 'caption' in metadata.columns:
                batch_indices = batch_data.index
                # Fix: Reshape the caption data to match the expected dimensions
                caption_data = padded_sequences[batch_indices]
                # For each position in the sequence, we need a separate target
                # This creates a sequence of targets for each word in the caption
                y['caption'] = caption_data
            else:
                y['caption'] = np.zeros((len(batch_data), max_length))
            
            yield batch_images, y

# ====================================================
# Step 5: Prepare Data
# ====================================================
# Split data
train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

# Create generators
train_gen = data_generator(train_paths, BATCH_SIZE)
val_gen = data_generator(val_paths, BATCH_SIZE)

# ====================================================
# Step 6: Create and Compile Model
# ====================================================
# Create model first
model = create_model()

# Then compile it with the improved configuration
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
    loss={
        'location': 'sparse_categorical_crossentropy',
        'dynasty': 'sparse_categorical_crossentropy',
        'style': 'sparse_categorical_crossentropy',
        'era': 'sparse_categorical_crossentropy',
        'caption': 'sparse_categorical_crossentropy'
    },
    loss_weights={
        'location': 1.0,
        'dynasty': 1.0,
        'style': 1.0,
        'era': 1.0,
        'caption': 0.3  # Reduced caption weight
    },
    metrics={
        'location': ['accuracy', 'top_k_categorical_accuracy'],
        'dynasty': ['accuracy', 'top_k_categorical_accuracy'],
        'style': ['accuracy', 'top_k_categorical_accuracy'],
        'era': ['accuracy', 'top_k_categorical_accuracy'],
        'caption': ['accuracy']
    }
)

# Update callbacks for better training
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss'),
    tf.keras.callbacks.ModelCheckpoint('trained_model/best_model.h5', save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6),
    tf.keras.callbacks.CSVLogger('training_log.csv')
]

# Calculate steps properly to avoid incomplete batches
steps_per_epoch = max(1, len(train_paths) // BATCH_SIZE)
validation_steps = max(1, len(val_paths) // BATCH_SIZE)

# Train
try:
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=100,
        callbacks=callbacks
    )
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during training: {e}")

# ====================================================
# Step 7: Save Model
# ====================================================
model.save('trained_model/final_model.h5')

# Save label encoders
import joblib
for col in categorical_cols:
    joblib.dump(label_encoders[col], f'trained_model/{col}_encoder.pkl')