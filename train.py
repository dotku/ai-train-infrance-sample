import tensorflow as tf
from transformers import BertTokenizer, TFAutoModelForQuestionAnswering
import numpy as np
from pathlib import Path
import logging
import time
import platform
import json
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_gpu():
    # Configure TensorFlow to use Metal
    if platform.system() == 'Darwin':  # macOS
        logger.info("Configuring TensorFlow for Metal GPU...")
        try:
            # List available devices
            logger.info("Available devices:")
            for device in tf.config.list_physical_devices():
                logger.info(f"- {device}")
            
            # Enable Metal plugin
            tf.config.experimental.set_visible_devices(
                tf.config.list_physical_devices('GPU'), 'GPU'
            )
        except Exception as e:
            logger.warning(f"Could not configure Metal GPU: {e}")
    else:
        logger.info("Configuring TensorFlow for available GPU...")
        try:
            for device in tf.config.list_physical_devices():
                logger.info(f"- {device}")
        except Exception as e:
            logger.warning(f"Could not list devices: {e}")

    # Enable mixed precision for better performance
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

def prepare_training_data(tokenizer, csv_path="sample_qa.csv", max_length=384):
    logger.info(f"Loading training data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Prepare training features
    input_ids_list = []
    attention_mask_list = []
    start_positions_list = []
    end_positions_list = []
    
    for _, row in df.iterrows():
        # Tokenize the input
        encoding = tokenizer(
            row['question'],
            row['context'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='tf'
        )
        
        # Find the answer span in the context
        context = row['context']
        answer = row['answer']
        answer_start = context.find(answer)
        answer_end = answer_start + len(answer)
        
        # Convert character positions to token positions
        input_ids = encoding['input_ids'][0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        # Find token-level start and end positions
        token_start = None
        token_end = None
        char_pos = 0
        for i, token in enumerate(tokens):
            if char_pos <= answer_start < char_pos + len(token):
                token_start = i
            if char_pos <= answer_end <= char_pos + len(token):
                token_end = i
                break
            char_pos += len(token)
        
        # If we couldn't find the answer span, use the first token as default
        if token_start is None:
            token_start = 0
        if token_end is None:
            token_end = 0
        
        input_ids_list.append(encoding['input_ids'][0])
        attention_mask_list.append(encoding['attention_mask'][0])
        start_positions_list.append(token_start)
        end_positions_list.append(token_end)
    
    # Convert to TensorFlow tensors
    return {
        'input_ids': tf.stack(input_ids_list),
        'attention_mask': tf.stack(attention_mask_list),
        'start_positions': tf.convert_to_tensor(start_positions_list),
        'end_positions': tf.convert_to_tensor(end_positions_list)
    }

def train_model(model_name="uer/roberta-base-chinese-extractive-qa", epochs=3, batch_size=8):
    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)
    
    # Prepare training data
    train_features = prepare_training_data(tokenizer)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices({
        'input_ids': train_features['input_ids'],
        'attention_mask': train_features['attention_mask'],
        'start_positions': train_features['start_positions'],
        'end_positions': train_features['end_positions']
    }).shuffle(1000).batch(batch_size)
    
    # Configure optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    
    # Training loop
    logger.info("Starting training...")
    
    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                training=True
            )
            
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
            # Calculate loss using sparse categorical crossentropy
            start_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=batch['start_positions'],
                    logits=start_logits
                )
            )
            end_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=batch['end_positions'],
                    logits=end_logits
                )
            )
            
            # Total loss is the average of start and end losses
            total_loss = (start_loss + end_loss) / 2.0
        
        # Calculate gradients
        gradients = tape.gradient(total_loss, model.trainable_variables)
        
        # Apply gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return total_loss
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        num_batches = 0
        
        for batch in dataset:
            loss = train_step(batch)
            total_loss += float(loss)
            num_batches += 1
            
            if num_batches % 5 == 0:  # Log every 5 batches
                logger.info(f"Batch {num_batches}, Loss: {float(loss):.4f}")
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}")
    
    # Save training configuration
    config_path = Path("model_config")
    config_path.mkdir(exist_ok=True)
    
    tokenizer_config = {
        "model_name": model_name,
        "max_length": 384,
        "padding": True,
        "truncation": True
    }
    with open(config_path / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f)
    
    # Save model and tokenizer
    model_path = Path("saved_model")
    model_path.mkdir(exist_ok=True)
    model.save_pretrained(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    
    logger.info("Model and tokenizer saved successfully")
    return model, tokenizer

if __name__ == "__main__":
    setup_gpu()
    
    # Check if GPU is available
    logger.info(f"TensorFlow GPU available: {tf.test.is_built_with_gpu_support()}")
    logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    logger.info(f"Devices: {tf.config.list_physical_devices()}")

    # Train and save model
    train_model()