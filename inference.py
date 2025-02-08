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

def load_model():
    # Load saved model and config
    config_path = Path("model_config")
    model_path = Path("saved_model")

    with open(config_path / "tokenizer_config.json", "r") as f:
        config = json.load(f)

    tokenizer = BertTokenizer.from_pretrained(str(model_path))
    model = TFAutoModelForQuestionAnswering.from_pretrained(str(model_path))

    return model, tokenizer, config

def get_answer(question, context, model, tokenizer, config):
    # 编码输入
    inputs = tokenizer(
        question,
        context,
        add_special_tokens=True,
        return_tensors="tf",
        padding=config["padding"],
        truncation=config["truncation"],
        max_length=config["max_length"]
    )

    # Convert inputs to float16 for mixed precision
    inputs = {k: tf.cast(v, tf.int32) for k, v in inputs.items()}

    # Run inference
    start_time = time.time()
    outputs = model(inputs)
    inference_time = time.time() - start_time

    # Get answer
    answer_start = int(tf.argmax(outputs.start_logits[0]))
    answer_end = int(tf.argmax(outputs.end_logits[0])) + 1
    answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])

    return answer, inference_time

def get_device_info():
    device_name = "CPU"
    if tf.config.list_physical_devices('GPU'):
        try:
            device = tf.test.gpu_device_name()
            if device:
                device_properties = tf.config.experimental.get_device_details('GPU:0')
                device_name = f"GPU: {device_properties.get('device_name', 'Metal GPU')}"
            logger.info(f"Device properties: {device_properties}")
        except Exception as e:
            logger.warning(f"Could not get detailed GPU info: {e}")
    return device_name

def evaluate_qa_from_csv(csv_file, model, tokenizer, config):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    total_time = 0
    correct_answers = 0
    
    print("\nEvaluating QA pairs from CSV:")
    print("-" * 50)
    
    for idx, row in df.iterrows():
        # Get answer from model
        predicted_answer, inference_time = get_answer(
            row['question'], 
            row['context'], 
            model, 
            tokenizer, 
            config
        )
        
        total_time += inference_time
        
        # Compare with ground truth (allowing for some flexibility in matching)
        is_correct = predicted_answer.strip() in row['answer'] or row['answer'].strip() in predicted_answer
        if is_correct:
            correct_answers += 1
            
        print(f"\nQuestion {idx + 1}:")
        print(f"Context: {row['context']}")
        print(f"Question: {row['question']}")
        print(f"Expected Answer: {row['answer']}")
        print(f"Model Answer: {predicted_answer}")
        print(f"Correct: {'✓' if is_correct else '✗'}")
        print(f"Inference time: {inference_time*1000:.2f}ms")
        print("-" * 50)
    
    # Print summary
    accuracy = (correct_answers / len(df)) * 100
    avg_time = (total_time / len(df)) * 1000  # Convert to ms
    
    print(f"\nSummary:")
    print(f"Total examples: {len(df)}")
    print(f"Correct answers: {correct_answers}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average inference time: {avg_time:.2f}ms")
    print(f"Device: {get_device_info()}")
    
    # Print performance information
    logger.info("\nPerformance Information:")
    logger.info(f"Mixed Precision Enabled: {tf.keras.mixed_precision.global_policy().name}")
    logger.info(f"Available devices: {[device.device_type for device in tf.config.list_logical_devices()]}")

if __name__ == "__main__":
    # Load model
    model, tokenizer, config = load_model()

    # Evaluate all QA pairs from CSV
    csv_file = "sample_qa.csv"
    evaluate_qa_from_csv(csv_file, model, tokenizer, config)