import tensorflow as tf
from transformers import BertTokenizer, TFAutoModelForQuestionAnswering
import numpy as np
from pathlib import Path
import logging
import time
import platform

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
else:  # Windows or other platforms
    logger.info("Configuring TensorFlow for available GPU...")
    try:
        for device in tf.config.list_physical_devices():
            logger.info(f"- {device}")
    except Exception as e:
        logger.warning(f"Could not list devices: {e}")

# Enable mixed precision for better performance
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Check if GPU is available
logger.info(f"TensorFlow GPU available: {tf.test.is_built_with_gpu_support()}")
logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
logger.info(f"Devices: {tf.config.list_physical_devices()}")

# 使用基础中文BERT模型
MODEL_NAME = "uer/roberta-base-chinese-extractive-qa"

# 加载分词器和模型
logger.info("Loading tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = TFAutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

# 准备示例输入
context = "深圳信迈科技有限公司创立于2016年。"
question = "深圳信迈科技有限公司哪一年成立?"

# 编码输入
logger.info("Processing input...")
inputs = tokenizer(
    question,
    context,
    add_special_tokens=True,
    return_tensors="tf",
    padding=True,
    truncation=True,
    max_length=384
)

# Convert inputs to float16 for mixed precision
inputs = {k: tf.cast(v, tf.int32) for k, v in inputs.items()}

# Warm up the GPU
logger.info("Warming up GPU...")
_ = model(inputs)

# 运行推理
logger.info("Running inference...")
start_time = time.time()
outputs = model(inputs)
inference_time = time.time() - start_time

# 获取答案
answer_start = int(tf.argmax(outputs.start_logits[0]))
answer_end = int(tf.argmax(outputs.end_logits[0])) + 1
answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])

print(f"\n**问题:** {question}")
print(f"**答案:** {answer}")

# Get device info
device_name = "CPU"
if tf.config.list_physical_devices('GPU'):
    try:
        # Get current device
        device = tf.test.gpu_device_name()
        if device:
            device_properties = tf.config.experimental.get_device_details('GPU:0')
            device_name = f"GPU: {device_properties.get('device_name', 'Metal GPU')}"
        logger.info(f"Device properties: {device_properties}")
    except Exception as e:
        logger.warning(f"Could not get detailed GPU info: {e}")

print(f"\nDevice: {device_name}")
print(f"Inference time: {inference_time*1000:.2f}ms")

# Print additional performance information
logger.info("\nPerformance Information:")
logger.info(f"Mixed Precision Enabled: {tf.keras.mixed_precision.global_policy().name}")
logger.info(f"Available devices: {[device.device_type for device in tf.config.list_logical_devices()]}")
try:
    memory_info = tf.config.experimental.get_memory_info('GPU:0')
    logger.info(f"GPU Memory Used: {memory_info['peak'] / 1024**2:.2f}MB")
except:
    pass
