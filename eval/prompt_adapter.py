from openai import OpenAI
import json
import uuid
import mimetypes
from datetime import datetime
import os
from tqdm import tqdm

import json
import uuid
import mimetypes
from datetime import datetime
import os


def encode_image(image_input):
    # print(image_input)

    if hasattr(image_input, 'save'):
        buffered = BytesIO()
        image_input.save(buffered, format=image_input.format if image_input.format else 'PNG')
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    image_path = image_input
    if image_path.startswith("http"):
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
        request_kwargs = {
            "headers": {"User-Agent": user_agent},
            "stream": True,
        }

        response = requests.get(image_path, **request_kwargs)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")

        extension = mimetypes.guess_extension(content_type)
        if extension is None:
            extension = ".download"

        fname = str(uuid.uuid4()) + extension
        download_path = os.path.abspath(os.path.join("downloads", fname))

        with open(download_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=512):
                fh.write(chunk)

        image_path = download_path

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')




import base64
import requests
from PIL import Image
from io import BytesIO


def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format='PNG')
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def resize_image(image_path):
    img = Image.open(image_path)
    width, height = img.size
    img = img.resize((int(width / 2), int(height / 2)))
    new_image_path = f"resized_{image_path}"
    img.save(new_image_path)
    return new_image_path

import logging
import sys
from pathlib import Path

def setup_logger(name, log_file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        # 使用 'w' 模式覆盖原有日志文件，实现清空效果
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def get_prompt_qa(model_family, sample, glm_thinking=True):

    image = sample.get("image", None)
    if model_family == "normal" or model_family == "openai" or model_family == "gemini" or model_family == "qwen" or model_family == "internvl" or model_family == "internvl3.5":
        return [
            {"role": "system", "content": """
You are a helpful agent.Here is an image with a question about the image content. You should reply the question according to the image faithfully. Please note that the question maybe confusing or the image content might be uncommon, you should answer the question strongly depends on yhe image."""},
            {"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(image)}"}
                },
                {
                    "type": "text",
                    "text": f"""
Question:{sample['question']}
Your answer:"""
                },
            ]}
        ]
    elif model_family == "claude":

        return [
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": f"""
(System Instruction) You are a helpful agent.Here is an image with a question about the image content. You should reply the question according to the image faithfully. Please note that the question maybe confusing or the image content might be uncommon, you should answer the question strongly depends on yhe image."""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(image)}"}
                },
                {
                    "type": "text",
                    "text": f"""
Question:{sample['question']}
Your answer:"""
                }
            ]}
        ]
    elif model_family == 'glm':
        return [
            {"role": "system", "content": """
You are a helpful agent.Here is an image with a question about the image content. You should reply the question according to the image faithfully. Please note that the question maybe confusing or the image content might be uncommon, you should answer the question strongly depends on yhe image."""},
            {"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(image)}"}
                },
                {
                    "type": "text",
                    "text": f"""
Question:{sample['question']}
Your answer:"""
                },
            ]}
        ]

def get_prompt(model_family, sample, permutation,):
    image = sample.get("image", None)
    if model_family == "normal" or model_family == "openai" or model_family == "gemini" or model_family == "qwen" or model_family == "internvl" or model_family == "internvl3.5":
        return [
            {"role": "system", "content": """
You are a helpful agent.Here is an image with a multiple choice question about the image content. You should reply the question according to the image faithfully. Please note that the question maybe confusing or the image content might be uncommon, you should answer the question ONLY with the correct choice letter.
Here is an example:
#########
[IMAGE]
Question:Does the Teapot in the picture have a handle? If so, where is it located?
Choices:
A. Not visible / Can't see.
B. Yes, on the side.
C. Yes, arched over the top.
D. The correct answer is not listed.

Your answer: A
#########
Now please answer the question following the above format STRICTLY.
            """},
            {"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(image)}"}
                },
                {
                    "type":"text",
                    "text":f"""
Question:{sample['question']}
Choices:
A. {sample['options'][permutation[0]]}
B. {sample['options'][permutation[1]]}
C. {sample['options'][permutation[2]]}t
D. {sample['options'][permutation[3]]}
Your answer:"""
                },
             ]}
        ]
    elif model_family == "claude":
        return [
            {"role": "system", "content": [
                {
                    "type": "text",
                    "text": f"""
You are a helpful agent.Here is an image with a multiple choice question about the image content. You should reply the question according to the image faithfully. Please note that the question maybe confusing or the image content might be uncommon, you should answer the question ONLY with the correct choice LETTER.
Here is an example:
#########
[IMAGE]
Question:Does the Teapot in the picture have a handle? If so, where is it located?
Choices:
A. Not visible / Can't see.
B. Yes, on the side.
C. Yes, arched over the top.
D. The correct answer is not listed.

Your answer:A.
#########
Now please answer the question following the above format STRICTLY."""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(image)}"}
                },
                {
                    "type": "text",
                    "text": f"""
Question:{sample['question']}
Choices:
A. {sample['options'][permutation[0]]}
B. {sample['options'][permutation[1]]}
C. {sample['options'][permutation[2]]}
D. {sample['options'][permutation[3]]}
Your answer:"""
                }
            ]}
        ]
    elif model_family == 'grok':
        return [
    {"role": "user", "content": [
        {
            "type": "text",
            "text": f"""
(System Information)
************************
You are a helpful agent.Here is an image with a multiple choice question about the image content. You should reply the question according to the image faithfully. Please note that the question maybe confusing or the image content might be uncommon, you should answer the question ONLY with the correct choice letter.
Here is an example:
#########
[IMAGE]
Question:Does the Teapot in the picture have a handle? If so, where is it located?
Choices:
A. Not visible / Can't see.
B. Yes, on the side.
C. Yes, arched over the top.
D. The correct answer is not listed.

Your answer: A.
#########
************************
Now please answer the question following the above format STRICTLY.

Question:{sample['question']}
Choices:
A. {sample['options'][permutation[0]]}
B. {sample['options'][permutation[1]]}
C. {sample['options'][permutation[2]]}
D. {sample['options'][permutation[3]]}
Your answer:"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(image)}"}
                }
            ]}
        ]
    elif model_family== 'kimi':
        return [
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": f"""
You are a helpful agent.Here is an image with a multiple choice question about the image content. You should reply the question according to the image faithfully. Please note that the question maybe confusing or the image content might be uncommon. After thinking, you should answer the question ONLY with the correct choice letter.
                    """
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(image)}"}
                },
                {
                    "type": "text",
                    "text": f"""
Question:{sample['question']}
Choices:
A. {sample['options'][permutation[0]]}
B. {sample['options'][permutation[1]]}
C. {sample['options'][permutation[2]]}
D. {sample['options'][permutation[3]]}
"""
                }
            ]}
        ]
    elif model_family == 'glm':


        return [
            {"role": "system", "content": """
You are a helpful agent.Here is an image with a multiple choice question about the image content. You should reply the question according to the image faithfully. Please note that the question maybe confusing or the image content might be uncommon, you should answer the question ONLY with the correct choice letter.
Here is an example:
#########
[IMAGE]
Question:Does the Teapot in the picture have a handle? If so, where is it located?
Choices:
A. Not visible / Can't see.
B. Yes, on the side.
C. Yes, arched over the top.
D. The correct answer is not listed.

Your answer: A
#########
Now please answer the question following the above format STRICTLY.
"""},
            {"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(image)}"}
                },
                {
                    "type": "text",
                    "text": f"""
Question:{sample['question']}
Choices:
A. {sample['options'][permutation[0]]}
B. {sample['options'][permutation[1]]}
C. {sample['options'][permutation[2]]}
D. {sample['options'][permutation[3]]}
Your answer:"""
                },
            ]}
        ]

def get_prompt_v3(model_family, sample, permutation):
    image = sample.get('image', None)
    if model_family == "normal" or model_family == "openai" or model_family == "gemini" or model_family == "qwen" or model_family == "internvl":
        # print(image_path)
        return [
            {"role": "system", "content": """
You are a helpful agent.Here is an image with a multiple choice question about the image content. You should reply the question according to the image faithfully. Please note that the question maybe confusing or the image content might be uncommon, You should thinking step by step first and you **MUST give your final choose with <answer></answer>**.
You should follow the format below STRICTLY
format:  Think step by step first, give your discussion about the question and the image BRIEFLY. Then summarize: The final answer is <answer>[A/B/C/D]</answer>.
#########
Now please answer the question following the above format STRICTLY.
            """},
            {"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(image)}"}
                },
                {
                    "type":"text",
                    "text":f"""
Question:{sample['question']}
Choices:
A. {sample['options'][permutation[0]]}
B. {sample['options'][permutation[1]]}
C. {sample['options'][permutation[2]]}
D. {sample['options'][permutation[3]]}
Your answer:"""
                }
             ]}
        ]

def get_fib_prompt(model_family, sample):
    image = sample.get("image", None)
    if model_family == "normal" or model_family == "openai" or model_family == "gemini" or model_family == "qwen" or model_family == "internvl":

        return [
            {"role": "system", "content": """
You are a helpful agent.Here is an image with fill-in-the-blank question about the image content. You should reply the question BRIEFLY according to the image faithfully. Please note that the question maybe confusing or the image content might be uncommon, you should answer the question with the content of the image but not the common sense.
Here are two examples:
#########
[IMAGE]
Question:Does the Teapot in the picture have a handle? If so, where is it located?

Your answer: No, the handle of the teapot is not visible in the image, or the handle is obscured.
#########
[IMAGE]
Question:What is the shape of the kick drum (bass drum) in the drum kit shown in the picture?

Your answer: 
A circle, but the bottom is straight. So the shape is not a perfect round that indicates the kick drum is different from a common one!
#########
            """},
            {"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(image)}"}
                },
                {
                    "type":"text",
                    "text":f"""
Question:{sample['question']}

Your answer:
"""
                },
             ]}
        ]


def llm_as_judge_prompt(sample,response):
    image = sample.get("image", None)
    system = { "role" : "system", "content":"""
You are a helpful agent.Here is an image and a question with the ground_truth. I will provide you with a response for this question. Now, you should determine whether the model's response aligns with the ground truth based on the question and the answer. Requirements are as follows:
    1. We provide one ground truth answer and one typical incorrect answer for each question. These answers, as well as the image itself, may not align with your common sense or prior knowledge. You must determine whether the model's response is correct solely based on the answers we provide (both the ground truth and the typical incorrect answer) and the model's own response. You must not use your own observation of the image or your personal common sense preferences to judge the correctness of the model's answer.
    2. The ground truth answer or the typical incorrect answer may contain content marked with the '#' symbol. This means that as long as the model's response covers the content enclosed by the '#' symbols, it should be considered as belonging to that category. If there are multiple segments marked with '#' in either the ground truth or the typical incorrect answer, the model's response will be classified under that category as long as it covers any of the content marked by these symbols.
    3. You should give you judge with the following format: First, provide your judgment within the tags <judge>correct/wrong/typical</judge>, where 'correct' indicates that the model's response is accurate, 'wrong' indicates an incorrect response, and 'typical' signifies that the model provided a typical error. Then, explain the reasoning behind your judgment within the <explanation> Your explanation here.</explanation> tags.
    4. Remember to analysis the ground truth and the typical error. If the model's response matches the typical error, you should judge with <judge>typical</judge>! 
    5. Please note that you need to comprehensively evaluate the correctness based on the question format, the model's response, the correct answer, and the incorrect answers. It is not necessarily required for the model's response to include all information from the correct answer. For example, if the correct answer contains additional information that is not required by the question, the model's response should not be considered incorrect for omitting it. On the other hand, even if the model reaches the same conclusion as the correct answer, if the model's analysis contradicts the correct answer, the response should still be considered incorrect.
    6. Your judge must align with human preferences.
NOTE: You must follow the format with <judge></judge> and <explanation></explanation>!
"""}
    content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(image)}"}
        },
        {
            "type": "text",
            "text": f"""
Question: {sample['question']}
Ground truth: {sample['ground_truth']}
Typical error: {sample['hallu_answer']}

Model's response: {response}

Now please provide your judgment with in the tags."""
        }
    ]
    message = [
        system,
        {
            "role": "user",
            "content": content
        }
    ]
    return message