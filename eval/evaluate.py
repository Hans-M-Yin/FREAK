import base64
import mimetypes
import re

import tempfile
import uuid
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm
import json
import json
import os
import argparse
import requests
from multiprocessing import Pool, cpu_count
import prompt_adapter
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='qwen', help='Model name for result save')
parser.add_argument('--judge_model_name', type=str, default='qwen', help='Model name for judgement')
parser.add_argument("--parallel_num", type=int, default=int(0.6 * cpu_count()))
parser.add_argument('--api_key', type=str, default='EMPTY', help='API key')
parser.add_argument('--api_url', type=str, default='', help='API URL')
parser.add_argument('--save_path', type=str, default='./', help='Path to save the results')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--cot',action='store_true', help='use CoT prompting')
parser.add_argument('--thinking', action='store_true', help='enable reasoning mode for reasoning models.')
parser.add_argument('--bot',type=str,default='<think>')
parser.add_argument('--eot',type=str,default='</think>')

# Sampling Prams
parser.add_argument('--temperature',type=float,default=0)
parser.add_argument('--repetition_penalty',type=float,default=1.0)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--top_p',type=float,default=1.0)




args = parser.parse_args()

sampling_params = {
    "temperature": args.temperature,
    "seed": args.seed,
    "top_p": args.top_p
}


openai_api_key = args.api_key
openai_api_base = args.api_url

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

client_judge = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
    base_url=os.environ['OPENAI_BASE_URL']
)

eval_model_name = args.model_name
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
def encode_image(image_path):
    if image_path.startswith("http"):
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
        request_kwargs = {
            "headers": {"User-Agent": user_agent},
            "stream": True,
        }

        # Send a HTTP request to the URL
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


permutation_list = [
    [0, 1, 2, 3],
    [0, 2, 1, 3],
    [1, 0, 2, 3],
    [1, 2, 0, 3],
    [2, 0, 1, 3],
    [2, 1, 0, 3]
]
permutation_list2 = [
    ['A', 'B', 'C', 'D'],
    ['A', 'C', 'B', 'D'],
    ['B', 'A', 'C', 'D'],
    ['C', 'A', 'B', 'D'],
    ['B', 'C', 'A', 'D'],
    ['C', 'B', 'A', 'D']
]


def extract_thinking_and_summary(text: str, bot: str = "<think>", eot: str = "</think>") -> str:

    if eot in text and bot in text:

        return text[text.index(bot) + len(bot):text.index(eot)].strip(), text[text.index(eot) + len(eot) :].strip()
    return "", text


def get_prompt(sample,perm,perm_rev,cot="False"):
    model_family = "normal"

    if cot:
        if sample['type'] == 'mcq':
            return prompt_adapter.get_prompt_v3(model_family,sample,perm)
        else:
            # TODO
            pass
    else:
        if sample['type'] == 'mcq':
            return prompt_adapter.get_prompt(model_family,sample,perm)
        else:
            return prompt_adapter.get_prompt_qa(model_family, sample)

def llm_as_judge(sample, model_response):

    prompt = prompt_adapter.llm_as_judge_prompt(sample,model_response)
    response = client_judge.chat.completions.create(
        model=args.judge_model_name,
        messages=prompt,
        **sampling_params
    )
    response = response.choices[0].message.content
    judge = re.findall(r'<judge>(.*?)</judge>', response, re.DOTALL)
    if len(judge) > 0:
        judge = judge[0].lower()
        # raise Exception(f"LLM judgement without format. :{response}")
    score = 0
    correct = False
    if judge == 'correct':
        correct = True
        score = 1
    elif judge == 'typical':
        score = -1
    return correct, score, response


def run_model_sample(sample):
    cot = args.cot
    thinking_mode = args.thinking
    bot_token = args.bot
    eot_token = args.eot

    responses = []
    scores = []
    corrects = []
    choices = []
    for permutation,permutation_reverse in zip(permutation_list,permutation_list2):

        prompt = get_prompt(sample, permutation, permutation_reverse,cot)
        response = client.chat.completions.create(
            model=eval_model_name,
            messages=prompt,
            max_tokens=6000,
            **sampling_params,

        )
        response = response.choices[0].message.content

        if sample['type'] == 'mcq':

            if thinking_mode:
                thinking, summary = extract_thinking_and_summary(response,bot_token,eot_token)
                choice = summary
            else:
                choice = response
            score = 0
            response = response.strip()

            pattern = r'<answer>(.*?)</answer>'
            choice1 = re.findall(pattern, choice, re.DOTALL)
            if len(choice1) >= 1:
                choice = choice1[0]
            if choice == 'A' or choice == 'B' or choice == 'C' or choice == 'D':
                choice = choice[0]
            else:
                if choice[-1] == 'A':
                    choice = 'A'
                elif choice[-1] == 'B' in choice:
                    choice = 'B'
                elif choice[-1] == 'C' in choice:
                    choice = 'C'
                elif choice[-1] == 'D' in choice:
                    choice = 'D'
                else:
                    # print(choice)
                    if 'A.' in choice or 'is A' in choice or 'is: A' in choice or "{A}" in choice:
                        choice = 'A'
                    elif 'B.' in choice or 'is B' in choice or 'is: B' in choice or "{B}" in choice:
                        choice = 'B'
                    elif 'C.' in choice or 'is C' in choice or 'is: C' in choice or "{C}" in choice:
                        choice = 'C'
                    elif 'D.' in choice or 'is D' in choice or 'is: D' in choice or "{D}" in choice:
                        choice = 'D'
                    else:
                        choice = 'D'
                        print(response)
                        print(sample['id'])
            correct = False
            if choice == permutation_reverse[0]:
                score = 1
                correct = True
            elif choice == (permutation_reverse[1]):
                score = -1
            responses.append(response)
            choices.append(choice)
            corrects.append(correct)
            scores.append(score)
        answer = {
            "id": sample['id'],
            "type": sample['type'],
            "responses":responses,
            "choices":choices,
            "scores":scores,
            "corrects":corrects
        }
    else:
        correct, score, judge_detail = llm_as_judge(sample, response)
        answer = {
            "id": sample['id'],
            "type": sample['type'],
            "responses": response,
            "scores": score,
            "corrects": correct,
            "llm_judge_detail": judge_detail
        }
    return answer

def run_model():
    # with open("dataset.json",'r',encoding="utf-8") as f:
    #     dataset = json.load(f)
    dataset, mcq_len, fib_len = utils.load_freaK_dataset_from_local()
    category = {
        "counting":0,
        "detection":0,
        "shape":0,
        "color":0,
        "logic":0,
        "ocr":0,
        "position":0
    }
    for i in dataset:
        if type(i["category"]) is list:
            for k in i["category"]:
                category[k] += 1
        else:
            category[i['category']] += 1
    thinking = False
    if args.thinking == 'True':
        thinking = True
    json_answer = []
    with Pool(processes=args.parallel_num) as pool:
        json_answer = list(tqdm(pool.imap(run_model_sample,dataset), total=len(dataset)))
    # for idx,i in enumerate(tqdm(dataset)):
    #     json_answer.append(run_model_sample(i))
    mcq_correct, mcq_hallu = 0, 0
    fib_correct, fib_hallu = 0, 0
    total_correct, total_hallu = 0, 0
    for i in json_answer:
        if i['type'] == 'mcq':
            mcq_correct += sum(i['corrects']) / 6
            mcq_hallu += sum([1 for j in i['scores'] if j == -1]) / 6
            total_correct += sum(i['corrects']) / 6
            total_hallu += sum([1 for j in i['scores'] if j == -1]) / 6
        else:
            if i['scores'] == 1:
                fib_correct += 1
                total_correct += 1
            elif i['scores'] == -1:
                fib_hallu += 1
                total_hallu += 1
    print(f"##################\nMetrics\n##################\n")
    print(f"Total correct:{(total_correct / len(dataset)):8.2f} Total Hallu: {(total_hallu / len(dataset)):8.2f} ")
    print(f"MCQ   correct:{(mcq_correct / mcq_len):8.2f} MCQ   Hallu: {(mcq_hallu / mcq_len):8.2f} ")
    print(f"FIB   correct:{(fib_correct / fib_len):8.2f} FIB   Hallu: {(fib_hallu / fib_len):8.2f} ")
    # print("SCORE:",score / (6 * len(json_answer)), ' CORRECT:',correct)
    with open(f"{eval_model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{args.cot}.json",'w',encoding='utf-8') as f:
        json.dump(json_answer, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # print('PROMPT TYPE:',args.cot, 'URL:',args.api_url)
    run_model()