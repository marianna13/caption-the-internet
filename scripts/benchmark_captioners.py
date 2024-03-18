import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torchmetrics.functional.multimodal import clip_score
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from functools import partial
import argparse
import json
from PIL import Image
import re
import os
import time
from open_clip.tokenizer import _tokenizer
import open_clip
import matplotlib.pyplot as plt
import numpy as np

try:
    import sglang as sgl
    sgl_function = sgl.function
except ImportError:
    print("sglang not found")
    sgl_function = lambda x: x

from transformers.utils import logging
from textwrap import wrap

# from moondream import Moondream

logging.set_verbosity(40)
# logger = logging.get_logger("transformers")


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")


def cogvlm_eval(model, model_name, tokenizer, images, args, stopping_criteria=None, max_new_tokens=1024, image_processor=None):
    

    query = "USER: {} ASSISTANT:".format(args.prompt)


    dtype = torch.float16 if args.fp16 else torch.float32

            

    # if image is None:
    #     input_by_model = model.build_conversation_input_ids(tokenizer, query=query, template_version='base')
    # else:
    #     input_by_model = model.build_conversation_input_ids(tokenizer, query=query, images=[image])

    outputs_list, images_processed = [], []
    for image in images:
        input_by_model = model.build_conversation_input_ids(tokenizer, query=query, images=[image])
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(torch.float16)]],
        }
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch.float32)]]

        # add any transformers params here.
        gen_kwargs = {"max_length": max_new_tokens,
                        "do_sample": False} # "temperature": 0.9
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])
            response = response.split("</s>")[0]
            outputs_list.append(response)
            images_processed.append(input_by_model['images'][0])
    images_processed = torch.stack(images_processed)
    return outputs_list, images_processed
    
def llava_eval(model, model_name, tokenizer, images, args, stopping_criteria=None, max_new_tokens=1024, image_processor=None):

    prompt = '<image>' + '\n' + args.prompt
    conv = conv_templates[MODELS_DICT[model_name]['conv_mode']].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    input_ids = torch.cat(
        [
        input_ids for _ in images
        ]
    )
    image_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values']
    images = image_tensor.half().cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            # stopping_criteria=stopping_criteria,
        )

    outputs = [
        x.strip()
         for x in tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        ]
    

    return outputs, images

def coca_eval(model, model_name, tokenizer, images, args, stopping_criteria=None, max_new_tokens=1024, image_processor=None):

    images = torch.stack([image_processor(img) for img in images])
    with torch.no_grad(), torch.cuda.amp.autocast():
        out = model.generate(images.to(DEVICE), max_seq_len=max_new_tokens)
    decoded = [open_clip.decode(i).split("<end_of_text>")[0].replace("<start_of_text>", "").strip() for i in out]
    return decoded, images

def moondream_eval(model, model_name, tokenizer, images, args, stopping_criteria=None, max_new_tokens=1024, image_processor=None):

    processed_images = model.vision_encoder.preprocess(images)
    prompts = [args.prompt for _ in images]
    output = []

    processed_images = torch.stack(processed_images)
    # for img in images:
    #     image_embeds = model.encode_image(img)


    #     out = model.generate(
    #         image_embeds,
    #         f"<image>\n\nQuestion: {args.prompt}\n\nAnswer:",
    #         eos_text="<END>",
    #         tokenizer=tokenizer,
    #         max_new_tokens=max_new_tokens,
    #     )

    #     out = re.sub("<$|<END$", "", out[0]).strip()
    #     output.append(out)
    
    output =  model.batch_answer(
        images=list(images),
        prompts=prompts,
        tokenizer=tokenizer,
    )
    return output, processed_images

@sgl_function
def llava_eval_sgl(model, model_name, tokenizer, images, args, stopping_criteria=None, max_new_tokens=1024, image_processor=None):
    return llava_eval(model, model_name, tokenizer, images, args, stopping_criteria, max_new_tokens, image_processor)


def plot_images_captions(images, captions, plot_title="", plot_file=None, num_images=5, padding=10):
    images = images[:num_images]
    fig, axs = plt.subplots(1, len(images), figsize=(20, 20))
    for i, ax in enumerate(axs):
        cap = captions[i]
        cap = '\n'.join(wrap(cap, 30))
        ax.imshow(images[i])
        ax.set_title(cap, fontsize=12, wrap=True, horizontalalignment='center')
        # ax.text(cap, wrap=True, horizontalalignment='center', fontsize=12)
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_aspect('auto')
    plt.suptitle(plot_title, fontsize=20, wrap=True, horizontalalignment='center')
    plt.savefig(plot_file)


def eval_timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Elapsed time: {end-start}")
        return result
    return wrapper

class Dataset:
    def __init__(self, images_dir, data_dir, batch_size):
        self.images_dir = images_dir

        with open(data_dir, "r") as f:
            self.data_list = json.load(f)["annotations"]

            self.data_list = self.data_list[:batch_size*1]

            # self.data_list = [x for x in self.data_list if 'image' in x]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        data = self.data_list[idx]
        image_id = data['image_id']
        image_file = f'COCO_val2014_{image_id:012d}.jpg'
   
        # image_file = self.data_list[idx]['image']
        image = Image.open(f'{self.images_dir}/{image_file}').convert("RGB")
        true_caption = data['caption']
        return image, true_caption, image_file



def calculate_clip_score(images, prompts):
    images = np.array(images)
    images_int = (images* 255).astype("uint8")
    # clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    clip_score = clip_score_fn(torch.from_numpy(images_int), prompts).detach()
    return round(float(clip_score), 4)

def load_hf_model(model_name, args):

    image_processor = None

    dtype = torch.float16 if args.fp16 else torch.float32

    model_path = MODELS_DICT[model_name]['path']

    if 'llava' in model_name:
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_name=model_name, model_path=model_path, model_base=None, load_8bit=args.quant==8, load_4bit=args.quant==4)
      
    elif 'cogvlm' in model_name:
        model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                low_cpu_mem_usage=True,
                load_in_4bit=args.quant==4,
                load_in_8bit=args.quant==8,
                trust_remote_code=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if (args.bf16 or args.quant==4) else torch.float16,
                )
        tokenizer_path = 'lmsys/vicuna-7b-v1.5'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    elif 'moondream' in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            load_in_4bit=args.quant==4,
            load_in_8bit=args.quant==8,
            bnb_4bit_compute_dtype=torch.bfloat16 if (args.bf16 or args.quant==4) else torch.float32
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif 'coca' in model_name:
        model, _, transform = open_clip.create_model_and_transforms(model_name, pretrained=model_path)
        tokenizer = _tokenizer
        image_processor = transform
        model = model.to(DEVICE)
        tokenizer = open_clip.get_tokenizer(model_name)

    try:
        model = model.to(DEVICE, dtype=dtype)
    except:
        pass

    return model.eval(), tokenizer, image_processor



def benchmark_captioner(model_name, args):
    model, tokenizer, image_processor = load_hf_model(model_name, args)
    dataset = Dataset(args.images_dir, args.data_dir, args.batch_size)

    output_dir = args.output_dir

    output_file = f"{model_name}_{args.quant if args.quant else 'full'}bit_{args.batch_size}bs.json"

    output_file = os.path.join(output_dir, output_file)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=lambda x: x)

    output_list = []
    total_time = 0
    for batch in dataloader:

        images, true_captions, image_file = zip(*batch)

        # images = images.to(DEVICE)

        eval_fn = MODELS_DICT[model_name]['eval_fn']

        start = time.time()

        response, images_processed = eval_fn(model, model_name, tokenizer, images, args, stopping_criteria=None, max_new_tokens=2048, image_processor=image_processor)
        clip_score = calculate_clip_score(images_processed.cpu(), response)

        end = time.time()

        eval_time = end - start
        total_time += eval_time

        samples_per_sec = len(images) / eval_time

        mem_usage = torch.cuda.memory_allocated(DEVICE) / 1024 ** 2

        cpu_mem_usage = torch.cuda.memory_reserved(DEVICE) / 1024 ** 2

        output = {
            "true_captions": true_captions,
            "response": response,
            "clip_score": clip_score,
            "eval_time": eval_time,
            'samples_per_sec': samples_per_sec,
            'mem_usage': mem_usage,
            'image_file': image_file,
            'cpu_mem_usage': cpu_mem_usage,
        }
        output_list.append(output)

        if args.plot:
            plot_file = os.path.join(output_dir, f"{model_name}_{args.quant if args.quant else 'full'}bit_{args.batch_size}bs.png")
            plot_images_captions(images, response, plot_title=model_name, plot_file=plot_file)
    
    avg_time = total_time / len(output_list)
    with open(output_file, "w") as f:
        json.dump(
            {
                'output' :output_list,
                'model_name': model_name,
                'prompt': args.prompt,
                'data_dir': args.data_dir,
                'quant': args.quant,
                'fp16': args.fp16,
                'bf16': args.bf16,
                'batch_size': args.batch_size,
                'images_dir': args.images_dir,
                'num_gpus': torch.cuda.device_count(),
                'num_cpus': os.cpu_count(),
                'gpu_memory_available': torch.cuda.get_device_properties(DEVICE).total_memory / 1024 ** 3,
                'ram_memory_available': os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024 ** 3,
                'sgl': 'sgl' in MODELS_DICT[model_name],
                'image_size': MODELS_DICT[model_name].get('image_size'),
                'avg_time': avg_time,
                'total_time': total_time,
                'time_per_sample': avg_time / args.batch_size,
             }, 
             f, 
             indent=4
             )


        # print(f"Clip score: {clip_score}")
        # print(f"Response: {response}")


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--prompt", type=str, default="Describe the image")  
    args.add_argument("--quant", choices=[4, 8], type=int, default=None, help='quantization bits')
    args.add_argument("--fp16", action="store_true")
    args.add_argument("--bf16", action="store_true")
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--images_dir", type=str, required=True)
    args.add_argument("--data_dir", type=str, required=True)
    args.add_argument("--output_dir", type=str, required=True)
    args.add_argument("--plot", action="store_true")
    args.add_argument("--config", type=str, default="config.json")

    args = args.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    MODELS_DICT = config['models']

    for m in MODELS_DICT.keys():
        if 'llava' in m:
            if 'sgl' in MODELS_DICT[m]:
                eval_fn = llava_eval_sgl
            else:
                eval_fn = llava_eval
        elif 'cogvlm' in m:
            eval_fn = cogvlm_eval
        elif 'moondream' in m:
            eval_fn = moondream_eval
        elif 'coca' in m:
            eval_fn = coca_eval
        else:
            raise ValueError(f"Model {m} not found")

        MODELS_DICT[m]['eval_fn'] = eval_fn


    # MODELS_DICT = {
    #     'llava-v1.6-34b': {
    #         'path':'LLaVA/llava-v1.6-34b',
    #         'eval_fn': llava_eval,
    #         'conv_mode': 'v0'
    #         },
    #     'cogvlm-chat-hf': {
    #         'path':'LLaVA/llava-v1.6-34b',
    #         'eval_fn': cogvlm_eval
    #         },
    #     'llava-v1.6-mistral-7b': {
    #         'path':'LLaVA/llava-v1.6-34b',
    #         'eval_fn': llava_eval,
    #         'conv_mode': 'v1'
    #         },
    #     'moondream2': {
    #         'path':'LLaVA/llava-v1.6-34b',
    #         'eval_fn': moondream_eval
    #         },
    # }
        
    for model_name in MODELS_DICT.keys():
        print(f"Running benchmark for {model_name}")
        # try:
        benchmark_captioner(model_name, args)
        # except Exception as e:
        #     print(f"Error running benchmark for {model_name}: {e}")