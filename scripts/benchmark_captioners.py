from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)
import torch
from torchmetrics.functional.multimodal import clip_score

from BakLLaVA.llava.model.builder import (
    load_pretrained_model as bakllava_load_pretrained_model,
)
from BakLLaVA.llava.conversation import conv_templates as bakllava_conv_templates
from BakLLaVA.llava.mm_utils import (
    tokenizer_image_token as bakllava_tokenizer_image_token,
)


from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX


from deepseek_vl.models import VLChatProcessor

from functools import partial
import argparse
import json
from PIL import Image
import os
import time
from open_clip.tokenizer import _tokenizer
import open_clip
import matplotlib.pyplot as plt
import numpy as np
import random
import zipfile


from transformers.utils import logging
from textwrap import wrap

# from moondream import Moondream

logging.set_verbosity(40)
# logger = logging.get_logger("transformers")


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")


def convert_conversation_to_prompts(conversation):
    prompts = []
    messages = conversation.messages

    for i in range(0, len(messages), 2):
        prompt = {
            "role": messages[i][0],
            "content": (
                messages[i][1][0]
                if isinstance(messages[i][1], tuple)
                else messages[i][1]
            ),
            "images": [messages[i][1][1]] if isinstance(messages[i][1], tuple) else [],
        }
        response = {"role": messages[i + 1][0], "content": messages[i + 1][1]}
        prompts.extend([prompt, response])


def get_prompt(conv) -> str:
    """Get the prompt for generation."""
    system_prompt = conv.system_template.format(system_message=conv.system_message)
    if conv.sep_style == SeparatorStyle.DeepSeek:
        seps = [conv.sep, conv.sep2]
        if system_prompt == "" or system_prompt is None:
            ret = ""
        else:
            ret = system_prompt + seps[0]
        for i, (role, message) in enumerate(conv.messages):
            if message:
                if type(message) is tuple:  # multimodal message
                    message, _ = message
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ":"
        return ret
    else:
        return conv.get_prompt


def generate_prompt_with_history(
    text, image, history, vl_chat_processor, tokenizer, max_length=2048
):
    """
    Generate a prompt with history for the deepseek application.
    Args:
        text (str): The text prompt.
        image (str): The image prompt.
        history (list): List of previous conversation messages.
        tokenizer: The tokenizer used for encoding the prompt.
        max_length (int): The maximum length of the prompt.
    Returns:
        tuple: A tuple containing the generated prompt, image list, conversation, and conversation copy. If the prompt could not be generated within the max_length limit, returns None.
    """

    sft_format = "deepseek"
    user_role_ind = 0
    bot_role_ind = 1

    # Initialize conversation
    conversation = vl_chat_processor.new_chat_template()

    if history:
        conversation.messages = history

    if image is not None:
        if "<image_placeholder>" not in text:
            text = (
                "<image_placeholder>" + "\n" + text
            )  # append the <image_placeholder> in a new line after the text prompt
        text = (text, image)

    conversation.append_message(conversation.roles[user_role_ind], text)
    conversation.append_message(conversation.roles[bot_role_ind], "")

    # Create a copy of the conversation to avoid history truncation in the UI
    conversation_copy = conversation.copy()

    rounds = len(conversation.messages) // 2

    for _ in range(rounds):
        current_prompt = get_prompt(conversation)
        current_prompt = (
            current_prompt.replace("</s>", "")
            if sft_format == "deepseek"
            else current_prompt
        )

        if current_prompt.count("<image_placeholder>") > 2:
            for _ in range(len(conversation_copy.messages) - 2):
                conversation_copy.messages.pop(0)
            return conversation_copy

        if torch.tensor(tokenizer.encode(current_prompt)).size(-1) <= max_length:
            return conversation_copy

        if len(conversation.messages) % 2 != 0:
            raise "The messages between user and assistant are not paired."

        try:
            for _ in range(2):  # pop out two messages in a row
                conversation.messages.pop(0)
        except IndexError:
            raise "Input text processing failed, unable to respond in this round."

    raise "Prompt could not be generated within max_length limit."


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ):
        for stop in self.stops:
            if input_ids.shape[-1] < len(stop):
                continue
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True

        return False


def deepseek_eval(
    model,
    model_name,
    tokenizer,
    images,
    args,
    stopping_criteria=None,
    max_new_tokens=1024,
    image_processor=None,
):
    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>{args.prompt}",
        },
        {"role": "Assistant", "content": ""},
    ]

    stop_words = image_processor.new_chat_template().stop_str

    stop_words_ids = [
        torch.tensor(tokenizer.encode(stop_word)) for stop_word in stop_words
    ]

    conversations = []
    images_processed = []
    outputs = []
    stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=stop_words_ids)]
    )
    for image in images:
        conversations.append(conversation[0])
        conversations.append(conversation[1])

        prepare_inputs = image_processor(
            conversations=conversation, images=[image], force_batchify=True
        ).to(device=DEVICE)

        images_processed.append(prepare_inputs["pixel_values"].squeeze(0))

        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

        with torch.no_grad():
            output = model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                use_cache=True,
                stopping_criteria=stopping_criteria,
            )
            outputs.append(output)

    outputs = [
        tokenizer.batch_decode(output.cpu().tolist(), skip_special_tokens=True)[0]
        for output in outputs
    ]

    images_processed = torch.stack(images_processed).squeeze(1)
    print(images_processed.shape)

    return outputs, images_processed.to(torch.float16)


def cogvlm_eval(
    model,
    model_name,
    tokenizer,
    images,
    args,
    stopping_criteria=None,
    max_new_tokens=1024,
    image_processor=None,
):
    query = "USER: {} ASSISTANT:".format(args.prompt)

    outputs_list, images_processed = [], []
    input_by_model = model.build_conversation_input_ids(
        tokenizer, query=query, images=images
    )
    inputs = {
        "input_ids": input_by_model["input_ids"].unsqueeze(0).to(DEVICE),
        "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(DEVICE),
        "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(DEVICE),
        "images": [[input_by_model["images"].to(DEVICE).to(torch.float16)]],
    }
    if "cross_images" in input_by_model and input_by_model["cross_images"]:
        inputs["cross_images"] = [
            [input_by_model["cross_images"][0].to(DEVICE).to(torch.float32)]
        ]

    # add any transformers params here.
    gen_kwargs = {
        "max_length": max_new_tokens,
        "do_sample": False,
    }  # "temperature": 0.9
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(outputs[0])
        response = response.split("</s>")[0]
        outputs_list.append(response)
        images_processed.append(input_by_model["images"][0])
    images_processed = torch.stack(images_processed)
    return outputs_list, images_processed

def florence_eval(
    model,
    model_name,
    tokenizer,
    images,
    args,
    stopping_criteria=None,
    max_new_tokens=1024,
    image_processor=None,
    config=None,
):

    task_prompt = '<MORE_DETAILED_CAPTION>'
    prompt = task_prompt + args.prompt
    print(prompt)
    inputs = image_processor(text=[task_prompt]*len(images), images=images, return_tensors="pt").to(DEVICE, dtype=torch.float16)
    print(inputs["pixel_values"].shape)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=max_new_tokens,
      early_stopping=False,
      do_sample=True,
      num_beams=3,
    )

    generated_text = image_processor.batch_decode(generated_ids, skip_special_tokens=False)
   
    parsed_answer = [image_processor.post_process_generation(
        text, 
        task=task_prompt, 
        image_size=(images[0].width, images[0].height)
    )[task_prompt].replace("<pad>", "").strip() for text in generated_text]
    print(parsed_answer)

    return parsed_answer, inputs["pixel_values"]

def llava_eval(
    model,
    model_name,
    tokenizer,
    images,
    args,
    stopping_criteria=None,
    max_new_tokens=1024,
    image_processor=None,
    config=None,
):
    prompt = "<image>" + "\n" + args.prompt
    if "bakllava" in model_name:
        conv = bakllava_conv_templates[config["conv_mode"]].copy()
    else:
        conv = conv_templates[config["conv_mode"]].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if "bakllava" in model_name:
        input_ids = (
            bakllava_tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
    else:
        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

    input_ids = torch.cat([input_ids for _ in images])
    image_tensor = image_processor.preprocess(images, return_tensors="pt")[
        "pixel_values"
    ]
    images = image_tensor.half().cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stop_str = "\n" if "phi" in model_name else stop_str

    with torch.inference_mode(), torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=images,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            # stopping_criteria=stopping_criteria,
        )

    # print(output_ids.shape)

    outputs = [
        x.strip() for x in tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    ]

    return outputs, images

def get_tgt_sizes(images, patch_size):
    tgt_sizes = []
    for image in images:
        tgt_sizes.append(torch.Tensor(image.shape[0] // patch_size, image.shape[1] // patch_size))
    return tgt_sizes


def minicpmv_eval(
    model,
    model_name,
    tokenizer,
    images,
    args,
    stopping_criteria=None,
    max_new_tokens=1024,
    image_processor=None,
    config=None,
):
    # prompt = tokenizer.im_start \
    #         + tokenizer.unk_token * model.config.query_num \
    #         + tokenizer.im_end + "\n" + args.prompt
    # print(prompt)

    # test chat
    # copy images to avoid modifying the original images
    images_copy = list(images).copy()
    # convert to tensor
    images_copy = torch.stack([torch.tensor(np.array(img.resize((100,100)))) for img in images_copy])

    model = model.to(device='cuda')
    model.eval()
    model = model.to(dtype=torch.float16)


    copy_msgs = [{"role": "user", "content": [image.convert('RGB'), args.prompt]} for image in images]

    images = []
    tgt_sizes = []
    input_ids_list = []
    images_list = []
    tgt_sizes_list = []
    for i, msg in enumerate(copy_msgs):
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            content = [content]

        cur_msgs = []
        for c in content:
            if isinstance(c, Image.Image):
                image = c
                if model.config.slice_mode:
                    slice_images, image_placeholder = model.get_slice_image_placeholder(
                        image, tokenizer
                    )
                    cur_msgs.append(image_placeholder)
                    for slice_image in slice_images:
                        slice_image = model.transform(slice_image)
                        H, W = slice_image.shape[1:]
                        images.append(model.reshape_by_patch(slice_image))
                        tgt_sizes.append(torch.Tensor([H // model.config.patch_size, W // model.config.patch_size]).type(torch.int32))
                else:
                    images.append(model.transform(image))
                    cur_msgs.append(
                        tokenizer.im_start
                        + tokenizer.unk_token * model.config.query_num
                        + tokenizer.im_end
                    )
            elif isinstance(c, str):
                cur_msgs.append(c)
        
        msg['content'] = '\n'.join(cur_msgs)
        images_list.append(images)
        images = []

        if tgt_sizes:
            # tgt_sizes = torch.vstack(tgt_sizes)
            tgt_sizes_list.append(torch.vstack(tgt_sizes))
            tgt_sizes = []
        

    
    # input_ids = tokenizer.apply_chat_template(copy_msgs, tokenize=True, add_generation_prompt=False)
    input_ids_list = [tokenizer.apply_chat_template([msg], tokenize=True, add_generation_prompt=False) for msg in copy_msgs]


    generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05
            }



    with torch.inference_mode():
            outputs = model.generate(
                input_id_list=input_ids_list,
                img_list=images_list,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                return_vision_hidden_states=False,
                tgt_sizes=tgt_sizes_list,
                **generation_config
            )

    return outputs, images_copy



def coca_eval(
    model,
    model_name,
    tokenizer,
    images,
    args,
    stopping_criteria=None,
    max_new_tokens=1024,
    image_processor=None,
):
    images = torch.stack([image_processor(img) for img in images])
    with torch.no_grad(), torch.cuda.amp.autocast():
        out = model.generate(
            images.to(DEVICE),
            generation_type="top_k",
            max_seq_len=77,
            min_seq_len=5,
            top_k=50,
        )
    decoded = [
        open_clip.decode(i)
        .split("<end_of_text>")[0]
        .replace("<start_of_text>", "")
        .strip()
        for i in out
    ]
    return decoded, images


def moondream_eval(
    model,
    model_name,
    tokenizer,
    images,
    args,
    stopping_criteria=None,
    max_new_tokens=1024,
    image_processor=None,
):
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

    output = model.batch_answer(
        images=list(images),
        prompts=prompts,
        tokenizer=tokenizer,
    )
    return output, processed_images


def plot_images_captions(
    images, captions, plot_title="", plot_file=None, num_images=5, padding=10
):
    images, captions = zip(*random.sample(list(zip(images, captions)), num_images))
    fig, axs = plt.subplots(1, len(images), figsize=(20, 20))
    for i, ax in enumerate(axs):
        cap = captions[i]
        cap = "\n".join(wrap(cap, 30))
        ax.imshow(images[i])
        ax.set_title(cap, fontsize=12, wrap=True, horizontalalignment="center")
        # ax.text(cap, wrap=True, horizontalalignment='center', fontsize=12)
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_aspect('auto')
    plt.suptitle(plot_title, fontsize=20, wrap=True, horizontalalignment="center")
    plt.savefig(plot_file)


def eval_timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Elapsed time: {end-start}")
        return result

    return wrapper


class ZipDataset:
    def __init__(self, zip_file):
        self.zip_file = zip_file
        self.zip = zipfile.ZipFile(zip_file)
        self.image_files = self.zip.namelist()
        self.image_files = [x for x in self.image_files if x.endswith(".jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = Image.open(self.zip.open(image_file))
        return image, None, image_file


class Dataset:
    def __init__(self, images_dir, data_dir, batch_size):
        self.images_dir = images_dir

        with open(data_dir, "r") as f:
            self.data_list = json.load(f)["annotations"]

            self.data_list = self.data_list[: batch_size * 1]

            # self.data_list = [x for x in self.data_list if 'image' in x]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        image_id = data["image_id"]
        image_file = f"COCO_val2014_{image_id:012d}.jpg"

        # image_file = self.data_list[idx]['image']
        image = Image.open(f"{self.images_dir}/{image_file}")
        true_caption = data["caption"]
        return image, true_caption, image_file


def calculate_clip_score(images, prompts):
    images = np.array(images)
    images_int = (images * 255).astype("uint8")
    # clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    clip_score = clip_score_fn(torch.from_numpy(images_int), prompts).detach()
    return round(float(clip_score), 4)


def load_hf_model(model_name, args, model_path):
    image_processor = None

    if "bakllava" in model_name:
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = bakllava_load_pretrained_model(
            model_name=model_name,
            model_path=model_path,
            model_base=None,
            load_8bit=args.quant == 8,
            load_4bit=args.quant == 4,
        )
    elif "llava" in model_name:
        model_name = get_model_name_from_path(model_path)
        print(args.device_map)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_name=model_name,
            model_path=model_path,
            model_base=None,
            load_8bit=args.quant == 8,
            load_4bit=args.quant == 4,
            device_map=args.device_map,
        )

    elif "cogvlm" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            load_in_4bit=args.quant == 4,
            load_in_8bit=args.quant == 8,
            trust_remote_code=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            if (args.bf16 or args.quant == 4)
            else torch.float16,
        )
        tokenizer_path = "lmsys/vicuna-7b-v1.5"
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
    elif "moondream" in model_name:
        model_id = "vikhyatk/moondream2"
        revision = "2b705eea63f9bff6dae9b52c2daeb26bc10e4aeb"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            load_in_4bit=args.quant == 4,
            load_in_8bit=args.quant == 8,
            # bnb_4bit_compute_dtype=torch.bfloat16 if (args.bf16 or args.quant==4) else torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif "coca" in model_name:
        model, _, transform = open_clip.create_model_and_transforms(
            model_name, pretrained=model_path
        )
        tokenizer = _tokenizer
        image_processor = transform
        model = model.to(DEVICE)
        tokenizer = open_clip.get_tokenizer(model_name)
    elif "deepseek" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # trust_remote_code=True,
            load_in_4bit=args.quant == 4,
            load_in_8bit=args.quant == 8,
            bnb_4bit_compute_dtype=torch.bfloat16
            if (args.bf16 or args.quant == 4)
            else torch.float32,
        )
        image_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        tokenizer = image_processor.tokenizer
        # model = model.to(dtype).cuda()
    elif "minicpmv" in model_name:
        model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            # torch_dtype=torch.float16,
            load_in_4bit=args.quant == 4,
            load_in_8bit=args.quant == 8
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif "florence" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            load_in_4bit=True,
            # load_in_8bit=args.quant == 8,
            # bnb_4bit_compute_dtype=torch.bfloat16
            # if (args.bf16 or args.quant == 4)
            # else torch.float32,
        )
        # model = model.to(DEVICE)
        image_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = None
    # try:
    #     model = model.to(DEVICE, dtype=dtype)
    # except:
    #     pass

    print(model.dtype)

    return model.eval(), tokenizer, image_processor


def benchmark_captioner(model_name, args):
    model, tokenizer, image_processor = load_hf_model(
        model_name, args, MODELS_DICT[model_name]["path"]
    )
    if args.images_dir.endswith(".zip"):
        dataset = ZipDataset(args.images_dir)

    else:
        dataset = Dataset(args.images_dir, args.data_dir, args.batch_size)

    output_dir = args.output_dir

    output_file = f"{model_name}_{args.quant if args.quant else 'full'}bit_{args.batch_size}bs.json"

    output_file = os.path.join(output_dir, output_file)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x,
    )

    output_list = []
    total_time = 0
    for batch in dataloader:
        images, true_captions, image_file = zip(*batch)

        # images = images.to(DEVICE)

        eval_fn = MODELS_DICT[model_name]["eval_fn"]

        start = time.time()

        response, images_processed = eval_fn(
            model,
            model_name,
            tokenizer,
            images,
            args,
            stopping_criteria=None,
            max_new_tokens=args.max_new_tokens,
            image_processor=image_processor,
            config=MODELS_DICT[model_name],
        )
        clip_score = calculate_clip_score(images_processed.cpu(), response)

        end = time.time()

        eval_time = end - start
        total_time += eval_time

        samples_per_sec = len(images) / eval_time

        mem_usage = torch.cuda.memory_allocated(DEVICE) / 1024**2

        cpu_mem_usage = torch.cuda.memory_reserved(DEVICE) / 1024**2

        output = {
            "true_captions": true_captions,
            "response": response,
            "clip_score": clip_score,
            "eval_time": eval_time,
            "samples_per_sec": samples_per_sec,
            "mem_usage": mem_usage,
            "image_file": image_file,
            "cpu_mem_usage": cpu_mem_usage,
        }
        output_list.append(output)

        if args.plot:
            plot_file = os.path.join(
                output_dir,
                f"{model_name}_{args.quant if args.quant else 'full'}bit_{args.batch_size}bs.png",
            )
            plot_images_captions(
                images,
                response,
                plot_title=model_name,
                plot_file=plot_file,
                num_images=args.num_images_to_plot,
            )

    avg_time = total_time / len(output_list)
    with open(output_file, "w") as f:
        json.dump(
            {
                "output": output_list,
                "model_name": model_name,
                "prompt": args.prompt,
                "data_dir": args.data_dir,
                "quant": args.quant,
                "fp16": args.fp16,
                "bf16": args.bf16,
                "batch_size": args.batch_size,
                "images_dir": args.images_dir,
                "num_gpus": torch.cuda.device_count(),
                "num_cpus": os.cpu_count(),
                "gpu_memory_available": torch.cuda.get_device_properties(
                    DEVICE
                ).total_memory
                / 1024**3,
                "ram_memory_available": os.sysconf("SC_PAGE_SIZE")
                * os.sysconf("SC_PHYS_PAGES")
                / 1024**3,
                "sgl": "sgl" in MODELS_DICT[model_name],
                "image_size": MODELS_DICT[model_name].get("image_size"),
                "avg_time": avg_time,
                "total_time": total_time,
                "time_per_sample": avg_time / args.batch_size,
            },
            f,
            indent=4,
        )

        # print(f"Clip score: {clip_score}")
        # print(f"Response: {response}")


def get_captioner_func(m):
    if "llava" in m:
        eval_fn = llava_eval
    elif "cogvlm" in m:
        eval_fn = cogvlm_eval
    elif "moondream" in m:
        eval_fn = moondream_eval
    elif "coca" in m:
        eval_fn = coca_eval
    elif "deepseek" in m:
        eval_fn = deepseek_eval
    elif "minicpmv" in m:
        eval_fn = minicpmv_eval
    elif "florence" in m:
        eval_fn = florence_eval
    else:
        raise ValueError(f"Model {m} not found")
    return eval_fn


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--prompt",
        type=str,
        default="Describe this picture in about 50 words. Describe the important content and its spatial relationships to one another. In which areas of the picture can which content be seen and how are they arranged in relation to one another. Describe textures, shapes and lighting conditions. If people are visible in the picture, describe their appearance and the emotions they seem to express. If there is text in the picture, transcribe it in quotation marks and say where it can be seen.",
    )
    args.add_argument(
        "--quant", choices=[4, 8], type=int, default=None, help="quantization bits"
    )
    args.add_argument("--fp16", action="store_true")
    args.add_argument("--bf16", action="store_true")
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--images_dir", type=str, required=True)
    args.add_argument("--data_dir", type=str, required=True)
    args.add_argument("--output_dir", type=str, required=True)
    args.add_argument("--plot", action="store_true")
    args.add_argument("--config", type=str, default="config.json")
    args.add_argument("--max_new_tokens", type=int, default=1024)
    args.add_argument("--num_images_to_plot", type=int, default=5)
    args.add_argument("--device_map", type=str, default="auto")

    args = args.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    MODELS_DICT = config["models"]

    for m in MODELS_DICT.keys():
        if "llava" in m:
            eval_fn = llava_eval
        elif "cogvlm" in m:
            eval_fn = cogvlm_eval
        elif "moondream" in m:
            eval_fn = moondream_eval
        elif "coca" in m:
            eval_fn = coca_eval
        elif "deepseek" in m:
            eval_fn = deepseek_eval
        elif "minicpmv" in m:
            eval_fn = minicpmv_eval
        elif "florence" in m:
            eval_fn = florence_eval
        else:
            raise ValueError(f"Model {m} not found")

        MODELS_DICT[m]["eval_fn"] = eval_fn

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
