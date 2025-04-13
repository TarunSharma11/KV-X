#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models
"""
import argparse
import logging

import numpy as np
import torch
import json
import tqdm 
import copy 

from src.monkeypatch import replace_llama
import transformers

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer , AutoTokenizer, AutoConfig
from transformers.models.llama.modeling_llama import LlamaAttention
from needle_in_a_haystack.prompt import Prompter

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_arch", type=str, default='llama')
    parser.add_argument("--model_name", type=str, default='huggyllama/llama-13b')
    parser.add_argument("--cache_dir", type=str, default='../../checkpoint/')
    parser.add_argument("--method", type=str, default='pyramidkv')

    parser.add_argument("--length", type=int, default=64)

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    args = parser.parse_args()

    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    config = LlamaConfig.from_pretrained(model_name)
    config._attn_implementation = LlamaAttention
    # config.transformers_version = "4.40"

    replace_llama(config)

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")
    set_seed(args)

    # Change to your custom prompt text
    # prompt_text = 'In the year 2087, humanity has achieved remarkable technological advancements and established colonies on multiple planets within the Milky Way galaxy. Interstellar travel has become commonplace, with faster-than-light spacecraft enabling people to explore distant star systems. Earth has undergone significant changes due to sustainable development efforts, such as harnessing renewable energy sources and implementing widespread ecological restoration projects. However, alongside these triumphs, new challenges have emerged, including the rise of artificial intelligence, ethical dilemmas surrounding genetic engineering, and interplanetary political tensions. Against this backdrop, a team of intrepid scientists embarks on a mission to uncover the secrets of an ancient alien civilization, hidden deep within an uncharted exoplanet. As they navigate treacherous terrains and encounter otherworldly phenomena, they must confront their own fears and reconcile humanity\'s thirst for knowledge with the potential consequences of uncovering secrets that were better left buried. The fate of both their mission and the future of humanity hang in the balance.'
    # inp = 'In a small, bustling cafe nestled in the heart of a vibrant city, a serendipitous event unfolded, leaving a lasting impression on all who witnessed it. As the patrons sat sipping their coffees and engaging in animated conversations, a talented street musician entered the cafe, carrying a weathered guitar and radiating an aura of creativity. Continue this story.'
    # prompt_text = "on what day was the United States founded?"
    # with open('/scratch/users/seliny2/prompt/prompt_300.txt', 'r') as file:
    #     prompt_text = file.read() 

    prompter = Prompter(
                tokenizer
            )
    context_len=1024
    # insert needle into context here
    context = prompter.generate_context(context_len, 100)
    inp = prompter.generate_prompt(context, context_len, 100)
    print(context)
    # messages = [
    #     {"role": "system", "content": "You are a story teller. Continue this story I give you."},
    #     {"role": "user", "content": {inp}},
    # ]
    messages = [
        {"role": "system", "content": f"Read the document below and answer this question: What is the best thing to do in San Francisco?"},
        {"role": "user", "content": {inp}},
    ]

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1])


if __name__ == "__main__":
    main()