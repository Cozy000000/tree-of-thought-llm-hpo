# import os
# import openai
# import backoff 

# completion_tokens = prompt_tokens = 0

# api_key = os.getenv("OPENAI_API_KEY", "")
# if api_key != "":
#     openai.api_key = api_key
# else:
#     print("Warning: OPENAI_API_KEY is not set")
    
# api_base = os.getenv("OPENAI_API_BASE", "")
# if api_base != "":
#     print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
#     openai.api_base = api_base

# @backoff.on_exception(backoff.expo, openai.error.OpenAIError)
# def completions_with_backoff(**kwargs):
#     return openai.ChatCompletion.create(**kwargs)

# def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
#     messages = [{"role": "user", "content": prompt}]
#     return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
# def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
#     global completion_tokens, prompt_tokens
#     outputs = []
#     while n > 0:
#         cnt = min(n, 20)
#         n -= cnt
#         res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
#         outputs.extend([choice.message.content for choice in res.choices])
#         # log completion tokens
#         completion_tokens += res.usage.completion_tokens
#         prompt_tokens += res.usage.prompt_tokens
#     return outputs
    
# def gpt_usage(backend="gpt-4"):
#     global completion_tokens, prompt_tokens
#     if backend == "gpt-4":
#         cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
#     elif backend == "gpt-3.5-turbo":
#         cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
#     elif backend == "gpt-4o":
#         cost = completion_tokens / 1000 * 0.00250 + prompt_tokens / 1000 * 0.01
#     return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}


import os
import openai
import backoff 
from transformers import LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

completion_tokens = prompt_tokens = 0

# 全局变量用于存储本地模型和tokenizer
local_model = None
local_tokenizer = None

api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")
    
api_base = os.getenv("OPENAI_API_BASE", "")
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def load_local_model(model_name="/data/llama-2/Llama-2-7b-hf"):
    global local_model, local_tokenizer
    if local_model is None:
        print(f"Loading local model: {model_name}")
        try:
            local_tokenizer = LlamaTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                trust_remote_code=True
            )
        except Exception as e:
            print("LlamaTokenizer failed, fallback to AutoTokenizer:", e)
            local_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                trust_remote_code=True
            )
        local_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Local model loaded successfully")

def local_model_call(messages, max_tokens=1000, temperature=0.7):
    """调用本地模型生成回复"""
    global local_model, local_tokenizer
    if local_model is None:
        load_local_model()
    
        # 兼容字符串 prompt 输入
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    
    # 将对话历史转换为Llama2格式的提示
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    
    #system_prompt = "You are a helpful, respectful and honest assistant."
    
    if messages[0]["role"] == "system":
        system_prompt = messages[0]["content"]
        messages = messages[1:]
    
    prompt = f"{B_INST} {B_SYS}{system_prompt}{E_SYS}"
    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            prompt += f"{msg['content']} {E_INST}"
        else:
            prompt += f" {msg['content']} </s><s> {B_INST}"
    
    # 生成回复
    inputs = local_tokenizer(prompt, return_tensors="pt").to(local_model.device)
    outputs = local_model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        pad_token_id=local_tokenizer.eos_token_id
    )
    
    # 解码并提取新生成的文本
    full_output = local_tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_output[len(prompt):].strip()
    return response

def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    
    # 判断是否为本地模型调用
    if model.startswith("local/"):
        model_name = model[6:]  # 移除"local/"前缀
        outputs = []
        for _ in range(n):
            response = local_model_call(
                messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            outputs.append(response)
        return outputs
    
    # 原有OpenAI API调用
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice.message.content for choice in res.choices])
        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    return outputs

def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    # 本地模型不计算token消耗
    if backend.startswith("local/"):
        return {"completion_tokens": 0, "prompt_tokens": 0, "cost": 0}
    
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif backend == "gpt-4o":
        cost = completion_tokens / 1000 * 0.00250 + prompt_tokens / 1000 * 0.01
    else:
        cost = 0
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}