from typing import Union
from transformers import AutoTokenizer, LogitsProcessorList, AutoModelForCausalLM
MODEL_PATH = 'THUDM/glm-4-9b-chat-hf'
import os
# 设置 HTTP 代理
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'

# 设置 HTTPS 代理 
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto").eval()

def process_model_outputs(inputs, outputs, tokenizer):
    responses = []
    for input_ids, output_ids in zip(inputs.input_ids, outputs):
        response = tokenizer.decode(output_ids[len(input_ids):], skip_special_tokens=True).strip()
        responses.append(response)
    return responses


def batch(
        model,
        tokenizer,
        messages: Union[str, list[str]],
        max_input_tokens: int = 8192,
        max_new_tokens: int = 8192,
        num_beams: int = 1,
        do_sample: bool = True,
        top_p: float = 0.8,
        temperature: float = 0.8,
        logits_processor=None,
):
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    messages = [messages] if isinstance(messages, str) else messages
    batched_inputs = tokenizer(
        messages,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_input_tokens).to(model.device)
    
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "do_sample": do_sample,
        "top_p": top_p,
        "temperature": temperature,
        "logits_processor": logits_processor,
        "eos_token_id": model.config.eos_token_id
    }
    batched_outputs = model.generate(**batched_inputs, **gen_kwargs)
    batched_response = process_model_outputs(batched_inputs, batched_outputs, tokenizer)
    return batched_response


def gen_caption(batch_message):

    # batch_message = [
    #     [
    #         {"role": "user", "content": "我的爸爸和妈妈结婚为什么不能带我去"},
    #         {"role": "assistant", "content": "因为他们结婚时你还没有出生"},
    #         {"role": "user", "content": "我刚才的提问是"}
    #     ],
    #     [
    #         {"role": "user", "content": "你好，你是谁"}
    #     ]
    # ]

    batch_inputs = []
    max_input_tokens = 128
    for i, messages in enumerate(batch_message):
        new_batch_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)[12:]
        max_input_tokens = max(max_input_tokens, len(new_batch_input))
        batch_inputs.append(new_batch_input)
    gen_kwargs = {
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": 256,
        "do_sample": True,
        "top_p": 0.8,
        "temperature": 0.8,
        "num_beams": 1,
    }

    batch_responses = batch(model, tokenizer, batch_inputs, **gen_kwargs)
    return batch_responses

def gen_batch_ques(ques):
    out_put=[]
    for i in ques:
        temp_dic={"role": "user"}
        temp_dic["content"]="Extract useful information about pedestrian appearance from the following sentences, remove all environmental information, descriptions about image clarity, person orientation, or visibility of certain appearance features, and rephrase it using a single passage of natural language:"+str(i)
        out_put.append(temp_dic)
    return out_put

#生成用
import os
import json

with open("./output_merged.json", 'r', encoding='utf-8') as file:
    data_re_id = json.load(file)

print(len(data_re_id))
# for i in range(0, len(data_re_id), 4):
#     chunk = data_re_id[i:i+4]
#     print(chunk)
# 按批次处理
batch_size = 2
count=0
for batch_start in range(0, len(data_re_id), batch_size):
    batch_data = data_re_id[batch_start:batch_start + batch_size]
    # print(f"Processing batch {batch_start // batch_size + 1} with {len(batch_data)} items.")
    batch_ques=[]
    for i in batch_data:
        generated_caption = i['generated_captions'][0]
        sentences = generated_caption.split('.')  # 分割并限制句子数
        sentences = list(set(s.strip() for s in sentences if s.strip()))  # 去重
        ques = gen_batch_ques(sentences)
        batch_ques.append(ques)
    # print("Generated questions for batch")
    responses = gen_caption(batch_ques)
    for idx, i in enumerate(batch_data):
        i['captions'] = str(responses[idx]).split('.')
        i['remake']="1"
    count=count+1
    if count%5==0:
        with open("./reid_raw.json", "w", encoding="utf-8") as f:
            json.dump(data_re_id, f, ensure_ascii=False, indent=4)
        print("40026/",count*2)


with open("./reid_raw.json", "w", encoding="utf-8") as f:
    json.dump(data_re_id, f, ensure_ascii=False, indent=4)