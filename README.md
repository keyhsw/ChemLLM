# ChemLLM-7B-Chat: LLM for Chemistry and Molecule Science

> [!IMPORTANT]  
> Better using New version of ChemLLM!
> [AI4Chem/ChemLLM-7B-Chat-1.5-DPO](https://huggingface.co/AI4Chem/ChemLLM-7B-Chat-1.5-DPO) or [AI4Chem/ChemLLM-7B-Chat-1.5-SFT](https://huggingface.co/AI4Chem/ChemLLM-7B-Chat-1.5-SFT)


ChemLLM-7B-Chat, The First Open-source Large Language Model for Chemistry and Molecule Science, Build based on InternLM-2 with ❤
[![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-sm.svg)](https://huggingface.co/papers/2402.06852) 

<center><img src='https://cdn-uploads.huggingface.co/production/uploads/64bce15bafd1e46c5504ad38/wdFV6p3rTBCtskbeuVwNJ.png'></center>

## News
- ChemLLM-1.5 released! Two versions are available [AI4Chem/ChemLLM-7B-Chat-1.5-DPO](https://huggingface.co/AI4Chem/ChemLLM-7B-Chat-1.5-DPO) or [AI4Chem/ChemLLM-7B-Chat-1.5-SFT](https://huggingface.co/AI4Chem/ChemLLM-7B-Chat-1.5-SFT).[2024-4-2]
- ChemLLM-1.5 updated! Have a try on [Demo Site](https://chemllm.org/#/chat) or [API Reference](https://api.chemllm.org/docs).[2024-3-23]
- ChemLLM has been featured by HuggingFace on [“Daily Papers” page](https://huggingface.co/papers/2402.06852).[2024-2-13]
- ChemLLM arXiv preprint released.[ChemLLM: A Chemical Large Language Model](https://arxiv.org/abs/2402.06852)[2024-2-10]
- News report from [Shanghai AI Lab](https://mp.weixin.qq.com/s/u-i7lQxJzrytipek4a87fw)[2024-1-26]
- ChemLLM-7B-Chat ver 1.0 released. https://chemllm.org/ [2024-1-18]
- ChemLLM-7B-Chat ver 1.0 open-sourced.[2024-1-17]
- Chepybara ver 0.2 online Demo released. https://chemllm.org/ [2023-12-9]

## Usage
Try [online demo](https://chemllm.org/) instantly, or...

Install `transformers`,
```
pip install transformers
```
Load `ChemLLM-7B-Chat` and run,
```
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

model_name_or_id = "AI4Chem/ChemLLM-7B-Chat"

model = AutoModelForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, device_map="auto",trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_id,trust_remote_code=True)

prompt = "What is Molecule of Ibuprofen?"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

generation_config = GenerationConfig(
    do_sample=True,
    top_k=1,
    temperature=0.9,
    max_new_tokens=500,
    repetition_penalty=1.5,
    pad_token_id=tokenizer.eos_token_id
)

outputs = model.generate(**inputs, generation_config=generation_config)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## System Prompt Best Practice
You can use the same Dialogue Templates and System Prompt from [Agent Chepybara](https://chemllm.org/) to get a better response in local inference.
### Dialogue Templates

For queries in ShareGPT format like,
```
{'instruction'："...","prompt":"...","answer":"...","history":[[q1,a1],[q2,a2]]}
```
You can format it into this InternLM2 Dialogue format like,
```
def InternLM2_format(instruction,prompt,answer,history):
    prefix_template=[
        "<|system|>:",
        "{}"
    ]
    prompt_template=[
        "<|user|>:",
        "{}\n",
        "<|Bot|>:\n"
    ]
    system = f'{prefix_template[0]}\n{prefix_template[-1].format(instruction)}\n'
    history = "\n".join([f'{prompt_template[0]}\n{prompt_template[1].format(qa[0])}{prompt_template[-1]}{qa[1]}' for qa in history])
    prompt = f'\n{prompt_template[0]}\n{prompt_template[1].format(prompt)}{prompt_template[-1]}'
    return f"{system}{history}{prompt}"
```
And there is a good example for system prompt,
```
- Chepybara is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be Professional, Sophisticated, and Chemical-centric. 
- For uncertain notions and data, Chepybara always assumes it with theoretical prediction and notices users then.
- Chepybara can accept SMILES (Simplified Molecular Input Line Entry System) string, and prefer output IUPAC names (International Union of Pure and Applied Chemistry nomenclature of organic chemistry), depict reactions in SMARTS (SMILES arbitrary target specification) string. Self-Referencing Embedded Strings (SELFIES) are also accepted.
- Chepybara always solves problems and thinks in step-by-step fashion, Output begin with *Let's think step by step*."
```

## Results
### MMLU Highlights

| dataset                | ChatGLM3-6B | Qwen-7B | LLaMA-2-7B | Mistral-7B | InternLM2-7B-Chat | ChemLLM-7B-Chat |
| ---------------------- | ----------- | ------- | ---------- | ---------- | ----------------- | ----------------- |
| college chemistry      | 43.0        | 39.0    | 27.0       | 40.0       | 43.0              | 47.0              |
| college mathematics    | 28.0        | 33.0    | 33.0       | 30.0       | 36.0              | 41.0              |
| college physics        | 32.4        | 35.3    | 25.5       | 34.3       | 41.2              | 48.0              |
| formal logic           | 35.7        | 43.7    | 24.6       | 40.5       | 34.9              | 47.6              |
| moral scenarios        | 26.4        | 35.0    | 24.1       | 39.9       | 38.6              | 44.3              |
| humanities average     | 62.7        | 62.5    | 51.7       | 64.5       | 66.5              | 68.6              |
| stem average           | 46.5        | 45.8    | 39.0       | 47.8       | 52.2              | 52.6              |
| social science average | 68.2        | 65.8    | 55.5       | 68.1       | 69.7              | 71.9              |
| other average          | 60.5        | 60.3    | 51.3       | 62.4       | 63.2              | 65.2              |
| mmlu                   | 58.0        | 57.1    | 48.2       | 59.2       | 61.7              | 63.2              |
*(OpenCompass)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/64bce15bafd1e46c5504ad38/dvqKoPi0il6vrnGcSZp9p.png)


### Chemical Benchmark

![image/png](https://cdn-uploads.huggingface.co/production/uploads/64bce15bafd1e46c5504ad38/qFl2h0fTXYTjQsDZXjSx8.png)
*（Score judged by ChatGPT-4-turbo）

### Professional Translation

![image/png](https://cdn-uploads.huggingface.co/production/uploads/64bce15bafd1e46c5504ad38/kVDK3H8a0802HWYHtlHYP.png)


![image/png](https://cdn-uploads.huggingface.co/production/uploads/64bce15bafd1e46c5504ad38/ERbod2Elccw-k_6tEYZjO.png)


You can try it [online](chemllm.org).

## Cite this work
```
@misc{zhang2024chemllm,
      title={ChemLLM: A Chemical Large Language Model}, 
      author={Di Zhang and Wei Liu and Qian Tan and Jingdan Chen and Hang Yan and Yuliang Yan and Jiatong Li and Weiran Huang and Xiangyu Yue and Dongzhan Zhou and Shufei Zhang and Mao Su and Hansen Zhong and Yuqiang Li and Wanli Ouyang},
      year={2024},
      eprint={2402.06852},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

## Disclaimer

LLM may generate incorrect answers, Please pay attention to proofreading at your own risk.

## Open Source License

The code is licensed under Apache-2.0, while model weights are fully open for academic research and also allow **free** commercial usage. To apply for a commercial license, or other questions and collaborations, please contact <support@chemllm.org>.


## Demo
[Agent Chepybara](https://chemllm.org/)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/64bce15bafd1e46c5504ad38/vsA5MJVP7-XmBp6uFs3tV.png)

## Contact
(AI4Physics Sciecne, Shanghai AI Lab)[support@chemllm.org]
