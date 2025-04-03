"""This module tests various pipeline tasks available on Hugging Face.
Note that the huggingface models are cached in $HOME/.cache/huggingface/hub

The task parameters:
    https://huggingface.co/docs/transformers/main_classes/pipelines
The models: https://huggingface.co/models

"""


from transformers import pipeline
from device import device
import gc


# task: text-generation, model:
llama_3_1 = pipeline("text-generation",
                     model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                     max_length=512, device=device)

response = llama_3_1(
    [{"role": "user", "content": "How many parameters are in Llama3.1?"},],
    num_return_sequences=2)
print(f"LLAMA: {response}")
del llama_3_1
gc.collect()

classifier = pipeline("sentiment-analysis",
                      model="cardiffnlp/twitter-roberta-base-sentiment",
                      device=device)
response = classifier("I think risk of this is very real - Elon Musk")
print(f"sentiment-analysis: {response}")
del classifier
gc.collect()
