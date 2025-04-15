"""This module tests various pipeline tasks available on Hugging Face.
Note that the huggingface models are cached in $HOME/.cache/huggingface/hub

The task parameters:
    https://huggingface.co/docs/transformers/main_classes/pipelines
The models: https://huggingface.co/models

"""


from transformers import pipeline
from device import device


def test_llama_3_1():
    llama_3_1 = pipeline("text-generation",
                         model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                         max_length=512, device=device)

    prompt = "How many parameters are in Llama3.1?"
    responses = llama_3_1(
        [{"role": "user", "content": prompt},],
        num_return_sequences=3)
    print(f"Prompt: {prompt}")
    for i, option in enumerate(responses):
        for response in option['generated_text']:
            if response['role'] == "assistant":
                print(f"\tLLAMA {i}: {response['content']}")


def test_sentiment_analysis():
    classifier = pipeline("sentiment-analysis",
                          model="cardiffnlp/twitter-roberta-base-sentiment",
                          device=device)
    prompt = "I think risk of this is very real - Elon Musk"
    responses = classifier(prompt)
    print(f"Sentiment score for: {prompt}")
    for response in responses:
        print(f"\t{response['label']}: {response['score']:0.2f}")


def main():
    # task: text-generation, model:
    test_llama_3_1()
    test_sentiment_analysis()


if __name__ == "__main__":
    main()
