"""Example of predicting next sequence in a code.

From the book:
    Machine Learning with Python: Keras, PyTorch, and TensorFlow:
    Unlocking the Power of AI and Deep Learning (Mastering AI and Python)

    By: Cuantum Technologies (Author)
"""
from tensorflow.keras.layers import TextVectorization  # type: ignore
import numpy as np

code_snippets = ["def hello_world():",
                 "print('Hello, world!')",
                 "if __name__ == '__main__':",
                 "hello_world()"]

vectorizer = TextVectorization(
    max_tokens=None,  # No limit on vocabulary size
    standardize=None,  # No need for text normalization for code
    split="whitespace",
    output_mode="int",
    output_sequence_length=None  # No padding/truncation for now
)
# Adapt the layer, vectorize and get sequences
vectorizer.adapt(code_snippets)
print(f"vectorizer: {vectorizer}")
vocab = vectorizer.get_vocabulary()
sequences = vectorizer(code_snippets)

# Create a sequence of tf.Tensor[]
input_sequence = []
for seq in sequences:
    for i in range(1, len(seq)+1):
        input_sequence.append(seq[:i])


X = np.ndarray([np.array(xi) for xi in input_sequence])
print(X)
