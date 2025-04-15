"""Example of predicting next sequence in a code.

From the book:
    Machine Learning with Python: Keras, PyTorch, and TensorFlow:
    Unlocking the Power of AI and Deep Learning (Mastering AI and Python)

    By: Cuantum Technologies (Author)
"""
import numpy as np
from tensorflow.keras.layers import TextVectorization


def main():
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
    print(f"vocab: {vocab}")
    sequences = vectorizer(code_snippets)

    # Create a sequence of tf.Tensor[]
    input_sequence = []
    max_length = 0
    for seq in sequences:
        for i in range(1, len(seq)+1):
            current_seq = seq[:i]
            input_sequence.append(current_seq)
            max_length = max(max_length, len(current_seq))

    # Pad sequences and convert to numpy array
    padded_sequences = []
    for seq in input_sequence:
        padded_seq = np.pad(seq.numpy(), (0, max_length -
                            len(seq)), mode='constant', constant_values=0)
        padded_sequences.append(padded_seq)

    X = np.array(padded_sequences)
    print("Shape of X:", X.shape)
    print("X:", X, sep="\n")


if __name__ == "__main__":
    main()
