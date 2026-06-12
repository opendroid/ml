from langchain_ollama import OllamaLLM
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.tracers.langchain import LangChainTracer
import os


class OllamaWrapper:
    def __init__(self, model_name="llama3:70b", enable_tracing=True):
        """
        Initialize the Ollama LLM wrapper.

        Args:
            model_name (str): Name of the Ollama model to use
            (default: "llama3:70b")
            enable_tracing (bool): Whether to enable LangSmith tracing
            (default: True)
        """
        callbacks = [StreamingStdOutCallbackHandler()]

        # Set up tracing if enabled and API key is available
        if enable_tracing:
            os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = os.getenv(
                "LANGSMITH_PROJECT")
            callbacks.append(LangChainTracer())

        # Initialize the Ollama model with streaming output and tracing
        self.llm = OllamaLLM(
            model=model_name,
            callbacks=callbacks,
            tags=["ollama", model_name],  # Add tags for better tracing
            metadata={"client": "local"}  # Add metadata for tracing
        )

    def generate(self, prompt: str) -> str:
        """
        Generate a response from the model.

        Args:
            prompt (str): The input prompt to send to the model

        Returns:
            str: The model's response
        """
        return self.llm.invoke(prompt)


def main():
    # Example usage
    # Disable tracing for this example
    ollama = OllamaWrapper(enable_tracing=True)
    response = ollama.generate("What is the capital of California?")
    print("\nResponse:", response)


if __name__ == "__main__":
    main()
