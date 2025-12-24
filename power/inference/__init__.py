# need documentation here,

# when llm (like chat gpt) is used in an app like this, there are TWO ways to perform inference:
# 1. API calls to an external service (like OpenAI API, Ollama, etc)
# 2. Local inference using a local model (inference via transformers library or at lowest level, tensor-wise operations on GPU via pytorch or tensorflow)

# API calls are a lot easier because it is built on top of needs of local inference setup.
# usually those services charge per token or per request.

# Local inference is more complex and requires domain knowledge of transformers as an AI model and GPU usage.
# If you have a powerful GPU (like GeForce RTX 5090/4090 or data center GPU) and appropriate knowledge, then this is the BEST choice for cost efficiency and data privacy.
# however if you rely on cloud GPU services like AWS or Lambda AI, then API calls are a lot cheaper option.



