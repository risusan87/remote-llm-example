import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_ollama import ChatOllama
from openai import OpenAI

# load_dotenv()  # WARNING: May be redundant Yes it is because already loaded in manage.py

class APIModel:
    def __init__(self, base_url:str, api_key:str, model: str):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model

    def call_inference(self, messages: list[dict]):
        print("calling")
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
        )
        print("calling")
        log = ''
        for chunk in response:
            log += str(chunk) + '\n'
            yield chunk
        with open('backend/power/data/logs/api_model.log', 'w') as f:
            f.write(log)

class OllamaAPIModel(APIModel):
    """
    LLM model using Ollama API
    """
    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        top_p: float = 0.5,
        top_k: float = 50,
        reasoning: str | bool = "low",
        tools: list | None = None,
    ):
        self._llm = ChatOllama(
            model=model,
            validate_model_on_init=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            reasoning=reasoning,
            base_url=os.getenv("OLLAMA_HOST"),
            tools=tools,
        )
        self.model_name = model

    def call_inference(self, messages: list[dict] | list[BaseMessage]):
        # print(f"Called {self.call_inference.__name__}\n\n")
        langchain_messages = messages if isinstance(messages[0], BaseMessage) else []
        for turn in messages:
            if isinstance(turn, BaseMessage):
                continue
            if turn['role'] == 'system':
                langchain_messages.append(SystemMessage(content=turn['content']))
            elif turn['role'] == 'user':
                langchain_messages.append(HumanMessage(content=turn['content']))
            elif turn['role'] == 'assistant':
                langchain_messages.append(AIMessage(content=turn['content']))
            else:
                raise ValueError(f"Unknown role: {turn['role']}")

        # print(str(messages) + "\n\n")
        response = self._llm.stream(langchain_messages)
        # log = ""

        for chunk in response:
            content = str(chunk.content)
            # log += content + "\n"
            yield content

        # with open("backend/power/logs/api_model.log", "w") as f:
        #     f.write(log)

class DeepseekAPIModel(APIModel):
    def __init__(self, api_key: str, model_name:str):
        super().__init__(base_url='https://api.lambda.ai/v1', api_key=api_key, model=model_name)

    def call_inference(self, messages: list[dict]):
        reasoning = False
        next_token = ''
        for chunk in super().call_inference(messages):
            content = chunk.choices[0].delta.content
            if content is None:
                next_token = chunk.choices[0].delta.reasoning_content
                if reasoning is False:
                    reasoning = True
                    next_token = "<think>" + "" if not next_token else next_token
            elif reasoning is True:
                reasoning = False
                next_token = "</think>" + content
            else:
                next_token = content
            yield next_token

class DeepseekAPIModelCompletion(DeepseekAPIModel):
    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_key=api_key, model_name='deepseek-llama3.3-70b')

    def call_inference(self, literal_prompt: str, system_literal_prompt: str):
        onetime_conversation = [
            {
                "role": "system",
                "content": system_literal_prompt
            },
            {
                "role": "user",
                "content": literal_prompt
            }
        ]
        yield from super().call_inference(onetime_conversation)

class GptOssAPIModel(APIModel):
    def __init__(self, api_key: str=None, model_name: str=None):
        super().__init__(base_url='http://10.5.0.2:11434/v1', api_key='ollama', model='gpt-oss')
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_temperature",
                    "description": "Get the current temperature at a location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location to get the temperature for, in the format 'City, Country'"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The unit to return the temperature in."
                            }
                        },
                        "required": ["location", "unit"]
                    }
                }
            }
        ]
    
    def call_inference(self, messages: list[dict]):
        response = self._llm.chat.completions.create(
            tools=self.tools,
            model=self.model_name,
            messages=messages,
            stream=True,
        )
        chunk_log = ''
        for chunk in response:
            with open('backend/power/data/logs/gpt_oss_api_model.log', 'a') as f:
                f.write(str(chunk) + '\n')
            if chunk.choices[0].finish_reason is not None:
                break
            content = chunk.choices[0].delta.content
            if content != '':
                yield content
            else:
                yield chunk.choices[0].delta.reasoning

# oss = GptOssAPIModel()
# conv = [
#     {
#         "role": "user",
#         "content": "What is the current weather in Ottawa?"
#     }
# ]
# for token in oss.call_inference(conv):
#     print(token, end='', flush=True)
