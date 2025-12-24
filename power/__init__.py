import os
from typing import (
    Sequence,
    Union,
    Optional,
    Literal,
    Callable,
    Any
)
from datetime import datetime
import json

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel, 
    ChatResult, 
    BaseMessage, 
)
from langchain_core.messages import (
    HumanMessage, 
    AIMessage, 
    SystemMessage,
    ToolMessage
)
from langchain_core.tools.base import BaseTool
from langchain_core.runnables import Runnable
from langchain_core.outputs import ChatGeneration, ChatResult
import openai_harmony as hmny
import modal

if os.environ.get('RUN_MAIN', 'false') == 'true':
    from backend.logger import power_logger as logger
    logger.info("This should be observed only once and it means power backend is starting up.")
from backend.power.inference.serverless import P2PEncryption, VLLMModel


class ChatGptOss(BaseChatModel):
    """
    this uses Modal vllm serverless function to run inference.
    GPT-OSS is trained strictly on Open AI Harmony input/output format.
    Refer to the documentation: https://cookbook.openai.com/articles/openai-harmony
    if you are to change/customize anything defined here.
    openai_harmony lacks documentation so I might create a parser.
    """
    model_name: str = "gpt-oss-20b"
    model_location: str = "https://huggingface.co/openai/gpt-oss-20b"
    vllm: Any = None
    encoding: hmny.HarmonyEncoding = hmny.load_harmony_encoding(hmny.HarmonyEncodingName.HARMONY_GPT_OSS)

    def __init__(self, reasoning_effort: str = "medium", **kwargs):
        super().__init__(**kwargs)
        reasoning_effort = reasoning_effort.lower()
        # TODO: is none possible?
        if reasoning_effort not in ["none", "low", "medium", "high"]:
            raise ValueError("reasoning_effort must be one of 'none', 'low', 'medium', or 'high'")
        reasoning_effort = (
            hmny.ReasoningEffort.LOW if reasoning_effort == "low"
            else hmny.ReasoningEffort.MEDIUM if reasoning_effort == "medium"
            else None if reasoning_effort == "none"
            else hmny.ReasoningEffort.HIGH
        )
        system_message = (
            hmny.SystemContent.new()
            .with_reasoning_effort(reasoning_effort)
            .with_conversation_start_date(datetime.now().strftime("%Y-%m-%d"))
        )
        self._harmony_messages = [hmny.Message.from_role_and_content(hmny.Role.SYSTEM, system_message)]
        download_repo: modal.Function = modal.Function.from_name("pitchjams", "download_repo")
        for _ in download_repo.remote_get(self.model_location):
            pass
        self.vllm = modal.Cls.from_name("pitchjams", "VLLMModel")(model_name="gpt-oss-20b")

    @property
    def _llm_type(self) -> str:
        return "gpt-oss-vllm-custom"
    
    def _build_harmony_messages(self, messages: list[BaseMessage], tools: list[hmny.ToolDescription]) -> list[hmny.Message]:        
        current_messages = self._harmony_messages.copy()
        dev_message = hmny.DeveloperContent.new()
        if messages[0].type == "system":
            dev_message = dev_message.with_instructions(messages[0].content)
            messages = messages[1:]
        if tools and len(tools) > 0:
            dev_message = dev_message.with_function_tools(tools)
        # TODO: support built-in tools that gpt-oss provides
        current_messages.append(hmny.Message.from_role_and_content(hmny.Role.DEVELOPER, dev_message))
        for message in messages:
            if message.type == "human":
                current_messages.append(hmny.Message.from_role_and_content(hmny.Role.USER, message.content))
            elif message.type == "ai":
                # TODO: also include tool calls and reasoning here?
                current_messages.append(hmny.Message.from_role_and_content(hmny.Role.ASSISTANT, message.content))
            elif message.type == "tool":
                current_messages.append(
                    hmny.Message.from_author_and_content(
                        hmny.Author.new(hmny.Role.TOOL, message.recipient), 
                        message.content
                    ).with_channel("commentary")
                )
        logger.info(f"Input Conversation:\n" + ''.join(f'{turn}\n' for turn in current_messages))
        return current_messages

    def _generate(self, messages: list[BaseMessage], stop=None, **kwargs) -> ChatResult:
        if not messages or len(messages) == 0:
            raise ValueError("Messages list is empty.")
        # prepare input prompt
        auto_call_functions = kwargs.get("auto_call_functions", False)
        available_tools = [tool_description for tool_description, _ in kwargs.get("tools", [])]
        conversation = hmny.Conversation.from_messages(self._build_harmony_messages(messages, available_tools))
        input_ids = self.encoding.render_conversation_for_completion(conversation, hmny.Role.ASSISTANT)
        # open Modal communication
        local_cipher: P2PEncryption = kwargs.get("local_cipher", None)
        output_parser = hmny.StreamableParser(self.encoding, hmny.Role.ASSISTANT)
        # encryption handshake
        if not local_cipher:
            local_cipher = P2PEncryption(is_remote=False)
            enc_request = self.vllm.encryption_request.remote()
            enc_response = local_cipher.encryption_response(*enc_request)
            if not self.vllm.encryption_acknowledged.remote(*enc_response):
                raise ValueError("Encryption handshake with Modal failed.")
        # perform inference on remote side
        # TODO: move this to _stream() then call it here
        encrypted_payload = local_cipher.cryptor.encrypt(json.dumps(input_ids).encode('utf-8'))
        remote_generator = self.vllm.inference.remote_gen(encrypted_payload)
        response = {
            "reasoning": None,
            "content": None,
            "content_type": None,
            "recipient": None,
            "stop_reason": None,
        }
        current_content = ""
        current_channel = None
        current_content_type = None
        current_recipient = None
        last_token = None
        for encrypted_response in remote_generator:
            output_ids: list[int] = json.loads(local_cipher.cryptor.decrypt(encrypted_response).decode('utf-8'))
            for token in output_ids: 
                output_parser.process(token)
                if current_channel and current_channel != output_parser.current_channel:
                    if current_channel == "analysis":
                        response["reasoning"] = current_content
                    else:
                        response["content"] = current_content
                        if current_channel == "commentary":
                            response["recipient"] = current_recipient
                            response["content_type"] = current_content_type
                    current_content = ""
                token_str = output_parser.last_content_delta
                current_content += token_str if token_str else ""
                current_channel = output_parser.current_channel
                current_content_type = output_parser.current_content_type
                current_recipient = output_parser.current_recipient
                last_token = token
                logger.info(f"current_role: {output_parser.current_role}, current_channel: {current_channel}, recipient: {current_recipient}, token_str: {token_str}, current_content_type: {output_parser.current_content_type}")
        if last_token == 200002: # stop token
            response["stop_reason"] = "completed"
        elif last_token == 200012: # function calling
            response["stop_reason"] = "function_call"
        else: # TODO: monitor any other stop token is possible
            response["stop_reason"] = "max_tokens"
        logger.info("Response Completed: end token:" + str(last_token))
        logger.info(response)
        if auto_call_functions and response["stop_reason"] == "function_call":
            tool_name = response["recipient"].split('.')[-1]
            target_tool: BaseTool = next((tool for tool_description, tool in kwargs.get("tools", []) if tool_description.name == tool_name), None)
            response = target_tool.invoke(json.loads(response["content"]))
            messages.append(AIMessage(content=response["content"], recipient=target_tool.name, content_type=response["content_type"]))
            messages.append(ToolMessage(content=f"{str(response)}", recipient=target_tool.name))
            return self._generate(messages, stop=stop, local_cipher=local_cipher, **kwargs)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=response["content"]))],
            llm_output={
                "reasoning": response["reasoning"],
                "stop_reason": response["stop_reason"],
            }
        )

    async def _agenerate(self, messages: list[BaseMessage], stop=None, **kwargs) -> ChatResult:
        self._generate(messages, stop=stop, **kwargs)
    
    def _convert_to_harmony_tool(self, tool: BaseTool) -> hmny.ToolDescription:
        return hmny.ToolDescription.new(
            tool.name,
            tool.description,
            parameters={
                "type": "object",
                "properties": {
                    attr_name: {
                        "type": attr_type.__name__
                    } for attr_name, attr_type in tool.args_schema.__annotations__.items()
                },
                "required": list(tool.args_schema.__annotations__.keys()),
            }
        )

    def bind_tools(
        self,
        tools: Sequence[Union[dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, str, Literal["auto", "any"], bool]] = None,  # noqa: PYI051
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        harmony_tools = [(self._convert_to_harmony_tool(tool), tool) for tool in tools]
        return super().bind(tools=harmony_tools, **kwargs)


