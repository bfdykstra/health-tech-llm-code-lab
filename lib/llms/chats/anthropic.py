from typing import TYPE_CHECKING, AsyncGenerator, Iterator, Optional, Type, Union

from anthropic import Anthropic, AsyncAnthropic
import instructor
from instructor import AsyncInstructor, Instructor, Mode, patch

from lib.base import AIMessage, BaseMessage, HumanMessage, LLMInterface, StructuredOutputLLMInterface
from theflow import Param

from pydantic import BaseModel

from .base import ChatLLM

from anthropic.types.message import Message as AnthropicMessage

if TYPE_CHECKING:
    
    # from anthropic.types.
    # from anthropic.types.completion import Completion as AnthropicChatCompletionMessageParam
    from openai.types.chat.chat_completion_message_param import (
        ChatCompletionMessageParam,
    )


class BaseChatAnthropic(ChatLLM):
    api_key: str = Param(help="API key", required=True)
    timeout: Optional[float] = Param(None, help="Timeout for the API request")
    max_retries: Optional[int] = Param(
        2, help="Maximum number of retries for the API request"
    )

    temperature: Optional[float] = Param(
        None,
        help=(
            "Number between 0 and 2 that controls the randomness of the generated "
            "tokens. Lower values make the model more deterministic, while higher "
            "values make the model more random."
        ),
    )
    max_tokens: Optional[int] = Param(
        2048,
        help=(
            "Maximum number of tokens to generate. The total length of input tokens "
            "and generated tokens is limited by the model's context length."
        ),
    )
    n: int = Param(
        1,
        help=(
            "Number of completions to generate. The API will generate n completion "
            "for each prompt."
        ),
    )
    stop: Optional[str | list[str]] = Param(
        None,
        help=(
            "Stop sequence. If a stop sequence is detected, generation will stop "
            "at that point. If not specified, generation will continue until the "
            "maximum token length is reached."
        ),
    )

    @Param.auto(depends_on=["max_retries"])
    def max_retries_(self):
        if self.max_retries is None:
            from anthropic._constants import DEFAULT_MAX_RETRIES

            return DEFAULT_MAX_RETRIES
        return self.max_retries
    
    def prepare_message(
        self, messages: str | BaseMessage | list[BaseMessage]
    ) -> list["ChatCompletionMessageParam"]:
        """Prepare the message into OpenAI format

        Returns:
            list[dict]: List of messages in OpenAI format
        """
        input_: list[BaseMessage] = []
        output_: list["ChatCompletionMessageParam"] = []

        if isinstance(messages, str):
            input_ = [HumanMessage(content=messages)]
        elif isinstance(messages, BaseMessage):
            input_ = [messages]
        else:
            input_ = messages

        for message in input_:
            output_.append(message.to_openai_format())
            # output_.append(message)

        return output_
    

    def prepare_output(self, resp: AnthropicMessage) -> LLMInterface:
        '''Convert anthropic response to LLMInterface
        https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/message.py
        '''

        additional_kwargs = {}
        usage = resp.usage
        content = resp.content[0].text if resp.content[0].type == 'text' else resp.content
        id = resp.id
        model = resp.model
        
        output = LLMInterface(
            content=content,
            total_tokens=usage.input_tokens + usage.output_tokens,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            messages=[
                # AIMessage(content=ob.text, type=ob.type) 
                AIMessage(content = ob.text)
                for ob in resp.content if ob.type != 'tool_use'],
            
        )

        return output
    

    def prepare_client(self, async_version: bool = False):
        '''Get the anthropic client
        Args:
            async_version (bool): Whether to get the async version of the client
        '''
        raise NotImplementedError
    

    def anthropic_response(self, client: Anthropic, **kwargs):
        raise NotImplementedError
    
    async def async_anthropic_response(self, client: AsyncAnthropic, **kwargs):
        raise NotImplementedError
    

    def invoke(
        self, messages: str | BaseMessage | list[BaseMessage], *args, **kwargs
    ) -> LLMInterface:
        
        client = self.prepare_client(async_version=False)
        input_messages = self.prepare_message(messages)

        resp = self.anthropic_response(
            client, messages = input_messages, stream = False, **kwargs
        )

        return self.prepare_output(resp)

    async def ainvoke(
        self, messages: str | BaseMessage | list[BaseMessage], *args, **kwargs
    ) -> LLMInterface:
        client = self.prepare_client(async_version=True)
        input_messages = self.prepare_message(messages)
        resp = await self.async_anthropic_response(
            client, messages=input_messages, stream=False, **kwargs
        ) #type: ignore

        return self.prepare_output(resp)
    

    def stream(
        self, messages: str | BaseMessage | list[BaseMessage], *args, **kwargs
    ) -> Iterator[LLMInterface]:
        client = self.prepare_client(async_version=False)
        input_messages = self.prepare_message(messages)
        resp = self.anthropic_response(
            client, messages=input_messages, stream=True, **kwargs
        )

        for chunk in resp:
            prepped_chunk = self.prepare_output(chunk)
            yield prepped_chunk

    async def astream(
        self, messages: str | BaseMessage | list[BaseMessage], *args, **kwargs
    ) -> AsyncGenerator[LLMInterface, None]:
        
        client = self.prepare_client(async_version=False)
        input_messages = self.prepare_message(messages)
        resp = self.anthropic_response(
            client, messages=input_messages, stream=True, **kwargs
        )

        async for chunk in resp:
            if not chunk.content:
                continue
            if chunk.content is not None:
                prepped_chunk = self.prepare_output(chunk)
                yield prepped_chunk



class ChatAnthropic(BaseChatAnthropic):

    # base_url: Optional[str] = Param(None, help="anthropic base URL")
    # organization: Optional[str] = Param(None, help="OpenAI organization")
    model: str = Param(help="Anthropic model model", required=True)

    def prepare_client(self, async_version: bool = False):
        params = {
            'api_key': self.api_key,
            'timeout': self.timeout,
            'max_retries': self.max_retries
        }

        if async_version:
            return AsyncAnthropic(**params)
        
        return Anthropic(**params)


    def anthropic_response(self, client: Anthropic, **kwargs):

        """get anthropic response"""

        params_ = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # "stop_sequences": [self.stop]
        }

        params = {k: v for k, v in params_.items() if v is not None}
        params.update(kwargs)


        return client.messages.create(**params)
    

    async def async_anthropic_response(self, client: AsyncAnthropic, **kwargs):

        params_ = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # "stop_sequences": [self.stop]
        }

        params = {k: v for k, v in params_.items() if v is not None}
        params.update(kwargs)

        res = await client.messages.create(**params)

        return res
    


class StructuredChatAnthropic(ChatAnthropic):
    """Anthropic doesn't provide an actual structured output mode, so we need to parse a given pydantic model to something
       the model can understand/
    """
    

    response_schema: Type[BaseModel] = Param(help="class that subclasses pydantics BaseModel", required = True)

    def prepare_output(self, resp: tuple[BaseModel, AnthropicMessage]) -> LLMInterface:
        '''Convert anthropic response to LLMInterface
        https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/message.py
        '''

        schema_res, anthropic_response = resp
        additional_kwargs = {}
        usage = anthropic_response.usage
        content = schema_res.model_dump_json()
        id = anthropic_response.id
        model = anthropic_response.model
        
        output = LLMInterface(
            content=content,
            total_tokens=usage.input_tokens + usage.output_tokens,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            messages=[
                AIMessage(content = schema_res.model_dump_json())
            ],
            
        )

        return output

    def prepare_client(self, async_version: bool = False):
        params = {
            'api_key': self.api_key,
            'timeout': self.timeout,
            'max_retries': self.max_retries
        }

        client = AsyncAnthropic(**params) if async_version else Anthropic(**params)
        return instructor.from_anthropic(client)
    

    def anthropic_response(self, client: Instructor, **kwargs) -> tuple[BaseModel, AnthropicMessage]:

        """get anthropic response"""

        params_ = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # "stop_sequences": [self.stop]
            "response_model": self.response_schema,
        }

        params = {k: v for k, v in params_.items() if v is not None}
        params.update(kwargs)

        return client.messages.create_with_completion(**params)

    async def async_anthropic_response(self, client: AsyncInstructor, **kwargs) -> tuple[BaseModel, AnthropicMessage]:
        params_ = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # "stop_sequences": [self.stop]
            "response_model": self.response_schema,
        }

        params = {k: v for k, v in params_.items() if v is not None}
        params.update(kwargs)

        # structured response is of type self.response_schema, anthropic_response is AnthropicMessage
        structured_response, anthropic_response = await client.messages.create_with_completion(**params)

        return structured_response, anthropic_response
    
