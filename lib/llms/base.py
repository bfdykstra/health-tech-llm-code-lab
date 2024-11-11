from typing import AsyncGenerator, Iterator

from langchain_core.language_models.base import BaseLanguageModel

from lib.base import LLMInterface
from lib.base_component import BaseComponent


class BaseLLM(BaseComponent):
    def to_langchain_format(self) -> BaseLanguageModel:
        raise NotImplementedError

    def invoke(self, *args, **kwargs) -> LLMInterface:
        raise NotImplementedError

    async def ainvoke(self, *args, **kwargs) -> LLMInterface:
        raise NotImplementedError

    def stream(self, *args, **kwargs) -> Iterator[LLMInterface]:
        raise NotImplementedError

    def astream(self, *args, **kwargs) -> AsyncGenerator[LLMInterface, None]:
        raise NotImplementedError

    def run(self, *args, **kwargs):
        if kwargs.get('use_async', False):
            return self.ainvoke(*args, **kwargs)
        return self.invoke(*args, **kwargs)
