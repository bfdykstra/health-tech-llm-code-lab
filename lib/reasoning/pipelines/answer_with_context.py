import traceback
import html
import logging
import threading
from collections import defaultdict
from difflib import SequenceMatcher
from functools import partial
from typing import Generator, Optional

import numpy as np
import tiktoken

from theflow.settings import settings as flowsettings
from theflow import Node
from lib.base_component import BaseComponent

from lib.base import (
    AIMessage,
    Document,
    HumanMessage,
    RetrievedDocument,
    SystemMessage,
)


from lib.indices.qa.citation import CitationPipeline
from lib.llms import ChatLLM, PromptTemplate

from ..default_prompts import DEFAULT_QA_TEXT_PROMPT, EVIDENCE_MODE_TEXT


from lib.reasoning.base import BaseReasoning

logger = logging.getLogger(__name__)


class AnswerWithContextPipeline(BaseComponent):
    """Answer the question based on the evidence

    Args:
        llm: the language model to generate the answer
        citation_pipeline: generates citation from the evidence
        qa_template: the prompt template for LLM to generate answer (refer to
            evidence_mode)
        qa_table_template: the prompt template for LLM to generate answer for table
            (refer to evidence_mode)
        qa_chatbot_template: the prompt template for LLM to generate answer for
            pre-made scenarios (refer to evidence_mode)
        lang: the language of the answer. Currently support English and Japanese
    """

    llm: ChatLLM 
    vlm_endpoint: str = getattr(flowsettings, "KH_VLM_ENDPOINT", "") # vision language model
    use_multimodal: bool = getattr(flowsettings, "KH_REASONINGS_USE_MULTIMODAL", True)
    # citation_pipeline: Optional[CitationPipeline]

    qa_template: str = DEFAULT_QA_TEXT_PROMPT

    enable_citation: bool = False
    system_prompt: str = ""
    lang: str = "English"  # support English
    n_last_interactions: int = 5

    def get_prompt(self, question, evidence, evidence_mode: int):
        """Prepare the prompt and other information for LLM"""
        if evidence_mode == EVIDENCE_MODE_TEXT:
            prompt_template = PromptTemplate(self.qa_template)
        

        prompt = prompt_template.populate(
            context=evidence,
            question=question,
        )

        return prompt, evidence

    def run(
        self, question: str, evidence: str, evidence_mode: int = 0, **kwargs
    ) -> Document:
        return self.invoke(question, evidence, evidence_mode, **kwargs)

    def invoke(
        self,
        question: str,
        evidence: str,
        evidence_mode: int = 0,
        images: list[str] = [],
        **kwargs,
    ) -> Document:
        
        # make these tuples of (human, ai)
        history = kwargs.get("history", [])
        print(f"Got {len(images)} images")
        # check if evidence exists, use QA prompt
        if evidence:
            prompt, evidence = self.get_prompt(question, evidence, evidence_mode)
        else:
            prompt = question

        messages = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))

        for human, ai in history:
            messages.append(HumanMessage(content=human))
            messages.append(AIMessage(content=ai))
    
        messages.append(HumanMessage(content=prompt))

        try:
            output = self.llm(messages)

            return Document(text = output.text, metadata = {"qa_score": np.exp(np.average(output.logprobs))})
        except Exception as e:
            print("hit error invoking in answer with context: ", e)
            raise e


    async def ainvoke(  # type: ignore
        self,
        question: str,
        evidence: str,
        evidence_mode: int = 0,
        images: list[str] = [],
        **kwargs,
    ) -> Document:
        """Answer the question based on the evidence

        In addition to the question and the evidence, this method also take into
        account evidence_mode. The evidence_mode tells which kind of evidence is.
        The kind of evidence affects:
            1. How the evidence is represented.
            2. The prompt to generate the answer.

        By default, the evidence_mode is 0, which means the evidence is plain text with
        no particular semantic representation. The evidence_mode can be:
            1. "table": There will be HTML markup telling that there is a table
                within the evidence.
            2. "chatbot": There will be HTML markup telling that there is a chatbot.
                This chatbot is a scenario, extracted from an Excel file, where each
                row corresponds to an interaction.

        Args:
            question: the original question posed by user
            evidence: the text that contain relevant information to answer the question
                (determined by retrieval pipeline)
            evidence_mode: the mode of evidence, 0 for text, 1 for table, 2 for chatbot
        """
       # make these tuples of (human, ai)
        history = kwargs.get("history", [])
        print(f"Got {len(images)} images")
        # check if evidence exists, use QA prompt
        if evidence:
            prompt, evidence = self.get_prompt(question, evidence, evidence_mode)
        else:
            prompt = question

        messages = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))

        for human, ai in history:
            messages.append(HumanMessage(content=human))
            messages.append(AIMessage(content=ai))
    
        messages.append(HumanMessage(content=prompt))

        try:
            output = await self.llm.ainvoke(messages)

            return Document(text = output.text, metadata = {"qa_score": np.exp(np.average(output.logprobs))})
        except Exception as e:
            print("hit error async invoking in answer with context: ", e)
            raise e

    def stream(  
        self,
        question: str,
        evidence: str,
        evidence_mode: int = 0,
        images: list[str] = [],
        **kwargs,
    ) -> Generator[Document, None, Document]:
        history = kwargs.get("history", [])
        print(f"Got {len(images)} images")
        # check if evidence exists, use QA prompt
        if evidence:
            prompt, evidence = self.get_prompt(question, evidence, evidence_mode)
        else:
            prompt = question

        # retrieve the citation
        citation = None

        # def citation_call():
        #     nonlocal citation
        #     citation = self.citation_pipeline(context=evidence, question=question)

        # if evidence and self.enable_citation:
        #     # execute function call in thread
        #     citation_thread = threading.Thread(target=citation_call)
        #     citation_thread.start()
        # else:
        #     citation_thread = None

        output = ""
        logprobs = []

        messages = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        for human, ai in history[-self.n_last_interactions :]:
            messages.append(HumanMessage(content=human))
            messages.append(AIMessage(content=ai))


        messages.append(HumanMessage(content=prompt))

        try:
            # try streaming first
            print("Trying LLM streaming")
            for out_msg in self.llm.stream(messages):
                output += out_msg.text
                logprobs += out_msg.logprobs
                yield Document(channel="chat", content=out_msg.text)
        except NotImplementedError:
            print("Streaming is not supported, falling back to normal processing")
            output = self.llm(messages).text
            yield Document(channel="chat", content=output)

        if logprobs:
            qa_score = np.exp(np.average(logprobs))
        else:
            qa_score = None

        # if citation_thread:
        #     citation_thread.join()
        answer = Document(
            text=output,
            metadata={"citation": citation, "qa_score": qa_score},
        )

        return answer
