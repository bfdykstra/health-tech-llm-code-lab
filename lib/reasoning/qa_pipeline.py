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

# from ktem.utils.render import Render
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

from lib.reasoning.pipelines.prepare_evidence import PrepareEvidencePipeline
from lib.reasoning.pipelines.answer_with_context import AnswerWithContextPipeline
from lib.reasoning.pipelines.document_retrieval import BaseRetriever


from lib.indices.qa.citation import CitationPipeline
from lib.indices.splitters import TokenSplitter
from lib.llms import ChatLLM, PromptTemplate


from .base import BaseReasoning

logger = logging.getLogger(__name__)

class FullQAPipeline(BaseReasoning):
    """Question answering pipeline. Handle from question to answer"""

    class Config:
        allow_extra = True

    # configuration parameters
    trigger_context: int = 150
    use_rewrite: bool = False

    retrievers: list[BaseRetriever]

    evidence_pipeline: PrepareEvidencePipeline
    answering_pipeline: AnswerWithContextPipeline

    def retrieve(
        self, message: str, document_ids: Optional[list[str]] = None,
    ) -> list[RetrievedDocument]:
        """Retrieve the documents based on the message"""

        query = None
        if not query:
            # TODO: previously return [], [] because we think this message as something
            # like "Hello", "I need help"...
            query = message

        docs, doc_ids = [], []

        for idx, retriever in enumerate(self.retrievers):
            # retriever_node = self._prepare_child(retriever, f"retriever_{idx}")
            retriever_docs = retriever(text=query, doc_ids = document_ids)

            retriever_docs_text = []
            retriever_docs_plot = []

            for doc in retriever_docs:
                if doc.metadata.get("type", "") == "plot":
                    retriever_docs_plot.append(doc)
                else:
                    retriever_docs_text.append(doc)

            for doc in retriever_docs_text:
                if doc.doc_id not in doc_ids:
                    docs.append(doc)
                    doc_ids.append(doc.doc_id)
        return [doc for doc in docs]


    async def ainvoke(  # type: ignore
        self, message: str, history: list, **kwargs  # type: ignore
    ) -> Document:  # type: ignore
        raise NotImplementedError


    def invoke(self, message: str, document_ids: Optional[list[str]] = None, **kwargs #type: ignore 
               ) -> tuple[Document, list[Document]]:
        print(f"Retrievers {self.retrievers}")
        # should populate the context
        docs= self.retrieve(message, document_ids)
        print(f"Got {len(docs)} retrieved documents")                

        evidence_mode, evidence, images = self.evidence_pipeline(docs).content

        scored_docs = self.retrievers[0].generate_relevant_scores(message, docs)

        answer = self.answering_pipeline(question = message, evidence = evidence, evidence_mode = evidence_mode, images = images)

        return answer, scored_docs
    

    def stream(  # type: ignore
        self, message: str, history: list, **kwargs  # type: ignore
    ) -> Generator[Document, None, Document]:

        print(f"Retrievers {self.retrievers}")
        # should populate the context
        docs, infos = self.retrieve(message, history)
        print(f"Got {len(docs)} retrieved documents")
        yield from infos

        evidence_mode, evidence, images = self.evidence_pipeline(docs).content

        def generate_relevant_scores():
            nonlocal docs
            docs = self.retrievers[0].generate_relevant_scores(message, docs)

        # generate relevant score using
        if evidence and self.retrievers:
            scoring_thread = threading.Thread(target=generate_relevant_scores)
            scoring_thread.start()
        else:
            scoring_thread = None

        answer = yield from self.answering_pipeline.stream(
            question=message,
            history=history,
            evidence=evidence,
            evidence_mode=evidence_mode,
            images=images,
            conv_id='',
            **kwargs,
        )

        # show the evidence
        if scoring_thread:
            scoring_thread.join()

        yield from self.show_citations(answer, docs)

        return answer

    # @classmethod
    # def get_pipeline(cls, settings, states, retrievers):
    #     """Get the reasoning pipeline

    #     Args:
    #         settings: the settings for the pipeline
    #         retrievers: the retrievers to use
    #     """
    #     max_context_length_setting = settings.get("reasoning.max_context_length", 32000)

    #     pipeline = cls(
    #         retrievers=retrievers,
    #         rewrite_pipeline=RewriteQuestionPipeline(),
    #     )

    #     prefix = f"reasoning.options.{cls.get_info()['id']}"
    #     llm_name = settings.get(f"{prefix}.llm", None)
    #     llm = llms.get(llm_name, llms.get_default())

    #     # prepare evidence pipeline configuration
    #     evidence_pipeline = pipeline.evidence_pipeline
    #     evidence_pipeline.max_context_length = max_context_length_setting

    #     # answering pipeline configuration
    #     answer_pipeline = pipeline.answering_pipeline
    #     answer_pipeline.llm = llm
    #     answer_pipeline.citation_pipeline.llm = llm
    #     answer_pipeline.n_last_interactions = settings[f"{prefix}.n_last_interactions"]
    #     answer_pipeline.enable_citation = settings[f"{prefix}.highlight_citation"]
    #     answer_pipeline.system_prompt = settings[f"{prefix}.system_prompt"]
    #     answer_pipeline.qa_template = settings[f"{prefix}.qa_prompt"]
    #     answer_pipeline.lang = "English"

    #     pipeline.add_query_context.llm = llm
    #     pipeline.add_query_context.n_last_interactions = settings[
    #         f"{prefix}.n_last_interactions"
    #     ]

    #     pipeline.trigger_context = settings[f"{prefix}.trigger_context"]
    #     pipeline.use_rewrite = states.get("app", {}).get("regen", False)
    #     if pipeline.rewrite_pipeline:
    #         pipeline.rewrite_pipeline.llm = llm
    #         pipeline.rewrite_pipeline.lang = SUPPORTED_LANGUAGE_MAP.get(
    #             settings["reasoning.lang"], "English"
    #         )
    #     return pipeline


    # @classmethod
    # def get_info(cls) -> dict:
    #     return {
    #         "id": "simple",
    #         "name": "Simple QA",
    #         "description": (
    #             "Simple RAG-based question answering pipeline. This pipeline can "
    #             "perform both keyword search and similarity search to retrieve the "
    #             "context. After that it includes that context to generate the answer."
    #         ),
    #     }