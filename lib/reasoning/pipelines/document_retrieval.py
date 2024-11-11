from theflow import Node, Param
from typing import Generator, Optional, Sequence
from enum import Enum
import logging
import sqlite3
import time
from collections import defaultdict
from abc import abstractmethod

from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from lib.base import Document, RetrievedDocument
from lib.base_component import BaseComponent
from lib.storages.docstores import BaseDocumentStore
from lib.storages.vectorstores import BaseVectorStore
from lib.embeddings import BaseEmbeddings
from lib.indices import VectorIndexing, VectorRetrieval

from lib.indices.rankings import (
    BaseReranking,
    CohereReranking,
    LLMReranking,
    LLMTrulensScoring,
)
from lib.indices.splitters import BaseSplitter, TokenSplitter

logger = logging.getLogger(__name__)


class VectorStoreQueryMode(str, Enum):
    """Vector store query mode."""

    DEFAULT = "default"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    TEXT_SEARCH = "text_search"
    SEMANTIC_HYBRID = "semantic_hybrid"

    # fit learners
    SVM = "svm"
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_REGRESSION = "linear_regression"

    # maximum marginal relevance
    MMR = "mmr"


class BaseRetriever(BaseComponent):

    embedding:BaseEmbeddings =  Param(help="the embedding class to embed queries")
    rerankers: Sequence[BaseReranking] = Param(default=[], help="List of rerankers")
    # use LLM to create relevant scores for displaying on UI
    llm_scorer: LLMReranking | None = Param(default = None)
    get_extra_table: bool = Param(default = False)
    mmr: bool = Param(default = False)
    top_k: int = Param(default = 5, help = "The top number of docs to retrieve")
    retrieval_mode: str = Param(default = "hybrid", help="hybrid, vector, text" )
    vector_store: BaseVectorStore = Param(help='The vector store')
    doc_store: BaseDocumentStore = Param(help = 'the document store')
    sql_lite_db = Param(default = "/Users/benjamindykstra/development/kotaemon/ktem_app_data/user_data/sql.db", help = "wherever the app data lives that connects document ids to chunk ids")
    
    @abstractmethod
    def fetch_chunk_ids(self, doc_ids: Sequence[str], index:str) -> Sequence[str]:
        '''
        Givena list of document ids from database, fetch their chunks
        '''
        ...

    @abstractmethod
    def generate_relevant_scores(
        self, query: str, documents: list[RetrievedDocument]
    ) -> list[RetrievedDocument]:
        ...

    @abstractmethod
    def run(
        self,
        text: str,
        doc_ids: Optional[list[str]] = None,
        *args,
        **kwargs,
    ) -> list[RetrievedDocument]:
        """Retrieve document excerpts similar to the text

        Args:
            text: the text to retrieve similar documents
            doc_ids: list of document ids to constraint the retrieval
        """
        ...


class DocumentRetrievalPipeline(BaseRetriever):
    """Retrieve relevant document

    Args:
        vector_retrieval: the retrieval pipeline that return the relevant documents
            given a text query
        reranker: the reranking pipeline that re-rank and filter the retrieved
            documents
        get_extra_table: if True, for each retrieved document, the pipeline will look
            for surrounding tables (e.g. within the page)
        top_k: number of documents to retrieve
        mmr: whether to use mmr to re-rank the documents
    """

    embedding:BaseEmbeddings =  Param(help="the embedding class to embed queries")
    rerankers: Sequence[BaseReranking] = Param(default=[], help="List of rerankers")
    # use LLM to create relevant scores for displaying on UI
    llm_scorer: LLMReranking | None = Param(default = None)
    get_extra_table: bool = Param(default = False)
    mmr: bool = Param(default = False)
    top_k: int = Param(default = 5, help = "The top number of docs to retrieve")
    retrieval_mode: str = Param(default = "hybrid", help="hybrid, vector, text" )
    vector_store: BaseVectorStore = Param(help='The vector store')
    doc_store: BaseDocumentStore = Param(help = 'the document store')
    sql_lite_db = Param(default = "/Users/benjamindykstra/development/kotaemon/ktem_app_data/user_data/sql.db", help = "wherever the app data lives that connects document ids to chunk ids")
    
    conn: sqlite3.Connection = Param(default = None)
    vector_retrieval: VectorRetrieval = Param(default = None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Initialize the parent class if necessary
        # Initialize the SQLite connection using the sql_lite_db path
        self.conn = sqlite3.connect(self.sql_lite_db)
        self.vector_retrieval = VectorRetrieval(
            embedding=self.embedding,
            vector_store=self.vector_store,
            doc_store=self.doc_store,
            retrieval_mode=self.retrieval_mode,  # type: ignore
            top_k= self.top_k,
            rerankers=self.rerankers,
        )

    # @Node.auto(depends_on=["embedding", "vector_store", "doc_store"])
    # def vector_retrieval(self) -> VectorRetrieval:
    #     return VectorRetrieval(
    #         embedding=self.embedding,
    #         vector_store=self.vector_store,
    #         doc_store=self.doc_store,
    #         retrieval_mode=self.retrieval_mode,  # type: ignore
    #         top_k= self.top_k,
    #         rerankers=self.rerankers,
    #     )

    def fetch_chunk_ids(self, doc_ids: Sequence[str], index = 'index__1__index'):
        '''
        Table index__1__index maps document ids to chunk ids
        '''
        # conn = sqlite3.connect(self.sql_lite_db)
        # try:
        stmt = f'''
        SELECT target_id from {index} where relation_type = "document" and source_id IN ({','.join(['?']*len(doc_ids))})
        '''
        try:
            c = self.conn.cursor()

            c.execute(stmt, doc_ids)

            all_results = c.fetchall()
            return [res[0] for res in all_results]
        finally:
            c.close()

    def run(
        self,
        text: str,
        doc_ids: list[str] = [],
        *args,
        **kwargs,
    ) -> list[RetrievedDocument]:
        """Retrieve document excerpts similar to the text

        Args:
            text: the text to retrieve similar documents
            doc_ids: list of document ids to constraint the retrieval
        """
        print("searching in doc_ids", doc_ids)
        if not doc_ids:
            logger.info(f"Skip retrieval because of no selected files: {self}")
            return []

        retrieval_kwargs: dict = {}
        
        chunk_ids = self.fetch_chunk_ids(doc_ids)

        # do first round top_k extension
        retrieval_kwargs["do_extend"] = True
        retrieval_kwargs["scope"] = chunk_ids
        retrieval_kwargs["filters"] = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="file_id",
                    value=doc_ids,
                    operator=FilterOperator.IN,
                )
            ],
            condition=FilterCondition.OR,
        )

        if self.mmr:
            # TODO: double check that llama-index MMR works correctly
            retrieval_kwargs["mode"] = VectorStoreQueryMode.MMR
            retrieval_kwargs["mmr_threshold"] = 0.5

        # rerank
        s_time = time.time()
        print(f"retrieval_kwargs: {retrieval_kwargs.keys()}")
        docs = self.vector_retrieval(text=text, top_k=self.top_k, **retrieval_kwargs)
        print("retrieval step took", time.time() - s_time)

        return docs
        

    def generate_relevant_scores(
        self, query: str, documents: list[RetrievedDocument]
    ) -> list[RetrievedDocument]:
        docs = (
            documents
            if not self.llm_scorer
            else self.llm_scorer(documents=documents, query=query)
        )
        return docs

    @classmethod
    def get_pipeline(cls, user_settings, index_settings, selected):
        raise NotImplementedError