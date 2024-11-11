from __future__ import annotations

import logging
import shutil
import threading
import time
import warnings
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Generator, Optional, Sequence

import tiktoken
from ktem.db.models import engine
from ktem.embeddings.manager import embedding_models_manager
from ktem.llms.manager import llms
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.file.base import default_file_metadata_func

from llama_index.core.vector_stores.types import VectorStoreQueryMode
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from theflow.settings import settings
from theflow.utils.modules import import_dotted_string
from theflow import Node, Param

import sqlite3


from lib.base import Document, RetrievedDocument
from lib.base_component import BaseComponent
from lib.embeddings import BaseEmbeddings
from lib.indices import VectorIndexing, VectorRetrieval
from lib.indices.ingests.files import (
    KH_DEFAULT_FILE_EXTRACTORS,
    adobe_reader,
    azure_reader,
    unstructured,
)
from lib.indices.rankings import (
    BaseReranking,
    CohereReranking,
    LLMReranking,
    LLMTrulensScoring,
)
from lib.indices.splitters import BaseSplitter, TokenSplitter

from .base import BaseFileIndexIndexing, BaseFileIndexRetriever

logger = logging.getLogger(__name__)


class IndexPipeline(BaseComponent):
    """Index a single file"""

    loader: BaseReader
    splitter: BaseSplitter | None
    chunk_batch_size: int = 200

    Source = Param(help="The SQLAlchemy Source table")
    Index = Param(help="The SQLAlchemy Index table")
    VS = Param(help="The VectorStore")
    DS = Param(help="The DocStore")
    FSPath = Param(help="The file storage path")
    user_id = Param(help="The user id")
    collection_name: str = "default"
    private: bool = False
    run_embedding_in_thread: bool = False
    embedding: BaseEmbeddings

    sql_lite_db = Param(default = "/Users/benjamindykstra/development/kotaemon/ktem_app_data/user_data/sql.db", help = "wherever the app data lives that connects document ids to chunk ids")
    conn: sqlite3.Connection = Param(default = None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Initialize the parent class if necessary

        defaults = {
            'splitter': None,
            'chunk_batch_size': 200,
            'collection_name': 'default',
            'private': False,
            'run_embedding_in_thread': False,
            'sql_lite_db': '/Users/benjamindykstra/development/kotaemon/ktem_app_data/user_data/sql.db',
        }

        # Set attributes using `setattr()`
        for key, val in kwargs.items():
            setattr(self, key, kwargs.get(key, defaults.get(key, None)))

        # Initialize the SQLite connection using the sql_lite_db path
        self.conn = sqlite3.connect(self.sql_lite_db)


    @Node.auto(depends_on=["Source", "Index", "embedding"])
    def vector_indexing(self) -> VectorIndexing:
        return VectorIndexing(
            vector_store=self.VS, doc_store=self.DS, embedding=self.embedding
        )

    def handle_docs(self, docs, file_id, file_name) -> Generator[Document, None, int]:
        s_time = time.time()
        text_docs = []
        non_text_docs = []
        thumbnail_docs = []

        for doc in docs:
            doc_type = doc.metadata.get("type", "text")
            if doc_type == "text":
                text_docs.append(doc)
            elif doc_type == "thumbnail":
                thumbnail_docs.append(doc)
            else:
                non_text_docs.append(doc)

        print(f"Got {len(thumbnail_docs)} page thumbnails")
        page_label_to_thumbnail = {
            doc.metadata["page_label"]: doc.doc_id for doc in thumbnail_docs
        }

        if self.splitter:
            all_chunks = self.splitter(text_docs)
        else:
            all_chunks = text_docs

        # add the thumbnails doc_id to the chunks
        for chunk in all_chunks:
            page_label = chunk.metadata.get("page_label", None)
            if page_label and page_label in page_label_to_thumbnail:
                chunk.metadata["thumbnail_doc_id"] = page_label_to_thumbnail[page_label]

        to_index_chunks = all_chunks + non_text_docs + thumbnail_docs

        # add to doc store
        chunks = []
        n_chunks = 0
        chunk_size = self.chunk_batch_size * 4
        for start_idx in range(0, len(to_index_chunks), chunk_size):
            chunks = to_index_chunks[start_idx : start_idx + chunk_size]
            self.handle_chunks_docstore(chunks, file_id)
            n_chunks += len(chunks)
            yield Document(
                f" => [{file_name}] Processed {n_chunks} chunks",
                channel="debug",
            )

        def insert_chunks_to_vectorstore():
            chunks = []
            n_chunks = 0
            chunk_size = self.chunk_batch_size
            for start_idx in range(0, len(to_index_chunks), chunk_size):
                chunks = to_index_chunks[start_idx : start_idx + chunk_size]
                self.handle_chunks_vectorstore(chunks, file_id)
                n_chunks += len(chunks)
                if self.VS:
                    yield Document(
                        f" => [{file_name}] Created embedding for {n_chunks} chunks",
                        channel="debug",
                    )

        # run vector indexing in thread if specified
        if self.run_embedding_in_thread:
            print("Running embedding in thread")
            threading.Thread(
                target=lambda: list(insert_chunks_to_vectorstore())
            ).start()
        else:
            yield from insert_chunks_to_vectorstore()

        print("indexing step took", time.time() - s_time)
        return n_chunks

    def handle_chunks_docstore(self, chunks, file_id):
        """Run chunks"""
        # run embedding, add to both vector store and doc store
        self.vector_indexing.add_to_docstore(chunks)

        stmt = f'''
            INSERT into index__1__index (source_id, target_id, relation_type) 
            VALUES ({','.join(['?']*len(chunks))})
        '''

        data_to_insert = [(file_id, chunk.doc_id, "document") for chunk in chunks]
        try:
            c = self.conn.cursor()
            c.executemany(stmt, data_to_insert)
        except Exception as e:
            self.conn.rollback()
        finally:
            self.conn.commit()
            c.close()


    def handle_chunks_vectorstore(self, chunks, file_id):
        """Run chunks"""
        # run embedding, add to both vector store and doc store
        self.vector_indexing.add_to_vectorstore(chunks)
        self.vector_indexing.write_chunk_to_file(chunks)

        stmt = f'''
            INSERT into index__1__index (source_id, target_id, relation_type) 
            VALUES ({','.join(['?']*len(chunks))})
        '''

        data_to_insert = [(file_id, chunk.doc_id, "vector") for chunk in chunks]
        try:
            c = self.conn.cursor()
            c.executemany(stmt, data_to_insert)
        except Exception as e:
            self.conn.rollback()
        finally:
            self.conn.commit()
            c.close()



    def get_id_if_exists(self, file_path: Path) -> Optional[str]:
        """Check if the file is already indexed

        Args:
            file_path: the path to the file

        Returns:
            the file id if the file is indexed, otherwise None
        """

        if self.private:
            cond = "name = ? and user = ?"
        else:
            cond = "name = ?"

        stmt = f" SELECT * from index__1__source where {cond} LIMIT = 1"

        try:
            vals = [file_path.name, self.user_id] if self.private else [file_path.name]
            c = self.conn.cursor()
            c.execute(stmt, *vals)

            results = c.fetchall()

            if results:
                return results[0][0]
        finally:
            c.close()

        return None

    def store_file(self, file_path: Path) -> str:
        """Store file into the database and storage, return the file id

        Args:
            file_path: the path to the file

        Returns:
            the file id
        """
        with file_path.open("rb") as fi:
            file_hash = sha256(fi.read()).hexdigest()

        shutil.copy(file_path, self.FSPath / file_hash)

        stmt = ''''
            INSERT into index__1__source (name, path, size, user) 
            VALUES (?)
        '''

        data_to_insert = [file_path.name, file_hash, file_path.stat().st_size, self.user_id]

        try:
            # insert in to db
            c = self.conn.cursor()
            c.executemany(stmt, data_to_insert)

        except Exception as e:
            self.conn.rollback()
        finally:
            self.conn.commit()


        try:
            # get it back so i can get the id
            stmt = 'Select id from index__1__source where path = ?'

            c.execute(stmt, file_hash)

            found = c.fetchall()
            # if found:
            #     return found[0][0]
        finally:
            c.close()
        
        # return file_id
        if found:
            return found[0][0]
        else:
            raise Exception('Could not find file id')

    def finish(self, file_id: str, file_path: Path) -> str:
        """Finish the indexing"""
        with Session(engine) as session:
            stmt = select(self.Source).where(self.Source.id == file_id)
            result = session.execute(stmt).first()
            if not result:
                return file_id

            item = result[0]

            # populate the number of tokens
            doc_ids_stmt = select(self.Index.target_id).where(
                self.Index.source_id == file_id,
                self.Index.relation_type == "document",
            )
            doc_ids = [_[0] for _ in session.execute(doc_ids_stmt)]
            token_func = self.get_token_func()
            if doc_ids and token_func:
                docs = self.DS.get(doc_ids)
                item.note["tokens"] = sum([len(token_func(doc.text)) for doc in docs])

            # populate the note
            item.note["loader"] = self.get_from_path("loader").__class__.__name__

            session.add(item)
            session.commit()

        return file_id

    def get_token_func(self):
        """Get the token function for calculating the number of tokens"""
        return _default_token_func

    def delete_file(self, file_id: str):
        """Delete a file from the db, including its chunks in docstore and vectorstore

        Args:
            file_id: the file id
        """
        with Session(engine) as session:
            session.execute(delete(self.Source).where(self.Source.id == file_id))
            vs_ids, ds_ids = [], []
            index = session.execute(
                select(self.Index).where(self.Index.source_id == file_id)
            ).all()
            for each in index:
                if each[0].relation_type == "vector":
                    vs_ids.append(each[0].target_id)
                elif each[0].relation_type == "document":
                    ds_ids.append(each[0].target_id)
                session.delete(each[0])
            session.commit()

        if vs_ids and self.VS:
            self.VS.delete(vs_ids)
        if ds_ids:
            self.DS.delete(ds_ids)

    def run(
        self, file_path: str | Path, reindex: bool, **kwargs
    ) -> tuple[str, list[Document]]:
        raise NotImplementedError

    def stream(
        self, file_path: str | Path, reindex: bool, **kwargs
    ) -> Generator[Document, None, tuple[str, list[Document]]]:
        # check for duplication
        file_path = Path(file_path).resolve()
        file_id = self.get_id_if_exists(file_path)
        if file_id is not None:
            if not reindex:
                raise ValueError(
                    f"File {file_path.name} already indexed. Please rerun with "
                    "reindex=True to force reindexing."
                )
            else:
                # remove the existing records
                yield Document(f" => Removing old {file_path.name}", channel="debug")
                self.delete_file(file_id)
                file_id = self.store_file(file_path)
        else:
            # add record to db
            file_id = self.store_file(file_path)

        # extract the file
        extra_info = default_file_metadata_func(str(file_path))
        extra_info["file_id"] = file_id
        extra_info["collection_name"] = self.collection_name

        yield Document(f" => Converting {file_path.name} to text", channel="debug")
        docs = self.loader.load_data(file_path, extra_info=extra_info)
        yield Document(f" => Converted {file_path.name} to text", channel="debug")
        yield from self.handle_docs(docs, file_id, file_path.name)

        self.finish(file_id, file_path)

        yield Document(f" => Finished indexing {file_path.name}", channel="debug")
        return file_id, docs


class IndexDocumentPipeline(BaseFileIndexIndexing):
    """Index the file. Decide which pipeline based on the file type.

    This method is essentially a factory to decide which indexing pipeline to use.

    We can decide the pipeline programmatically, and/or automatically based on an LLM.
    If we based on the LLM, essentially we will log the LLM thought process in a file,
    and then during the indexing, we will read that file to decide which pipeline
    to use, and then log the operation in that file. Overtime, the LLM can learn to
    decide which pipeline should be used.
    """

    reader_mode: str = Param("default", help="The reader mode")
    embedding: BaseEmbeddings
    run_embedding_in_thread: bool = False

    @Param.auto(depends_on="reader_mode")
    def readers(self):
        readers = deepcopy(KH_DEFAULT_FILE_EXTRACTORS)
        print("reader_mode", self.reader_mode)
        if self.reader_mode == "adobe":
            readers[".pdf"] = adobe_reader
        elif self.reader_mode == "azure-di":
            readers[".pdf"] = azure_reader

        dev_readers, _, _ = dev_settings()
        readers.update(dev_readers)

        return readers

    @classmethod
    def get_user_settings(cls):
        return {
            "reader_mode": {
                "name": "File loader",
                "value": "default",
                "choices": [
                    ("Default (open-source)", "default"),
                    ("Adobe API (figure+table extraction)", "adobe"),
                    (
                        "Azure AI Document Intelligence (figure+table extraction)",
                        "azure-di",
                    ),
                ],
                "component": "dropdown",
            },
        }

    @classmethod
    def get_pipeline(cls, user_settings, index_settings) -> BaseFileIndexIndexing:
        use_quick_index_mode = user_settings.get("quick_index_mode", False)
        print("use_quick_index_mode", use_quick_index_mode)
        obj = cls(
            embedding=embedding_models_manager[
                index_settings.get(
                    "embedding", embedding_models_manager.get_default_name()
                )
            ],
            run_embedding_in_thread=use_quick_index_mode,
            reader_mode=user_settings.get("reader_mode", "default"),
        )
        return obj

    def route(self, file_path: Path) -> IndexPipeline:
        """Decide the pipeline based on the file type

        Can subclass this method for a more elaborate pipeline routing strategy.
        """
        _, chunk_size, chunk_overlap = dev_settings()

        ext = file_path.suffix.lower()
        reader = self.readers.get(ext, unstructured)
        if reader is None:
            raise NotImplementedError(
                f"No supported pipeline to index {file_path.name}. Please specify "
                "the suitable pipeline for this file type in the settings."
            )

        print("Using reader", reader)
        pipeline: IndexPipeline = IndexPipeline(
            loader=reader,
            splitter=TokenSplitter(
                chunk_size=chunk_size or 1024,
                chunk_overlap=chunk_overlap or 256,
                separator="\n\n",
                backup_separators=["\n", ".", "\u200B"],
            ),
            run_embedding_in_thread=self.run_embedding_in_thread,
            Source=self.Source,
            Index=self.Index,
            VS=self.VS,
            DS=self.DS,
            FSPath=self.FSPath,
            user_id=self.user_id,
            private=self.private,
            embedding=self.embedding,
        )

        return pipeline

    def run(
        self, file_paths: str | Path | list[str | Path], *args, **kwargs
    ) -> tuple[list[str | None], list[str | None]]:
        raise NotImplementedError

    def stream(
        self, file_paths: str | Path | list[str | Path], reindex: bool = False, **kwargs
    ) -> Generator[
        Document, None, tuple[list[str | None], list[str | None], list[Document]]
    ]:
        """Return a list of indexed file ids, and a list of errors"""
        if not isinstance(file_paths, list):
            file_paths = [file_paths]

        file_ids: list[str | None] = []
        errors: list[str | None] = []
        all_docs = []

        n_files = len(file_paths)
        for idx, file_path in enumerate(file_paths):
            file_path = Path(file_path)
            yield Document(
                content=f"Indexing [{idx+1}/{n_files}]: {file_path.name}",
                channel="debug",
            )

            try:
                pipeline = self.route(file_path)
                file_id, docs = yield from pipeline.stream(
                    file_path, reindex=reindex, **kwargs
                )
                all_docs.extend(docs)
                file_ids.append(file_id)
                errors.append(None)
                yield Document(
                    content={"file_path": file_path, "status": "success"},
                    channel="index",
                )
            except Exception as e:
                logger.exception(e)
                file_ids.append(None)
                errors.append(str(e))
                yield Document(
                    content={
                        "file_path": file_path,
                        "status": "failed",
                        "message": str(e),
                    },
                    channel="index",
                )

        return file_ids, errors, all_docs
