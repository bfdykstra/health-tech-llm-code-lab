from pathlib import Path
from typing import Type

from decouple import config

# from lib.loaders import BaseReader
from llama_index.core.readers.base import BaseReader

from theflow.settings import settings as flowsettings
from theflow import Param

from lib.base import Document
from lib.base_component import BaseComponent 
from lib.indices.extractors import BaseDocParser
from lib.indices.splitters import BaseSplitter, TokenSplitter
from lib.loaders import (
    TxtReader, DirectoryReader
)


DEFAULT_FILE_EXTRACTORS: dict[str, BaseReader] = {
    ".txt": TxtReader(),
    ".md": TxtReader(),
}


class DocumentIngestor(BaseComponent):
    """Ingest common office document types into Document for indexing

    Document types:
        - .txt, .md

    Args:
        doc_parsers: list of document parsers to parse the document
        text_splitter: splitter to split the document into text nodes
        override_file_extractors: override file extractors for specific file extensions
            The default file extractors are stored in `DEFAULT_FILE_EXTRACTORS`
    """

    doc_parsers: list[BaseDocParser] = Param(default_callback=lambda _: [])
    text_splitter: BaseSplitter = TokenSplitter.withx(
        chunk_size=1024,
        chunk_overlap=256,
        separator="\n\n",
        backup_separators=["\n", ".", " ", "\u200B"],
    )
    override_file_extractors: dict[str, Type[BaseReader]] = {}

    def _get_reader(self, input_files: list[str | Path]):
        """Get appropriate readers for the input files based on file extension"""
        file_extractors: dict[str, BaseReader] = {
            ext: reader for ext, reader in DEFAULT_FILE_EXTRACTORS.items()
        }
        for ext, cls in self.override_file_extractors.items():
            file_extractors[ext] = cls()

        main_reader = DirectoryReader(
            input_files=input_files,
            file_extractor=file_extractors,
        )

        return main_reader

    def run(self, file_paths: list[str | Path] | str | Path) -> list[Document]:
        """Ingest the file paths into Document

        Args:
            file_paths: list of file paths or a single file path

        Returns:
            list of parsed Documents
        """
        if not isinstance(file_paths, list):
            file_paths = [file_paths]

        documents = self._get_reader(input_files=file_paths)()
        print(f"Read {len(file_paths)} files into {len(documents)} documents.")
        nodes = self.text_splitter(documents)
        print(f"Transform {len(documents)} documents into {len(nodes)} nodes.")
        self.log_progress(".num_docs", num_docs=len(nodes))

        # document parsers call
        if self.doc_parsers:
            for parser in self.doc_parsers:
                nodes = parser(nodes)

        return nodes
