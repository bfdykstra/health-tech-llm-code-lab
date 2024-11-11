from __future__ import annotations

from abc import abstractmethod

from lib.base import Document
from lib.base_component import BaseComponent


class BaseReranking(BaseComponent):
    @abstractmethod
    def run(self, documents: list[Document], query: str) -> list[Document]:
        """Main method to transform list of documents
        (re-ranking, filtering, etc)"""
        ...
