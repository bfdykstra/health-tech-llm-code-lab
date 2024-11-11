from typing import Optional

from lib.base_component import BaseComponent


class BaseReasoning(BaseComponent):
    """The reasoning pipeline that handles each of the user chat messages

    This reasoning pipeline has access to:
        - the retrievers
        - the user settings
        - the message
        - the conversation id
        - the message history
    """

    @classmethod
    def get_pipeline(
        cls,
        user_settings: dict,
        state: dict,
        retrievers: Optional[list["BaseComponent"]] = None,
    ) -> "BaseReasoning":
        """Get the reasoning pipeline for the app to execute

        Args:
            user_setting: user settings
            state: conversation state
            retrievers (list): List of retrievers
        """
        return cls()

    def run(self, message: str, conv_id: str, history: list, **kwargs):  # type: ignore
        """Execute the reasoning pipeline"""
        raise NotImplementedError
