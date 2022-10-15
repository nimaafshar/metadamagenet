import logging
from typing import Any, MutableMapping

import emoji

logging.basicConfig(format="%(message)s")


class EmojiAdapter(logging.LoggerAdapter):

    def process(self, msg: Any, **kwargs) -> tuple[Any, MutableMapping[str, Any]]:
        return emoji.emojize(msg, language='alias'), kwargs
