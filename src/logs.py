import emoji


def log(text: str):
    print(emoji.emojize(text, language='alias'))
