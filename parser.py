import wikipedia
import warnings
import os

def is_article(title: str) -> bool:
    if not title.startswith("User:") and not title.startswith("Talk:"):
        return True
    return False


if __name__ == '__main__':
    wikipedia.set_lang("en")
    if not os.path.exists("local_fs2"):
        os.makedirs("local_fs2")
    page_titles = wikipedia.random(pages=20)
    warnings.catch_warnings()
    warnings.simplefilter("ignore")
    for page_title in page_titles:
        try:
            page = wikipedia.page(title=page_title)
            if is_article(page.title):
                page_url = "local_fs2/" + page.title + ".txt"
                with open(page_url, "w+", encoding="utf-8") as file:
                    file.write(page.summary)
        except AttributeError:
                pass
        except wikipedia.exceptions.PageError:
                pass
        except wikipedia.DisambiguationError:
                pass