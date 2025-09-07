import pandas as pd
import os

# pd.set_option('display.max_columns', None) 
# pd.set_option('display.max_rows', None)  
# pd.set_option('display.max_colwidth', None) 
# pd.set_option('display.width', 1000)  

class Crawler:
    def __init__(self):
        self.columns = ["title", "url", "text"]
        self.database = pd.DataFrame(columns=self.columns)
        self.crawl(path="local_fs/")
        self.database.fillna(0, inplace=True)
        # self.set_weights()
        # self.save_database()

    def crawl(self, path: str) -> None:
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                file_title = os.path.splitext(file)[0]
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_text = f.read()
                except Exception as e:
                    file_text = ''
                main_data = pd.DataFrame(
                    columns=self.columns,
                    data = [
                        [file_title, file_path, file_text]
                    ]
                )  

crawl = Crawler()