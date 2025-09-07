import pandas as pd
import os
from .text_processing import normalize
import logging

import math

pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)  
pd.set_option('display.max_colwidth', None) 
pd.set_option('display.width', 1000) 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
class Crawler:
    def __init__(self):
        self.columns = ["TITLE", "URL", "TEXT"]
        self.database = pd.DataFrame(columns=self.columns)
        logging.info("Initializing Crawler and starting crawl")
        self.crawl(path="local_fs")
        self.database.fillna(0, inplace=True)
        self.set_weights()
        self.save_database()
        logging.info("Crawler initialization completed")

    @staticmethod
    def tokenize(sentence: str) -> dict:
        logging.debug(f'Tokenizing sentence: {sentence}')
        sentence = normalize(sentence)

        frequency = dict(
            (word, sentence.count(word)) for word in set(sentence)
        )

        logging.debug(f'Token frequency: {frequency}')
        return frequency

    def crawl(self, path: str) -> None:
        logging.info(f'Start crawling directory: {path}')
        for root, dirs, files in os.walk(path):
            logging.info(f'Visiting directory: {root} containing {len(files)} files')
            for file in files:
                file_path = os.path.join(root, file)
                file_title = os.path.splitext(file)[0]
                logging.info(f'Processing file: {file_path}')
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_text = f.read()
                    logging.info(f'Read file successfully: {file_path}')
                except Exception as e:
                    file_text = ''
                    logging.warning(f'Failed to read file {file_path}: {e}')

                main_data = pd.DataFrame(
                    columns=self.columns,
                    data=[
                        [file_title, file_path, file_text]
                    ]
                )

                tokenized_sentence = self.tokenize(
                    main_data["TEXT"].iat[-1]
                )


                indexing_data = pd.DataFrame(
                    columns=tokenized_sentence.keys(),
                    data=[tokenized_sentence.values()],
                    dtype=pd.Int8Dtype()
                )
                logging.debug(f'Indexing data created with columns: {indexing_data.columns.tolist()}')

                data = pd.concat([main_data, indexing_data], axis=1, ignore_index=False)
                self.database = pd.concat([self.database, data], axis=0, ignore_index=True)
    
    def set_weights(self) -> None:
        counter = 0
        for column in self.database.columns[len(self.columns):]:
            counter += 1
            coef = math.log(len(self.database) / len(self.database[self.database[column] > 0]))
            self.database[column] *= coef

    def save_database(self) -> None:
        self.database.to_csv("processors/data.csv")


if __name__ == "__main__":
    crawl = Crawler()
