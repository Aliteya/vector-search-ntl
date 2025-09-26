import os
import logging
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from watchdog.observers import Observer
from processors import VectorIndexer, VectorSearch, FileChangeHandler 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PATHS_TO_WATCH = ["local_fs", "local_fs2"]
INDEX_PATH = "processors/vector_index.faiss"
MAP_PATH = "processors/path_map.pkl"

for path in PATHS_TO_WATCH:
    if not os.path.exists(path):
        logging.warning(f"Папка '{path}' не найдена. Создаю пустую папку.")
        os.makedirs(path)

logging.info("Запуск индексации (векторной)...")
indexer = VectorIndexer(watch_paths=PATHS_TO_WATCH, index_path=INDEX_PATH, map_path=MAP_PATH)
logging.info("Векторный индекс готов.")

event_handler = FileChangeHandler(indexer)
observer = Observer()

for path in PATHS_TO_WATCH:
    observer.schedule(event_handler, path, recursive=True)
    logging.info(f"Наблюдатель настроен для папки: '{path}'.")

observer.start()
logging.info(f"Наблюдатель запущен и отслеживает изменения в: {PATHS_TO_WATCH}.")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def startup_event():
    app.state.indexer = indexer

@app.get("/", response_class=HTMLResponse)
async def show_search_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": []})

@app.post("/", response_class=HTMLResponse)
async def handle_search_query(request: Request, query: str = Form(...)):
    results_list = []
    if query:
        search_engine = VectorSearch(indexer=app.state.indexer)
        
        results_list = search_engine.search(query)
        
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "results": results_list, 
        "query": query
    })

if __name__ == '__main__':
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000)
    finally:
        logging.info("Application is shutting down...")
        observer.stop()
        observer.join()

        indexer.save_index() 
        logging.info("Shutdown complete.")
# import os
# import logging
# import uvicorn
# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from watchdog.observers import Observer

# from processors import IndexUpdater, FileChangeHandler
# from processors import Search

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# PATHS_TO_WATCH = ["local_fs", "local_fs2"]
# DB_PATH = "processors/data.csv"

# for path in PATHS_TO_WATCH:
#     if not os.path.exists(path):
#         logging.warning(f"Папка '{path}' не найдена. Создаю пустую папку.")
#         os.makedirs(path)

# logging.info("Запуск первоначальной индексации...")
# indexer = IndexUpdater(watch_paths=PATHS_TO_WATCH, db_path=DB_PATH)
# logging.info("Первоначальная индексация завершена.")

# event_handler = FileChangeHandler(indexer)
# observer = Observer()

# for path in PATHS_TO_WATCH:
#     observer.schedule(event_handler, path, recursive=True)
#     logging.info(f"Наблюдатель настроен для папки: '{path}'.")

# observer.start()
# logging.info(f"Наблюдатель запущен и отслеживает изменения в: {PATHS_TO_WATCH}.")

# app = FastAPI()

# templates = Jinja2Templates(directory="templates")

# @app.get("/", response_class=HTMLResponse)
# async def show_search_form(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/", response_class=HTMLResponse)
# async def handle_search_query(request: Request, query: str = Form(...)):
#     results_list = []
#     if query:
#         if not os.path.exists(DB_PATH):
#              return templates.TemplateResponse("index.html", {
#                 "request": request, "query": query, "results": []
#             })
        
#         search_engine = Search(database=indexer.tfidf_database)
#         search_results_df = search_engine.search(query)
        
#         results_list = search_results_df.to_dict(orient='records')

#     return templates.TemplateResponse("index.html", {
#         "request": request, 
#         "results": results_list, 
#         "query": query
#     })


# if __name__ == '__main__':
#     try:
#         uvicorn.run(app, host="127.0.0.1", port=8000)
#     finally:
#         logging.info("Application is shutting down...")
#         observer.stop()
#         observer.join()
#         indexer.save_database()
#         logging.info("Shutdown complete.")