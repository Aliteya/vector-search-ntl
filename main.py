from processors import Crawler, Search
import os

def main_menu():

    while True:
        print("\n===== Главное меню поисковой системы =====")
        print("1. Создать/Обновить поисковый индекс")
        print("2. Начать поиск")
        print("3. Выйти")
        
        choice = input("Выберите действие: ")
        
        if choice == '1':
            if not os.path.exists("local_fs") or not os.listdir("local_fs"):
                print("Ошибка: папка 'local_fs' пуста. Сначала скачайте статьи (пункт 1).")
                continue
            print("\n--- Начинается процесс индексации... ---")
            crawler = Crawler()
            print("--- Индексация успешно завершена. Файл 'data.csv' создан/обновлен. ---")

        elif choice == '2':
            if not os.path.exists("processors/data.csv"):
                print("Ошибка: файл 'data.csv' не найден. Сначала создайте индекс (пункт 2).")
                continue

            search_engine = Search(database="processors/data.csv")
            print("\n--- Режим поиска ---")
            
            while True:
                user_query = input("Введите ваш запрос (или 'exit' для возврата в меню): ")
                if user_query.lower() == 'exit':
                    break
                
                search_results = search_engine.search(user_query)
                
                if search_results.empty:
                    print("По вашему запросу ничего не найдено.")
                else:
                    print("\n--- Результаты поиска: ---")
                    print(search_results)

        elif choice == '3':
            print("Выход из программы.")
            break
            
        else:
            print("Неверный выбор. Пожалуйста, введите число от 1 до 4.")

if __name__ == '__main__':
    main_menu()