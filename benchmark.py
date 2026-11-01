import time
import json
import http.client
from concurrent.futures import ThreadPoolExecutor

# Жесткие низкоуровневые настройки сети
HOST = "127.0.0.1"
PORT = 8000
PATH = "/predict"

print("[BENCHMARK] Инициализация низкоуровневого теста http.client...")
print("[BENCHMARK] Отправка 100 запросов к FastAPI напрямую в сокет...")

# Идеальный валидный JSON для твоего Pydantic (ровно 30 признаков)
payload_data = {"features": [0.1] * 30}
payload_bytes = json.dumps(payload_data).encode("utf-8")

def send_single_socket_request(request_id):
    conn = None
    try:
        # Открываем прямое TCP-соединение с сокетом
        conn = http.client.HTTPConnection(HOST, PORT, timeout=2.0)
        
        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(payload_bytes))
        }
        
        # Отправляем сырой POST запрос
        conn.request("POST", PATH, body=payload_bytes, headers=headers)
        
        # Получаем ответ сервера
        response = conn.getresponse()
        status = response.status
        
        # Вычитываем тело, чтобы освободить сокет
        response.read()
        
        if status != 200 and request_id == 0:
            print(f"\n[DEBUG] Сервер ответил кодом: {status}")
            
        return status == 200
    except Exception as e:
        if request_id == 0:
            print(f"\n[DEBUG] Ошибка сокета: {e}")
        return False
    finally:
        if conn:
            conn.close()

def main():
    print("[BENCHMARK] Запуск одновременной отправки 100 запросов через 10 потоков...")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(send_single_socket_request, range(100)))
        
    end_time = time.time()
    total_time = end_time - start_time
    
    successful_requests = sum(1 for r in results if r is True)
    failed_requests = len(results) - successful_requests

    print("\n=== РЕЗУЛЬТАТЫ БЕНЧМАРКА ===")
    print(f"Всего запросов: 100")
    print(f"Успешных ответов (200 OK): {successful_requests}")
    print(f"Ошибок/Таймаутов: {failed_requests}")
    
    if successful_requests > 0:
        print(f"Общее время выполнения теста: {total_time:.2f} сек")
        print(f"Среднее время на один запрос: {(total_time / 100) * 1000:.2f} мс")

if __name__ == "__main__":
    main()
