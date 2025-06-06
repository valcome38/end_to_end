from datetime import datetime, timedelta
import requests
import json

# Генерация 168 часовых точек
start = datetime.strptime("2024-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
data = []

for i in range(168):
    dt = start + timedelta(hours=i)
    data.append({
        "datetime": dt.isoformat(),
        "zone_id": "zone_0"
    })

payload = {"data": data}

# Отправка запроса на локальный FastAPI
response = requests.post("http://127.0.0.1:8000/predict", json=payload)

# Вывод результата
print(response.status_code)
print(response.json())

# Сохраняем в файл
#with open("predict_payload.json", "w") as f:
#    json.dump(payload, f, indent=2)
#
#print("✅ JSON сохранён в predict_payload.json")
