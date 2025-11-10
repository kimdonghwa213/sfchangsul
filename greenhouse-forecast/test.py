import requests
from datetime import datetime

# 특정 날짜 테스트
url = "http://203.239.47.148:8080/dspnet.aspx"
params = {
    'Site': 85,
    'Dev': 1,
    'Year': 2024,
    'Mon': 10,
    'Day': 22
}

response = requests.get(url, params=params, timeout=10)
print(f"응답 길이: {len(response.text)}")
print(f"첫 100자: {response.text[:100]}")