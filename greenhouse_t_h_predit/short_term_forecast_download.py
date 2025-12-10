import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# 출력 인코딩 강제 설정 (윈도우 이모지 오류 방지)
sys.stdout.reconfigure(encoding='utf-8')

class WeatherForecastCollector:
    """기상청 단기예보 데이터 수집기"""
    
    def __init__(self, service_key):
        self.service_key = service_key
        self.base_url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0"
        
        # 김제시 백구면 좌표
        self.nx = 60
        self.ny = 90
        
        # 카테고리 매핑
        self.category_names = {
            'POP': 'rain_prob',          # 강수확률(%)
            'PTY': 'rain_type',          # 강수형태
            'PCP': 'rainfall',           # 1시간 강수량(mm)
            'REH': 'outer_hum',          # 습도(%)
            'SNO': 'snow',               # 1시간 신적설(cm)
            'SKY': 'sky_status',         # 하늘상태
            'TMP': 'outer_temp',         # 기온(℃)
            'TMN': 'min_temp',           # 최저기온(℃)
            'TMX': 'max_temp',           # 최고기온(℃)
            'UUU': 'wind_ew',            # 풍속-동서성분(m/s)
            'VVV': 'wind_ns',            # 풍속-남북성분(m/s)
            'WAV': 'wave_height',        # 파고(m)
            'VEC': 'wind_dir',           # 풍향(deg)
            'WSD': 'wind_speed',         # 풍속(m/s)
            'T1H': 'outer_temp',         # 기온(℃)
            'RN1': 'rainfall',           # 1시간 강수량(mm)
            'LGT': 'lightning'           # 낙뢰
        }
    
    def get_base_time_for_forecast(self, target_datetime):
        """단기예보 발표시각 계산 (02, 05, 08, 11, 14, 17, 20, 23시)"""
        hour = target_datetime.hour
        if hour < 2 or (hour == 2 and target_datetime.minute < 10):
            base_dt = target_datetime - timedelta(days=1)
            base_time = '2300'
        elif hour < 5 or (hour == 5 and target_datetime.minute < 10):
            base_dt = target_datetime
            base_time = '0200'
        elif hour < 8 or (hour == 8 and target_datetime.minute < 10):
            base_dt = target_datetime
            base_time = '0500'
        elif hour < 11 or (hour == 11 and target_datetime.minute < 10):
            base_dt = target_datetime
            base_time = '0800'
        elif hour < 14 or (hour == 14 and target_datetime.minute < 10):
            base_dt = target_datetime
            base_time = '1100'
        elif hour < 17 or (hour == 17 and target_datetime.minute < 10):
            base_dt = target_datetime
            base_time = '1400'
        elif hour < 20 or (hour == 20 and target_datetime.minute < 10):
            base_dt = target_datetime
            base_time = '1700'
        elif hour < 23 or (hour == 23 and target_datetime.minute < 10):
            base_dt = target_datetime
            base_time = '2000'
        else:
            base_dt = target_datetime
            base_time = '2300'
        
        return base_dt.strftime('%Y%m%d'), base_time
    
    def get_vilage_fcst(self, target_date=None):
        url = f"{self.base_url}/getVilageFcst"
        if target_date is None: target_date = datetime.now()
        base_date, base_time = self.get_base_time_for_forecast(target_date)
        
        params = {
            'serviceKey': self.service_key,
            'numOfRows': 1000,
            'pageNo': 1,
            'dataType': 'JSON',
            'base_date': base_date,
            'base_time': base_time,
            'nx': self.nx,
            'ny': self.ny
        }
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API 오류: {e}")
            return None
    
    def parse_forecast_to_dataframe(self, forecast_data):
        if not forecast_data or 'response' not in forecast_data: return None
        items = forecast_data['response']['body']['items']['item']
        
        forecast_dict = {}
        for item in items:
            fcst_date = item['fcstDate']
            fcst_time = item['fcstTime']
            time_key = f"{fcst_date}{fcst_time}"
            
            if time_key not in forecast_dict:
                forecast_dict[time_key] = {'Date&Time': pd.to_datetime(time_key, format='%Y%m%d%H%M')}
            
            category = item['category']
            if category in self.category_names:
                col_name = self.category_names[category]
                val = item['fcstValue']
                
                # 강수량 등 문자열 처리
                try:
                    if col_name == 'rainfall':
                        if '강수없음' in val: val = 0.0
                        else: val = float(val.replace('mm', ''))
                    else:
                        val = float(val)
                except: val = 0.0
                
                forecast_dict[time_key][col_name] = val
        
        df = pd.DataFrame.from_dict(forecast_dict, orient='index')
        df = df.sort_values('Date&Time').reset_index(drop=True)
        
        # 필수 컬럼 채우기
        req_cols = ['outer_temp', 'outer_hum', 'wind_speed', 'rain_prob', 'rainfall', 'sky_status', 'wind_dir']
        for col in req_cols:
            if col not in df.columns: df[col] = 0.0
            
        return df[['Date&Time'] + req_cols]
    
    def get_current_forecast(self):
        data = self.get_vilage_fcst()
        return self.parse_forecast_to_dataframe(data)

if __name__ == "__main__":
    SERVICE_KEY = ""
    collector = WeatherForecastCollector(SERVICE_KEY)
    
    print("📍 기상청 예보 다운로드 시작...")
    df = collector.get_current_forecast()
    
    if df is not None:
        # 폴더 생성
        os.makedirs('input', exist_ok=True)
        
        # 파일 저장 (app.py와 경로 일치시킴)
        SAVE_PATH = 'input/weather_forecast.csv'
        df.to_csv(SAVE_PATH, index=False, encoding='utf-8-sig')
        
        print(f"✅ 저장 완료: {SAVE_PATH}")
        print(f"📊 데이터 기간: {df['Date&Time'].min()} ~ {df['Date&Time'].max()}")
        print(f"📊 데이터 개수: {len(df)} rows")
    else:

        print("❌ 데이터 수집 실패")
