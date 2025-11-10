import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Twilio (선택적)
try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

# ==================== 설정 ====================
st.set_page_config(
    page_title="🌱 온실 기상 예측 시스템",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== AWS 데이터 수집기 ====================
class AWSDataCollector:
    def __init__(self, site_id=85, dev_id=1):
        self.site_id = site_id
        self.dev_id = dev_id
        self.base_url = "http://203.239.47.148:8080/dspnet.aspx"
    
    def fetch_single_day(self, date):
        params = {
            'Site': self.site_id,
            'Dev': self.dev_id,
            'Year': date.year,
            'Mon': date.month,
            'Day': date.day
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            lines = response.text.strip().split('\n')
            data = []
            
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 17:
                    try:
                        data.append({
                            'datetime': datetime.strptime(parts[0].strip(), '%Y-%m-%d %H:%M:%S'),
                            'temperature': float(parts[1].strip()),
                            'humidity': float(parts[2].strip()),
                            'solar_radiation': float(parts[6].strip()),
                            'wind_direction': float(parts[7].strip()),
                            'wind_speed': float(parts[13].strip()),
                            'rainfall': float(parts[14].strip()),
                            'max_wind_gust': float(parts[15].strip()),
                            'battery_voltage': float(parts[16].strip())
                        })
                    except (ValueError, IndexError):
                        continue
            
            return pd.DataFrame(data)
        
        except Exception as e:
            return None
    
    def fetch_date_range(self, start_date, end_date):
        all_data = []
        current_date = start_date
        
        progress = st.progress(0, text="데이터 수집 중...")
        total_days = (end_date - start_date).days + 1
        
        day_count = 0
        while current_date <= end_date:
            df = self.fetch_single_day(current_date)
            
            if df is not None and len(df) > 0:
                all_data.append(df)
            
            current_date += timedelta(days=1)
            day_count += 1
            progress.progress(day_count / total_days, text=f"수집 중: {current_date.date()}")
        
        progress.empty()
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result = result.sort_values('datetime').reset_index(drop=True)
            return result
        else:
            return None

# ==================== 특성 공학 ====================
class WeatherFeatureEngineering:
    @staticmethod
    def add_time_features(df):
        df = df.copy()
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_year'] = df['datetime'].dt.dayofyear
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    @staticmethod
    def add_lag_features(df, columns, lags=[1, 3, 6, 12, 24]):
        df = df.copy()
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        return df
    
    @staticmethod
    def add_rolling_features(df, columns, windows=[3, 6, 12, 24]):
        df = df.copy()
        for col in columns:
            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
        return df
    
    @staticmethod
    def create_features(df):
        df = WeatherFeatureEngineering.add_time_features(df)
        main_cols = ['temperature', 'humidity', 'wind_speed', 'solar_radiation']
        df = WeatherFeatureEngineering.add_lag_features(df, main_cols)
        df = WeatherFeatureEngineering.add_rolling_features(df, main_cols)
        df = df.dropna().reset_index(drop=True)
        return df

# ==================== Random Forest 모델 ====================
class RandomForestWeatherModel:
    def __init__(self):
        self.models = {}
        self.scaler = MinMaxScaler()
        self.feature_columns = None
    
    def train(self, df, targets=['temperature', 'humidity', 'wind_speed']):
        self.feature_columns = [col for col in df.columns 
                               if col not in ['datetime', 'battery_voltage'] + targets]
        
        X = df[self.feature_columns].values
        X = self.scaler.fit_transform(X)
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        
        results = {}
        
        for target in targets:
            y = df[target].values
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            test_pred = model.predict(X_test)
            test_mae = mean_absolute_error(y_test, test_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            results[target] = {
                'mae': test_mae,
                'r2': test_r2
            }
            
            self.models[target] = model
        
        return results
    
    def predict(self, df, hours=24):
        predictions = {target: [] for target in self.models.keys()}
        current_df = df.copy()
        
        for hour in range(hours):
            X = current_df[self.feature_columns].tail(1).values
            X_scaled = self.scaler.transform(X)
            
            for target, model in self.models.items():
                pred = model.predict(X_scaled)[0]
                predictions[target].append(pred)
            
            next_row = current_df.iloc[-1].copy()
            next_row['datetime'] = next_row['datetime'] + timedelta(hours=1)
            
            for target in self.models.keys():
                next_row[target] = predictions[target][-1]
            
            next_row['hour'] = next_row['datetime'].hour
            next_row['day'] = next_row['datetime'].day
            next_row['month'] = next_row['datetime'].month
            
            current_df = pd.concat([current_df, next_row.to_frame().T], ignore_index=True)
            current_df = WeatherFeatureEngineering.create_features(current_df)
        
        return predictions

# ==================== 헬퍼 함수 ====================
def analyze_greenhouse_control(forecast_df, thresholds):
    if forecast_df is None or len(forecast_df) == 0:
        return []
    
    recommendations = []
    
    max_temp = forecast_df['temperature'].max()
    min_temp = forecast_df['temperature'].min()
    avg_humidity = forecast_df['humidity'].mean()
    
    if max_temp > thresholds['temp_high']:
        recommendations.append({
            'level': '⚠️ 경고',
            'category': '온도',
            'message': f'최고 {max_temp:.1f}°C 예상',
            'action': '환기창 개방, 차광막 설치'
        })
    elif min_temp < thresholds['temp_low']:
        recommendations.append({
            'level': '⚠️ 경고',
            'category': '온도',
            'message': f'최저 {min_temp:.1f}°C 예상',
            'action': '난방 시스템 가동'
        })
    else:
        recommendations.append({
            'level': '✅ 정상',
            'category': '온도',
            'message': f'{min_temp:.1f}~{max_temp:.1f}°C 적정',
            'action': '자동 모드 유지'
        })
    
    if avg_humidity > thresholds['humidity_high']:
        recommendations.append({
            'level': '⚠️ 경고',
            'category': '습도',
            'message': f'평균 {avg_humidity:.1f}% 예상',
            'action': '제습기 가동, 환기 강화'
        })
    elif avg_humidity < thresholds['humidity_low']:
        recommendations.append({
            'level': '⚠️ 경고',
            'category': '습도',
            'message': f'평균 {avg_humidity:.1f}% 예상',
            'action': '가습기 가동'
        })
    else:
        recommendations.append({
            'level': '✅ 정상',
            'category': '습도',
            'message': f'{avg_humidity:.1f}% 적정',
            'action': '자동 모드 유지'
        })
    
    return recommendations

def send_sms_alert(recommendations, twilio_config):
    if not TWILIO_AVAILABLE:
        return False, "Twilio가 설치되지 않았습니다."
    
    if not all([twilio_config['sid'], twilio_config['token'], 
                twilio_config['from'], twilio_config['to']]):
        return False, "Twilio 설정을 완료해주세요."
    
    try:
        client = Client(twilio_config['sid'], twilio_config['token'])
        alerts = [r for r in recommendations if '경고' in r['level']]
        
        if alerts:
            body = "🌱 온실 제어 알림\n\n"
            for alert in alerts:
                body += f"{alert['level']} {alert['category']}\n"
                body += f"{alert['message']}\n"
                body += f"→ {alert['action']}\n\n"
            
            message = client.messages.create(
                body=body,
                from_=twilio_config['from'],
                to=twilio_config['to']
            )
            return True, "알림 발송 완료"
        else:
            return True, "모든 항목 정상"
    except Exception as e:
        return False, f"발송 실패: {str(e)}"

# ==================== 세션 초기화 ====================
if 'trained_model' not in st.session_state:
    st.session_state['trained_model'] = None

# ==================== 사이드바 ====================
with st.sidebar:
    st.header("⚙️ 시스템 설정")
    
    st.subheader("📡 AWS 데이터")
    site_id = st.number_input("Site ID", value=85, min_value=1)
    dev_id = st.number_input("Device ID", value=1, min_value=1)
    
    st.subheader("📱 Twilio 알림")
    if TWILIO_AVAILABLE:
        with st.expander("Twilio 설정"):
            twilio_account_sid = st.text_input("Account SID", type="password")
            twilio_auth_token = st.text_input("Auth Token", type="password")
            twilio_from_number = st.text_input("From Number")
            twilio_to_number = st.text_input("To Number")
    else:
        st.warning("Twilio 미설치 (SMS 기능 비활성)")
        twilio_account_sid = twilio_auth_token = ""
        twilio_from_number = twilio_to_number = ""
    
    st.subheader("⚠️ 알림 임계값")
    temp_high = st.number_input("고온 (°C)", value=35.0)
    temp_low = st.number_input("저온 (°C)", value=5.0)
    humidity_high = st.number_input("고습 (%)", value=90.0)
    humidity_low = st.number_input("저습 (%)", value=30.0)

# ==================== 메인 화면 ====================
st.title("🌱 온실 기상 예측 시스템")
st.markdown("### Random Forest 기반 AI 예측")
st.info("💡 이 버전은 Python 3.12 호환을 위해 Random Forest 모델만 사용합니다.")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 데이터 수집", "🤖 모델 학습", "🔮 예측 및 제어"])

# ==================== 탭 1: 데이터 수집 ====================
with tab1:
    st.header("📡 AWS 데이터 수집")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        date_range = st.date_input(
            "수집 기간",
            value=(datetime.now() - timedelta(days=7), datetime.now()),
            max_value=datetime.now()
        )
    
    with col2:
        if st.button("🔄 수집 시작", type="primary", use_container_width=True):
            if len(date_range) == 2:
                collector = AWSDataCollector(site_id, dev_id)
                start_date = datetime.combine(date_range[0], datetime.min.time())
                end_date = datetime.combine(date_range[1], datetime.max.time())
                
                df = collector.fetch_date_range(start_date, end_date)
                
                if df is not None and len(df) > 0:
                    st.session_state['raw_data'] = df
                    st.success(f"✅ {len(df)}개 데이터 수집 완료!")
                else:
                    st.error("데이터 수집 실패")
    
    if 'raw_data' in st.session_state:
        df = st.session_state['raw_data']
        
        st.subheader("📌 현재 기상 상태")
        latest = df.iloc[-1]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("온도", f"{latest['temperature']:.1f}°C")
        col2.metric("습도", f"{latest['humidity']:.1f}%")
        col3.metric("풍속", f"{latest['wind_speed']:.1f}m/s")
        col4.metric("일사량", f"{latest['solar_radiation']:.0f}W/m²")
        col5.metric("강우", f"{latest['rainfall']:.1f}mm")
        
        st.subheader("📈 시계열 그래프")
        
        fig = make_subplots(rows=2, cols=2, subplot_titles=('온도', '습도', '풍속', '일사량'))
        
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['temperature'], 
                      name='온도', line=dict(color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['humidity'],
                      name='습도', line=dict(color='blue')), row=1, col=2)
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['wind_speed'],
                      name='풍속', line=dict(color='green')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['solar_radiation'],
                      name='일사량', line=dict(color='orange')), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ==================== 탭 2: 모델 학습 ====================
with tab2:
    st.header("🤖 Random Forest 모델 학습")
    
    if 'raw_data' not in st.session_state:
        st.warning("⚠️ 먼저 데이터를 수집해주세요.")
    else:
        df = st.session_state['raw_data']
        st.info(f"📊 데이터: {len(df)}개 ({df['datetime'].min()} ~ {df['datetime'].max()})")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("예측 타겟")
            predict_temp = st.checkbox("온도", value=True)
            predict_humidity = st.checkbox("습도", value=True)
            predict_wind = st.checkbox("풍속", value=True)
        
        if st.button("🚀 학습 시작", type="primary", use_container_width=True):
            with st.spinner("모델 학습 중..."):
                progress = st.progress(0, text="특성 생성 중...")
                df_features = WeatherFeatureEngineering.create_features(df)
                st.session_state['training_data'] = df_features
                progress.progress(30, text="특성 생성 완료")
                
                targets = []
                if predict_temp: targets.append('temperature')
                if predict_humidity: targets.append('humidity')
                if predict_wind: targets.append('wind_speed')
                
                progress.progress(50, text="Random Forest 학습 중...")
                model = RandomForestWeatherModel()
                results = model.train(df_features, targets=targets)
                
                st.session_state['trained_model'] = model
                progress.progress(100, text="학습 완료!")
                
                st.success("✅ 학습 완료!")
                
                for target, metrics in results.items():
                    st.metric(f"{target} MAE", f"{metrics['mae']:.3f}")
                    st.metric(f"{target} R²", f"{metrics['r2']:.3f}")

# ==================== 탭 3: 예측 ====================
with tab3:
    st.header("🔮 24시간 기상 예측")
    
    if 'training_data' not in st.session_state:
        st.warning("⚠️ 먼저 모델을 학습해주세요.")
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            forecast_hours = st.slider("예측 시간 (시간)", 6, 48, 24)
        
        with col2:
            if st.button("🔮 예측", type="primary", use_container_width=True):
                with st.spinner("예측 중..."):
                    model = st.session_state['trained_model']
                    df_features = st.session_state['training_data']
                    
                    preds = model.predict(df_features, hours=forecast_hours)
                    
                    last_datetime = df_features['datetime'].iloc[-1]
                    forecast_times = [last_datetime + timedelta(hours=i+1)
                                    for i in range(forecast_hours)]
                    
                    predictions = pd.DataFrame({
                        'datetime': forecast_times,
                        'temperature': preds.get('temperature', []),
                        'humidity': preds.get('humidity', []),
                        'wind_speed': preds.get('wind_speed', [])
                    })
                    
                    st.session_state['predictions'] = predictions
                    st.success("✅ 예측 완료!")
        
        if 'predictions' in st.session_state:
            predictions = st.session_state['predictions']
            
            st.subheader("📊 예측 결과")
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=('온도 예측', '습도 예측'))
            
            df_features = st.session_state['training_data']
            recent = df_features.tail(forecast_hours)
            
            fig.add_trace(go.Scatter(x=recent['datetime'], y=recent['temperature'],
                          name='실제', line=dict(color='red', dash='solid')), row=1, col=1)
            fig.add_trace(go.Scatter(x=predictions['datetime'], y=predictions['temperature'],
                          name='예측', line=dict(color='red', dash='dash')), row=1, col=1)
            
            if 'humidity' in predictions.columns:
                fig.add_trace(go.Scatter(x=recent['datetime'], y=recent['humidity'],
                              name='실제', line=dict(color='blue', dash='solid')), row=1, col=2)
                fig.add_trace(go.Scatter(x=predictions['datetime'], y=predictions['humidity'],
                              name='예측', line=dict(color='blue', dash='dash')), row=1, col=2)
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🎯 온실 제어 권장사항")
            
            thresholds = {
                'temp_high': temp_high,
                'temp_low': temp_low,
                'humidity_high': humidity_high,
                'humidity_low': humidity_low
            }
            
            recommendations = analyze_greenhouse_control(predictions, thresholds)
            
            for rec in recommendations:
                if '경고' in rec['level']:
                    st.warning(f"**{rec['level']} [{rec['category']}]**\n\n{rec['message']}\n\n**→ {rec['action']}**")
                else:
                    st.success(f"**{rec['level']} [{rec['category']}]**\n\n{rec['message']}\n\n**→ {rec['action']}**")
            
            if TWILIO_AVAILABLE:
                st.markdown("---")
                if st.button("📱 SMS 알림 발송"):
                    twilio_config = {
                        'sid': twilio_account_sid,
                        'token': twilio_auth_token,
                        'from': twilio_from_number,
                        'to': twilio_to_number
                    }
                    
                    success, message = send_sms_alert(recommendations, twilio_config)
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>🌱 온실 기상 예측 시스템 v1.0</div>", 
            unsafe_allow_html=True)