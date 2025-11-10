"""
AWS 기상 데이터 기반 ML 예측 모델
- LSTM: 시계열 딥러닝 모델
- Prophet: Facebook 시계열 예측
- Random Forest: 앙상블 모델
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML 라이브러리
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 딥러닝
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Prophet (Facebook 시계열 예측)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet 설치 권장: pip install prophet")


class AWSDataCollector:
    """AWS 데이터 수집기"""
    
    def __init__(self, site_id=85, dev_id=1):
        self.site_id = site_id
        self.dev_id = dev_id
        self.base_url = "http://203.239.47.148:8080/dspnet.aspx"
    
    def fetch_single_day(self, date):
        """하루 데이터 가져오기"""
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
            print(f"날짜 {date} 데이터 수집 실패: {str(e)}")
            return None
    
    def fetch_date_range(self, start_date, end_date):
        """기간 데이터 수집"""
        all_data = []
        current_date = start_date
        
        print(f"데이터 수집 중: {start_date.date()} ~ {end_date.date()}")
        
        while current_date <= end_date:
            print(f"  수집 중: {current_date.date()}", end='\r')
            df = self.fetch_single_day(current_date)
            
            if df is not None and len(df) > 0:
                all_data.append(df)
            
            current_date += timedelta(days=1)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result = result.sort_values('datetime').reset_index(drop=True)
            print(f"\n✅ 총 {len(result)}개 데이터 수집 완료")
            return result
        else:
            print("\n❌ 데이터 수집 실패")
            return None


class WeatherFeatureEngineering:
    """기상 데이터 특성 공학"""
    
    @staticmethod
    def add_time_features(df):
        """시간 관련 특성 추가"""
        df = df.copy()
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_year'] = df['datetime'].dt.dayofyear
        
        # 주기적 특성 (사인/코사인 인코딩)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    @staticmethod
    def add_lag_features(df, columns, lags=[1, 3, 6, 12, 24]):
        """과거 값 특성 추가"""
        df = df.copy()
        
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    @staticmethod
    def add_rolling_features(df, columns, windows=[3, 6, 12, 24]):
        """이동 평균 특성 추가"""
        df = df.copy()
        
        for col in columns:
            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
        
        return df
    
    @staticmethod
    def create_features(df):
        """전체 특성 생성"""
        df = WeatherFeatureEngineering.add_time_features(df)
        
        # 주요 변수에 대한 lag 및 rolling 특성
        main_cols = ['temperature', 'humidity', 'wind_speed', 'solar_radiation']
        df = WeatherFeatureEngineering.add_lag_features(df, main_cols)
        df = WeatherFeatureEngineering.add_rolling_features(df, main_cols)
        
        # 결측치 제거 (lag, rolling으로 인한)
        df = df.dropna().reset_index(drop=True)
        
        return df


class LSTMWeatherModel:
    """LSTM 기반 시계열 예측 모델"""
    
    def __init__(self, sequence_length=24, n_features=None):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_columns = None
        
    def prepare_sequences(self, data, target_col):
        """시퀀스 데이터 생성"""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length][target_col])
        
        return np.array(X), np.array(y)
    
    def build_model(self):
        """LSTM 모델 구축"""
        model = Sequential([
            LSTM(128, activation='relu', return_sequences=True, 
                 input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            LSTM(64, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train(self, df, target_col='temperature', epochs=100, batch_size=32):
        """모델 학습"""
        # 특성 선택
        self.feature_columns = [col for col in df.columns 
                               if col not in ['datetime', 'battery_voltage']]
        
        # 데이터 스케일링
        scaled_data = self.scaler.fit_transform(df[self.feature_columns])
        
        # 타겟 컬럼 인덱스
        target_idx = self.feature_columns.index(target_col)
        
        # 시퀀스 생성
        X, y = self.prepare_sequences(scaled_data, target_idx)
        
        # Train/Validation 분할
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"학습 데이터: {X_train.shape}, 검증 데이터: {X_val.shape}")
        
        # 모델 구축
        self.n_features = X.shape[2]
        self.model = self.build_model()
        
        # 콜백 설정
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('best_lstm_model.h5', save_best_only=True, monitor='val_loss')
        ]
        
        # 학습
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # 평가
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        print(f"\n✅ 학습 완료!")
        print(f"Train MAE: {train_mae:.4f}")
        print(f"Validation MAE: {val_mae:.4f}")
        
        return history
    
    def predict_future(self, df, hours=24):
        """미래 예측"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # 최근 데이터로 시퀀스 생성
        recent_data = df[self.feature_columns].tail(self.sequence_length).values
        scaled_recent = self.scaler.transform(recent_data)
        
        predictions = []
        current_sequence = scaled_recent.copy()
        
        for _ in range(hours):
            # 예측
            X_pred = current_sequence.reshape(1, self.sequence_length, self.n_features)
            pred_scaled = self.model.predict(X_pred, verbose=0)[0, 0]
            
            # 다음 시퀀스 생성 (예측값을 포함)
            next_step = current_sequence[-1].copy()
            next_step[0] = pred_scaled  # 온도 위치에 예측값 업데이트
            
            current_sequence = np.vstack([current_sequence[1:], next_step])
            predictions.append(pred_scaled)
        
        # 역스케일링
        predictions = np.array(predictions).reshape(-1, 1)
        
        # 전체 특성을 위한 더미 데이터 생성
        dummy = np.zeros((len(predictions), len(self.feature_columns)))
        dummy[:, 0] = predictions.flatten()
        predictions_unscaled = self.scaler.inverse_transform(dummy)[:, 0]
        
        return predictions_unscaled
    
    def save(self, filepath='lstm_model.h5'):
        """모델 저장"""
        self.model.save(filepath)
        
        # 스케일러와 설정 저장
        config = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length,
            'n_features': self.n_features
        }
        with open(filepath.replace('.h5', '_config.pkl'), 'wb') as f:
            pickle.dump(config, f)
    
    def load(self, filepath='lstm_model.h5'):
        """모델 로드"""
        self.model = load_model(filepath)
        
        with open(filepath.replace('.h5', '_config.pkl'), 'rb') as f:
            config = pickle.load(f)
        
        self.scaler = config['scaler']
        self.feature_columns = config['feature_columns']
        self.sequence_length = config['sequence_length']
        self.n_features = config['n_features']


class RandomForestWeatherModel:
    """Random Forest 기반 예측 모델"""
    
    def __init__(self):
        self.models = {}
        self.scaler = MinMaxScaler()
        self.feature_columns = None
    
    def train(self, df, targets=['temperature', 'humidity', 'wind_speed']):
        """다중 타겟 학습"""
        # 특성 선택
        self.feature_columns = [col for col in df.columns 
                               if col not in ['datetime', 'battery_voltage'] + targets]
        
        X = df[self.feature_columns].values
        X = self.scaler.fit_transform(X)
        
        # Train/Test 분할
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        
        print(f"학습 데이터: {X_train.shape[0]}, 테스트 데이터: {X_test.shape[0]}")
        
        # 각 타겟별 모델 학습
        for target in targets:
            print(f"\n📊 {target} 모델 학습 중...")
            
            y = df[target].values
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Random Forest 모델
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # 평가
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            print(f"  Train MAE: {train_mae:.4f}")
            print(f"  Test MAE: {test_mae:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            
            # 중요 특성 출력
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"  Top 5 중요 특성:")
            for idx, row in feature_importance.head(5).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")
            
            self.models[target] = model
        
        print("\n✅ 모든 모델 학습 완료!")
    
    def predict(self, df, hours=24):
        """미래 예측"""
        predictions = {target: [] for target in self.models.keys()}
        
        # 최근 데이터로 시작
        current_df = df.copy()
        
        for hour in range(hours):
            # 특성 추출
            X = current_df[self.feature_columns].tail(1).values
            X_scaled = self.scaler.transform(X)
            
            # 각 타겟 예측
            for target, model in self.models.items():
                pred = model.predict(X_scaled)[0]
                predictions[target].append(pred)
            
            # 다음 시간 데이터 생성 (예측값 사용)
            next_row = current_df.iloc[-1].copy()
            next_row['datetime'] = next_row['datetime'] + timedelta(hours=1)
            
            for target in self.models.keys():
                next_row[target] = predictions[target][-1]
            
            # 특성 재계산 (시간 관련)
            next_row['hour'] = next_row['datetime'].hour
            next_row['day'] = next_row['datetime'].day
            next_row['month'] = next_row['datetime'].month
            
            current_df = pd.concat([current_df, next_row.to_frame().T], ignore_index=True)
        
        return predictions
    
    def save(self, filepath='rf_models.pkl'):
        """모델 저장"""
        config = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        with open(filepath, 'wb') as f:
            pickle.dump(config, f)
    
    def load(self, filepath='rf_models.pkl'):
        """모델 로드"""
        with open(filepath, 'rb') as f:
            config = pickle.load(f)
        
        self.models = config['models']
        self.scaler = config['scaler']
        self.feature_columns = config['feature_columns']


class ProphetWeatherModel:
    """Prophet 기반 예측 모델"""
    
    def __init__(self):
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet이 설치되지 않았습니다: pip install prophet")
        
        self.models = {}
    
    def train(self, df, targets=['temperature', 'humidity']):
        """Prophet 모델 학습"""
        for target in targets:
            print(f"\n📊 {target} Prophet 모델 학습 중...")
            
            # Prophet 형식으로 변환
            prophet_df = pd.DataFrame({
                'ds': df['datetime'],
                'y': df[target]
            })
            
            # 모델 생성 및 학습
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            model.fit(prophet_df)
            self.models[target] = model
            
            print(f"  ✅ {target} 모델 학습 완료")
    
    def predict(self, hours=24):
        """미래 예측"""
        predictions = {}
        
        for target, model in self.models.items():
            # 미래 날짜 생성
            future = model.make_future_dataframe(periods=hours, freq='H')
            
            # 예측
            forecast = model.predict(future)
            
            # 마지막 24시간 예측값 추출
            predictions[target] = forecast['yhat'].tail(hours).values
        
        return predictions
    
    def save(self, filepath='prophet_models.pkl'):
        """모델 저장"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.models, f)
    
    def load(self, filepath='prophet_models.pkl'):
        """모델 로드"""
        with open(filepath, 'rb') as f:
            self.models = pickle.load(f)


# ==================== 사용 예제 ====================

if __name__ == "__main__":
    print("🌱 AWS 기상 데이터 ML 예측 시스템\n")
    
    # 1. 데이터 수집
    print("=" * 50)
    print("1️⃣ 데이터 수집")
    print("=" * 50)
    
    collector = AWSDataCollector(site_id=85, dev_id=1)
    
    # 최근 3개월 데이터 수집 (실제로는 더 많은 데이터 권장)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    df = collector.fetch_date_range(start_date, end_date)
    
    if df is None or len(df) == 0:
        print("❌ 데이터 수집 실패. 샘플 데이터로 진행합니다.")
        # 샘플 데이터 생성
        dates = pd.date_range(start=start_date, end=end_date, freq='10T')
        df = pd.DataFrame({
            'datetime': dates,
            'temperature': 20 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 144) + np.random.randn(len(dates)),
            'humidity': 60 + 20 * np.cos(np.arange(len(dates)) * 2 * np.pi / 144) + np.random.randn(len(dates)) * 5,
            'wind_speed': 2 + np.random.randn(len(dates)) * 0.5,
            'solar_radiation': np.maximum(0, 500 * np.sin(np.arange(len(dates)) * 2 * np.pi / 144)),
            'wind_direction': np.random.uniform(0, 360, len(dates)),
            'rainfall': 0,
            'max_wind_gust': 3 + np.random.randn(len(dates)) * 0.5,
            'battery_voltage': 12.5 + np.random.randn(len(dates)) * 0.1
        })
    
    # 2. 특성 공학
    print("\n" + "=" * 50)
    print("2️⃣ 특성 공학")
    print("=" * 50)
    
    df_features = WeatherFeatureEngineering.create_features(df)
    print(f"특성 개수: {len(df_features.columns)}")
    print(f"데이터 개수: {len(df_features)}")
    
    # 3. LSTM 모델 학습
    print("\n" + "=" * 50)
    print("3️⃣ LSTM 모델 학습")
    print("=" * 50)
    
    lstm_model = LSTMWeatherModel(sequence_length=24)
    lstm_model.train(df_features, target_col='temperature', epochs=50)
    lstm_model.save('lstm_weather_model.h5')
    
    # 예측
    lstm_predictions = lstm_model.predict_future(df_features, hours=24)
    print(f"\nLSTM 24시간 온도 예측: {lstm_predictions[:5]} ...")
    
    # 4. Random Forest 모델 학습
    print("\n" + "=" * 50)
    print("4️⃣ Random Forest 모델 학습")
    print("=" * 50)
    
    rf_model = RandomForestWeatherModel()
    rf_model.train(df_features, targets=['temperature', 'humidity', 'wind_speed'])
    rf_model.save('rf_weather_models.pkl')
    
    # 예측
    rf_predictions = rf_model.predict(df_features, hours=24)
    print(f"\nRandom Forest 24시간 온도 예측: {rf_predictions['temperature'][:5]} ...")
    
    # 5. Prophet 모델 학습 (선택적)
    if PROPHET_AVAILABLE:
        print("\n" + "=" * 50)
        print("5️⃣ Prophet 모델 학습")
        print("=" * 50)
        
        prophet_model = ProphetWeatherModel()
        prophet_model.train(df, targets=['temperature', 'humidity'])
        prophet_model.save('prophet_weather_models.pkl')
        
        prophet_predictions = prophet_model.predict(hours=24)
        print(f"\nProphet 24시간 온도 예측: {prophet_predictions['temperature'][:5]} ...")
    
    print("\n" + "=" * 50)
    print("✅ 모든 모델 학습 및 저장 완료!")
    print("=" * 50)