# mqtt_listener.py (수정됨: 서버 메인 로직)

import paho.mqtt.client as mqtt
import json
import joblib
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

print("--- 하이브리드 위험 예측 서버 (MQTT) ---")
print("모델과 스케일러를 로드합니다...")

# =============================================================================
# 파트 1: 규칙 기반 모델 (분류기)
# =============================================================================
# (AI 학습 및 예측에 모두 필요하므로 훈련 스크립트와 동일하게 정의)

WIND_CHILL_DATA = {
    25: {25: 22.2, 30: 22.8, 35: 23.3, 40: 23.8, 45: 24.2, 50: 24.6, 55: 25.1, 60: 25.5, 65: 25.9, 70: 26.2, 75: 26.6, 80: 27.0, 85: 27.3, 90: 27.7, 95: 28.0, 100: 28.4},
    26: {25: 23.1, 30: 23.7, 35: 24.2, 40: 24.7, 45: 25.2, 50: 25.6, 55: 26.0, 60: 26.5, 65: 26.9, 70: 27.2, 75: 27.6, 80: 28.0, 85: 28.4, 90: 28.7, 95: 29.1, 100: 29.4},
    27: {25: 24.0, 30: 24.6, 35: 25.2, 40: 25.7, 45: 26.1, 50: 26.6, 55: 27.0, 60: 27.5, 65: 27.9, 70: 28.2, 75: 28.6, 80: 29.0, 85: 29.4, 90: 29.7, 95: 30.1, 100: 30.5},
    28: {25: 24.9, 30: 25.5, 35: 26.1, 40: 26.6, 45: 27.0, 50: 27.5, 55: 28.0, 60: 28.4, 65: 28.9, 70: 29.3, 75: 29.7, 80: 30.0, 85: 30.4, 90: 30.8, 95: 31.1, 100: 31.5},
    29: {25: 25.8, 30: 26.5, 35: 27.0, 40: 27.6, 45: 28.1, 50: 28.6, 55: 29.0, 60: 29.4, 65: 29.9, 70: 30.3, 75: 30.7, 80: 31.1, 85: 31.4, 90: 31.8, 95: 32.2, 100: 32.6},
    30: {25: 26.7, 30: 27.4, 35: 28.0, 40: 28.5, 45: 29.0, 50: 29.5, 55: 30.0, 60: 30.4, 65: 30.9, 70: 31.3, 75: 31.7, 80: 32.1, 85: 32.5, 90: 32.9, 95: 33.2, 100: 33.6},
    31: {25: 27.6, 30: 28.3, 35: 28.9, 40: 29.4, 45: 30.0, 50: 30.5, 55: 31.0, 60: 31.4, 65: 31.9, 70: 32.3, 75: 32.7, 80: 33.1, 85: 33.5, 90: 33.9, 95: 34.3, 100: 34.6},
    32: {25: 28.5, 30: 29.2, 35: 29.8, 40: 30.4, 45: 31.0, 50: 31.5, 55: 32.0, 60: 32.4, 65: 32.9, 70: 33.3, 75: 33.7, 80: 34.1, 85: 34.5, 90: 34.9, 95: 35.3, 100: 35.7},
    33: {25: 29.5, 30: 30.2, 35: 30.8, 40: 31.4, 45: 32.0, 50: 32.5, 55: 33.0, 60: 33.5, 65: 33.9, 70: 34.3, 75: 34.8, 80: 35.2, 85: 35.6, 90: 36.0, 95: 36.4, 100: 36.7},
    34: {25: 30.4, 30: 31.1, 35: 31.8, 40: 32.4, 45: 32.9, 50: 33.5, 55: 34.0, 60: 34.5, 65: 34.9, 70: 35.4, 75: 35.8, 80: 36.2, 85: 36.6, 90: 37.0, 95: 37.4, 100: 37.8},
    35: {25: 31.3, 30: 32.0, 35: 32.7, 40: 33.3, 45: 33.9, 50: 34.5, 55: 35.0, 60: 35.5, 65: 35.9, 70: 36.4, 75: 36.8, 80: 37.2, 85: 37.7, 90: 38.1, 95: 38.5, 100: 38.9},
    36: {25: 32.2, 30: 33.0, 35: 33.7, 40: 34.3, 45: 34.9, 50: 35.4, 55: 36.0, 60: 36.5, 65: 36.9, 70: 37.4, 75: 37.8, 80: 38.3, 85: 38.7, 90: 39.1, 95: 39.5, 100: 40.0},
    37: {25: 33.1, 30: 33.9, 35: 34.6, 40: 35.3, 45: 35.9, 50: 36.4, 55: 37.0, 60: 37.5, 65: 38.0, 70: 38.4, 75: 38.9, 80: 39.3, 85: 39.7, 90: 40.2, 95: 40.6, 100: 41.0},
    38: {25: 34.0, 30: 34.8, 35: 35.6, 40: 36.2, 45: 36.9, 50: 37.4, 55: 38.0, 60: 38.5, 65: 39.0, 70: 39.5, 75: 39.9, 80: 40.4, 85: 40.8, 90: 41.2, 95: 41.6, 100: 42.0},
    39: {25: 35.0, 30: 35.8, 35: 36.5, 40: 37.2, 45: 37.8, 50: 38.4, 55: 39.0, 60: 39.5, 65: 40.0, 70: 40.5, 75: 40.9, 80: 41.4, 85: 41.8, 90: 42.3, 95: 42.7, 100: 43.1},
    40: {25: 35.9, 30: 36.7, 35: 37.5, 40: 38.2, 45: 38.8, 50: 39.4, 55: 40.0, 60: 40.5, 65: 41.0, 70: 41.5, 75: 42.0, 80: 42.4, 85: 42.8, 90: 43.3, 95: 43.7, 100: 44.1}
}

def get_wind_chill(env_temp, env_humidity):
    """ (수정됨) Key 에러 방지를 위해 .get() 사용 """
    rounded_temp = int(round(env_temp))
    if rounded_temp < 25 or rounded_temp > 40:
        return None
    temp_data = WIND_CHILL_DATA.get(rounded_temp) # .get()으로 변경
    if temp_data is None:
        return None
        
    rounded_humidity = int(round(env_humidity / 5.0) * 5)
    if rounded_humidity < 25:
        return None
    if rounded_humidity > 100:
        rounded_humidity = 100
        
    return temp_data.get(rounded_humidity) # .get()으로 변경

def get_effective_env_temp(env_temp, env_humidity):
    """ (신규 추가) 유효 환경 스트레스 온도 계산 함수 """
    if env_temp >= 25.0:
        wind_chill = get_wind_chill(env_temp, env_humidity)
        if wind_chill is not None:
            return wind_chill
        else:
            return env_temp # 표 범위 밖 고온은 기온 자체 사용
    else:
        # 저온/일반 환경 (풍속 데이터 없으므로 기온 자체 사용)
        return env_temp

def classify_risk_rules(body_temp, heart_rate, env_temp, env_humidity):
    """ (수정됨) 유효 환경 스트레스 온도를 사용하도록 변경 """
    
    # **수정**: 유효 환경 온도 계산
    effective_env_temp = get_effective_env_temp(env_temp, env_humidity)

    HIGH_ENV_TEMP = 31.0   # 고온 환경 기준 (섭씨)
    LOW_ENV_TEMP = 4.0     # 저온 환경 기준 (섭씨)
    HIGH_WRIST_TEMP_DANGER = 35.0 # 고체온 위험 기준
    LOW_WRIST_TEMP_DANGER = 33.0  # 저체온 위험 기준

    # 규칙 기반 분류
    if effective_env_temp >= HIGH_ENV_TEMP and body_temp >= HIGH_WRIST_TEMP_DANGER:
        return "열사병 위험"
    elif env_temp <= LOW_ENV_TEMP and body_temp <= LOW_WRIST_TEMP_DANGER:
        return "저체온증 위험"
    else:
      return "원인 불명 이상치"


# =============================================================================
# 파트 2: 하이브리드 예측 함수 (***핵심 수정***)
# =============================================================================
def predict_hybrid(data_list, fitted_scaler, fitted_model):
    """ (수정됨) 훈련 시와 동일한 특징 공학(Temp Diff)을 적용하여 예측 """
    results = []
    # 1. 원본 4개 데이터로 DataFrame 생성
    df_data = pd.DataFrame(data_list, columns=['body_temp', 'heart_rate', 'env_temp', 'env_humidity'])
    
    # 2. 입력 데이터 유효성 검사
    for col in df_data.columns:
        if df_data[col].isnull().any() or not np.all(np.isfinite(df_data[col])):
            print(f"[예측 오류] 입력 데이터에 Null 또는 비유한 값이 있습니다: {data_list}")
            return [f"데이터 오류 ({col})"]

    # 3. 특징 공학 (Feature Engineering) 적용 (훈련 시와 동일)
    # 3-1. 유효 환경 스트레스 온도 계산
    df_data['effective_env_temp'] = df_data.apply(
        lambda row: get_effective_env_temp(row['env_temp'], row['env_humidity']),
        axis=1
    )
    # 3-2. Temp Diff 계산
    df_data['temp_diff'] = df_data['body_temp'] - df_data['effective_env_temp']
    
    # 4. AI 모델 입력 특징 선택 (훈련 시 사용한 2개)
    X_predict_features = df_data[['heart_rate', 'temp_diff']]
    
    # 5. 스케일링 및 예측
    data_scaled = fitted_scaler.transform(X_predict_features)
    anomaly_predictions = fitted_model.predict(data_scaled)

    # 6. 결과 해석
    for i in range(len(df_data)):
        if anomaly_predictions[i] == 1:
            results.append("정상 (Normal)")
        else:
            # AI가 이상치로 판단하면, 규칙 기반 분류기(If문)가 위험 종류 판단
            row = df_data.iloc[i]
            # (주의: 규칙 분류 함수는 원본 4개 특징을 모두 사용)
            rule_result = classify_risk_rules(row['body_temp'], row['heart_rate'], row['env_temp'], row['env_humidity'])
            results.append(rule_result)
    return results

# =============================================================================
# 파트 3: 저장된 모델/스케일러 로드
# =============================================================================
try:
    scaler = joblib.load('scaler.joblib')
    model = joblib.load('model.joblib')
    print("모델과 스케일러 로드 성공.")
except FileNotFoundError:
    print("[오류] 'scaler.joblib' 또는 'model.joblib' 파일을 찾을 수 없습니다.")
    print("먼저 train_model.py를 실행하여 모델 파일을 생성하세요.")
    exit()

# =============================================================================
# 파트 4: 서버 상태 관리 (기존 코드 유지)
# =============================================================================

BEACON_TO_ENV_MAP = {
    "40011-55612": "환경 1 (고온)",
    "9999-1": "환경 2 (저온)",
    "9999-2": "환경 3 (일반)",
}
environment_state = {} 
ENV_DATA_TIMEOUT_SECONDS = 30.0 # 타임아웃 30초로 수정 (기존 값은 너무 길었음)

# =============================================================================
# 파트 5: MQTT 메시지 핸들러 (기존 코드 유지)
# =============================================================================

def handle_env_message(data):
    try:
        major = data.get('beacon_major')
        minor = data.get('beacon_minor')
        if major is None or minor is None:
            print(f" [환경 데이터 오류] 비콘 ID(Major/Minor)가 없습니다: {data}")
            return

        beacon_key = f"{major}-{minor}" 
        env_name = BEACON_TO_ENV_MAP.get(beacon_key)
        if env_name is None:
            print(f" [환경 데이터 경고] 매핑되지 않은 비콘 ID입니다: {beacon_key}")
            return

        temp = data.get('temperature')
        humi = data.get('humidity')

        if temp is None or humi is None:
            print(f" [환경 데이터 오류] 온습도 값이 없습니다: {data}")
            return
            
        environment_state[env_name] = {
            'env_temp': temp,
            'env_humi': humi,
            'last_updated': time.time()
        }
        print()
        print(f"[환경 상태 갱신] {env_name} (비콘 {beacon_key}) -> Temp: {temp}°C, Humi: {humi}%")
        
    except Exception as e:
        print(f" [오류] handle_env_message 처리 중 예외 발생: {e}")

def handle_vest_message(client, data):
    try:
        # 1. 조끼 데이터 추출
        mcu_id = data.get('mcu_id')
        body_temp = data.get('body_temp')
        hr = data.get('hr')
        is_fell = data.get('is_fell')
        scanned_beacons = data.get('beacons', [])

        if mcu_id is None:
            print(f" [조끼 데이터 오류] mcu_id가 없습니다: {data}")
            return

        print(f"\n[조끼 데이터 수신: {mcu_id}] Temp: {body_temp}°C, HR: {hr}, Fell: {is_fell}, Beacons: {len(scanned_beacons)}개")

        # 2. 위치(환경) 매핑
        if not scanned_beacons:
            print(f" └ [오류] {mcu_id}가 스캔한 비콘이 없습니다. 위치를 매핑할 수 없습니다.")
            return

        strongest_beacon = max(scanned_beacons, key=lambda b: b['rssi'])
        major = strongest_beacon.get('major')
        minor = strongest_beacon.get('minor')
        beacon_key = f"{major}-{minor}"
        
        env_name = BEACON_TO_ENV_MAP.get(beacon_key)
        
        print(f" └ [위치 매핑] 가장 강한 비콘: {beacon_key} -> {env_name if env_name else '알 수 없는 환경'}")

        if env_name is None:
            print(f" └ [오류] {mcu_id}가 스캔한 비콘({beacon_key})을 환경에 매핑할 수 없습니다.")
            return

        # 3. 환경 데이터 융합
        env_data = environment_state.get(env_name)
        if env_data is None:
            print(f" └ [오류] '{env_name}'({beacon_key})에 해당하는 환경 정보가 서버에 없습니다.")
            print(f" └ (현재 서버 상태: {list(environment_state.keys())})")
            return
        
        current_time = time.time()
        data_age = current_time - env_data.get('last_updated', 0)
        
        if data_age > ENV_DATA_TIMEOUT_SECONDS:
            print(f" └ [오류] '{env_name}'의 환경 데이터가 너무 오래되었습니다! ({data_age:.0f}초 경과)")
            print(f" └ (조끼 {mcu_id}에 대한 예측을 건너뜁니다.)")
            return
            
        env_temp = env_data['env_temp']
        env_humi = env_data['env_humi']
        
        # 4. 최종 예측 (IMU 우선)
        final_prediction = ""
        if is_fell == 1:
            final_prediction = "낙상 의심 (IMU)"
        else:
            # 5. 하이브리드 모델 예측
            if body_temp is None or hr is None or env_temp is None or env_humi is None:
                print(f" └ [오류] 예측에 필요한 데이터가 누락되었습니다 (Body/HR/Env).")
                return
                
            # **수정 없음**: predict_hybrid 함수가 내부적으로 특징 공학을 수행함
            input_data = [[body_temp, hr, env_temp, env_humi]]
            prediction_result = predict_hybrid(input_data, scaler, model)
            final_prediction = prediction_result[0]

        print(f" └ [최종 예측 결과]: >>> {final_prediction} <<<")

        # 6. 조치 수행 (경보 또는 정상 신호 전송)
        alert_topic = f"vest/alert/{mcu_id}"

        if final_prediction != "정상 (Normal)":
            client.publish(alert_topic, "ALERT")
            print(f" └ [조치] {mcu_id} 조끼로 'ALERT' ({final_prediction}) 전송 완료.")
        else:
            client.publish(alert_topic, "NORMAL")
            print(f" └ [조치] {mcu_id} 조끼로 'NORMAL' 전송 완료.")
        
    except Exception as e:
        print(f" [오류] handle_vest_message 처리 중 예외 발생: {e}")


# =============================================================================
# 파트 6: MQTT 설정 및 실행 (기존 코드 유지)
# =============================================================================

MQTT_BROKER_HOST = 'broker.hivemq.com'
MQTT_BROKER_PORT = 1883
MQTT_SUB_TOPICS = [
    ("vest/data", 0),
    ("env/data", 0)
]

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"MQTT 브로커에 연결되었습니다 (Host: {MQTT_BROKER_HOST})")
        client.subscribe(MQTT_SUB_TOPICS)
        print(f"'{MQTT_SUB_TOPICS[0][0]}' 및 '{MQTT_SUB_TOPICS[1][0]}' 토픽을 구독합니다.")
    else:
        print(f"MQTT 연결 실패 (Code: {rc})")

def on_message(client, userdata, msg):
    try:
        payload_str = msg.payload.decode('utf-8')
        data = json.loads(payload_str)
        
        if msg.topic == "env/data":
            handle_env_message(data)
            
        elif msg.topic == "vest/data":
            handle_vest_message(client, data)

    except json.JSONDecodeError:
        print(f" [오류] 수신된 데이터가 유효한 JSON 형식이 아닙니다: {payload_str}")
    except Exception as e:
        print(f" [오류] on_message 처리 중 예외 발생: {e}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
    print(f"'{MQTT_BROKER_HOST}'에 연결을 시도합니다...")
    client.loop_forever()

except KeyboardInterrupt:
    print("\n사용자에 의해 프로그램이 종료되었습니다.")
    client.disconnect()
except Exception as e:
    print(f"[오류] MQTT 연결에 실패했습니다: {e}")
    print(" (브로커 주소나 네트워크 연결을 확인하세요)")