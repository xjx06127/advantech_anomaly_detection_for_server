# mqtt_listener.py

import paho.mqtt.client as mqtt
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

print("--- 하이브리드 위험 예측 리스너 (MQTT) ---")
print("모델과 스케일러를 로드합니다...")

# =============================================================================
# 파트 1: 규칙 기반 모델 (분류기) - (기존 코드 복사)
# =============================================================================
# (예측에 필요하므로 기존 코드를 그대로 가져옵니다)

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
    rounded_temp = int(round(env_temp))
    if rounded_temp < 25 or rounded_temp > 40:
        return None
    temp_data = WIND_CHILL_DATA[rounded_temp]
    rounded_humidity = int(round(env_humidity / 5.0) * 5)
    if rounded_humidity < 25:
        return None
    if rounded_humidity > 100:
        return None
    if rounded_humidity in temp_data:
        return temp_data[rounded_humidity]
    else:
        return None

def classify_risk_rules(body_temp, heart_rate, env_temp, env_humidity):
    wind_chill = get_wind_chill(env_temp, env_humidity)
    if wind_chill is None:
        effective_env_temp = env_temp
    else:
        effective_env_temp = wind_chill
    HIGH_ENV_TEMP = 31.0
    LOW_ENV_TEMP = 4.0
    HIGH_WRIST_TEMP_DANGER = 35.0
    LOW_WRIST_TEMP_DANGER = 33.0
    if effective_env_temp >= HIGH_ENV_TEMP and body_temp >= HIGH_WRIST_TEMP_DANGER:
        return "열사병 위험"
    elif env_temp <= LOW_ENV_TEMP and body_temp <= LOW_WRIST_TEMP_DANGER:
        return "저체온증 위험"
    elif body_temp >= HIGH_WRIST_TEMP_DANGER:
        return "일반환경 고열"
    elif body_temp <= LOW_WRIST_TEMP_DANGER:
        return "일반환경 저열"
    elif effective_env_temp >= HIGH_ENV_TEMP:
        return "고온환경 일반온도"
    elif env_temp <= LOW_ENV_TEMP:
        return "저온환경 일반온도"
    else:
        return "원인 불명 이상치"

# =============================================================================
# 파트 2: 하이브리드 예측 함수 - (기존 코드 복사)
# =============================================================================
# (예측에 필요하므로 기존 코드를 그대로 가져옵니다)

def predict_hybrid(data_list, fitted_scaler, fitted_model):
    results = []
    df_data = pd.DataFrame(data_list, columns=['body_temp', 'heart_rate', 'env_temp', 'env_humidity'])
    data_scaled = fitted_scaler.transform(df_data)
    anomaly_predictions = fitted_model.predict(data_scaled)

    for i in range(len(df_data)):
        if anomaly_predictions[i] == 1:
            results.append("정상 (Normal)")
        else:
            row = df_data.iloc[i]
            rule_result = classify_risk_rules(row['body_temp'], row['heart_rate'], row['env_temp'], row['env_humidity'])
            results.append(rule_result)
    return results

# =============================================================================
# 파트 3: 저장된 모델/스케일러 로드
# =============================================================================
try:
    # 1단계에서 저장한 파일들을 불러옵니다.
    scaler = joblib.load('scaler.joblib')
    model = joblib.load('model.joblib')
    print("모델과 스케일러 로드 성공.")
except FileNotFoundError:
    print("[오류] 'scaler.joblib' 또는 'model.joblib' 파일을 찾을 수 없습니다.")
    print("먼저 train_model.py를 실행하여 모델 파일을 생성하세요.")
    exit()

# =============================================================================
# 파트 4: MQTT 설정 및 실행
# =============================================================================

# --- MQTT 설정 (여기만 수정하세요) ---

# 1. MQTT 브로커 주소
# 가장 쉬운 방법: public MQTT 브로커 사용 (예: 'broker.hivemq.com')
# 또는 로컬에 Mosquitto 등을 설치했다면 'localhost' 또는 '127.0.0.1'
MQTT_BROKER_HOST = 'broker.hivemq.com'
MQTT_BROKER_PORT = 1883

# 2. ESP32가 발행(Publish)할 토픽 이름
# ESP32 코드에 설정된 토픽과 정확히 일치해야 합니다.
MQTT_TOPIC = "esp32/sensor_data"

# ------------------------------------

# MQTT 클라이언트가 브로커에 연결되었을 때 호출될 함수
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"MQTT 브로커에 연결되었습니다 (Host: {MQTT_BROKER_HOST})")
        # 연결 성공 시, ESP32가 보낼 토픽을 구독(subscribe)
        # f라는 키워드를 통해 fstring사용. 즉 중괄호 부분을 코드로 인식
        client.subscribe(MQTT_TOPIC)
        print(f"'{MQTT_TOPIC}' 토픽을 구독합니다.")
    else:
        print(f"MQTT 연결 실패 (Code: {rc})")

# MQTT 메시지(데이터)를 수신했을 때 호출될 함수
def on_message(client, userdata, msg):
    print(f"\n[{msg.topic}] 메시지 수신:")
    
    try:
        # 1. 메시지(payload)를 문자열로 디코딩 및 JSON 파싱
        payload_str = msg.payload.decode('utf-8')
        data = json.loads(payload_str)
        
        # 2. 데이터 추출
        mcu_id = data.get('mcu_id', 'UNKNOWN_MCU')
        body_temp = data.get('body_temp', 'N/A')
        hr = data.get('hr', 'N/A')
        env_temp = data.get('env_temp', 'N/A')
        humidity = data.get('humidity', 'N/A')
        beacons = data.get('beacons', [])
        beacon_count = len(beacons)

        # 3. "한 줄 요약 로그" 출력
        print(f"[MCU: {mcu_id}] Temp: {body_temp}°C, HR: {hr}, Env: {env_temp}°C/{humidity}%, Beacons: {beacon_count}")

        # 4. "상세 비콘 목록" 출력 (들여쓰기 적용)
        if beacon_count > 0:
            print("  └ [비콘 상세]: ", end="")
            beacon_details = []
            for beacon in beacons:
                beacon_details.append(
                    f"Major/Minor: {beacon.get('major', 'N/A')}/{beacon.get('minor', 'N/A')} (RSSI: {beacon.get('rssi', 'N/A')} dBm)"
                )
            # 쉼표로 연결하여 한 줄 또는 여러 줄로 출력
            print(", ".join(beacon_details))

        input_data = [[body_temp, hr, env_temp, humidity]]

        # 하이브리드 모델로 예측 실행
        prediction_result = predict_hybrid(input_data, scaler, model)
            
        # 7. 예측 결과 출력 (들여쓰기 추가)
        print(f"  └ [예측 결과]: >>> {prediction_result[0]}")

    except json.JSONDecodeError:
        print(f"  [오류] 수신된 데이터가 유효한 JSON 형식이 아닙니다: {payload_str}")
    except Exception as e:
        print(f"  [오류] 데이터 처리 중 예외 발생: {e}")

# --- MQTT 클라이언트 실행 ---
client = mqtt.Client()
client.on_connect = on_connect  # 연결 콜백 함수 지정
client.on_message = on_message  # 메시지 수신 콜백 함수 지정

try:
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
    # 60초 동안 메세지 안오면 연결 확인 상태 메세지 보냄
    # loop_forever(): 스크립트가 종료되지 않고 계속 실행되며 MQTT 메시지를 기다림
    print(f"'{MQTT_BROKER_HOST}'에 연결을 시도합니다...")
    client.loop_forever()

except KeyboardInterrupt:
    print("\n사용자에 의해 프로그램이 종료되었습니다.")
    client.disconnect()
except Exception as e:
    print(f"[오류] MQTT 연결에 실패했습니다: {e}")
    print(" (브로커 주소나 네트워크 연결을 확인하세요)")