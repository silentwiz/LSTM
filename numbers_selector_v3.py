import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
import sys
import logging
import json
import time

# 로거 기본 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# LSTM + moving-slide + Ensemble (Improved Version)

class Config:
    def __init__(self):
        ROOT_DIR = os.getcwd()
        DB_DIR = os.path.join(ROOT_DIR,'database')
        self.JP_LOTO_FILE = os.path.join(DB_DIR,'japan_loto6.txt')
        self.SEQUENCE_LENGTHS = [4, 8, 16, 32, 48, 96, 192, 384] # 4 => 최근 2주, 8 => 최근 한달, 16=> 두달,
        self.RANDOM_STATES = [20260518, 20260518, 20260518, 20260518, 20260518, 20260518, 20260518, 20260518]
        self.DROPOUT_RATES = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        self.RESULT_NUM = np.zeros(46, dtype=int)
        self.epochs = 100
        self.patience = 100
        self.ENSEMBLE_COUNT = 8
        self.MODEL_ACCURACY = []
        self.MODEL_LOSS = []
        # [추가됨] 개별 모델의 예측 결과를 저장할 리스트
        self.MODEL_PREDICTIONS = [] 

class MinimalLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"-- Epoch {epoch + 1}/{self.params['epochs']}", end='\r')
        sys.stdout.flush()

    def on_train_end(self, logs=None):
        print()
def save_results(results, config, acc_ave, loss_ave, filepath="prediction_results.json"):
    """결과를 JSON 파일로 저장"""
    try:
        individual_preds = {}
        for idx, preds in enumerate(config.MODEL_PREDICTIONS):
            individual_preds[f"model_{idx+1}_seq{config.SEQUENCE_LENGTHS[idx]}"] = preds

        # [추가됨] 순위, 번호, 출현 빈도를 보기 좋게 묶어주는 리스트 생성
        final_detailed = []
        for i, num in enumerate(results[:15]): # 상위 15개 번호 기록
            final_detailed.append({
                "rank": i + 1,
                "number": int(num),
                "frequency": int(config.RESULT_NUM[num])
            })
            
        top10_detailed = final_detailed[:10] # 상위 10개만 따로 분리

        result_data = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "config": {
                "SEQUENCE_LENGTHS": config.SEQUENCE_LENGTHS,
                "EPOCHS": config.epochs,
                "ENSEMBLE_COUNT": config.ENSEMBLE_COUNT,
                "MODEL_ACCURACY_AVG": float(acc_ave),
                "MODEL_LOSS_AVG" : float(loss_ave),
                "MODEL_ACCURACIES": [float(a) for a in config.MODEL_ACCURACY],
                "MODEL_LOSSES": [float(l) for l in config.MODEL_LOSS],
            },
            "individual_model_predictions": individual_preds,
            
            # [수정됨] 빈도수가 포함된 상세 결과로 저장
            "final_predictions_detailed": final_detailed,
            "top_10_numbers_detailed": top10_detailed,
            
            # (선택 사항) 만약 프로그램의 다른 부분에서 단순 숫자 배열만 필요할 경우를 대비해 원본도 남겨둡니다.
            "final_predictions_raw": results[:15].tolist(),
            "top_10_numbers_raw": results[:10].tolist()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        logger.info(f"結果が {filepath} に保存されました。")

    except Exception as e:
        logger.error(f"結果保存中にエラーが発生: {e}")
        
def predict_next(model, latest_data):
    input_data = np.expand_dims(latest_data, axis=0)
    prediction = model.predict(input_data, verbose=0)

    probabilities = prediction[0][1:]
    sorted_number = np.argsort(probabilities)[::-1]

    predicted_numbers = []
    for i in range(len(sorted_number)):
        number = sorted_number[i] + 1
        prob = probabilities[sorted_number[i]]
        predicted_numbers.append(number)
        print(f"順位{i+1:2d} : 番号 : {number:2d}. 確率：{prob:.2%}")

    print("\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    print(f"おすすめ番号達：{predicted_numbers[:15]}")
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    return predicted_numbers[:15]

def train_model(model, epochs, patience, X_train, X_test, Y_train, Y_test):
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
    minimal_logger = MinimalLogger()

    history = model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_test, Y_test),
        verbose=0,
        #callbacks=[early_stopping, reduce_lr, minimal_logger]
        callbacks=[reduce_lr, minimal_logger]
    )

    return model

def create_model(input_shape, dropout_rate):
    model = Sequential([
        Input(shape=input_shape),

        LSTM(256, return_sequences=True),
        BatchNormalization(),
        Dropout(dropout_rate),

        LSTM(128),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(64, activation='relu'),
        Dropout(dropout_rate * 0.5),

        Dense(46, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def moving_slide(SEQUENCE_LENGTH, numbers_df, processed_number):
    X, Y = [], []
    X_source = numbers_df.values
    for i in range(len(X_source) - SEQUENCE_LENGTH):
        X.append(X_source[i:i+SEQUENCE_LENGTH])
        Y.append(processed_number[i+SEQUENCE_LENGTH])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def preprocessing(SEQUENCE_LENGTH, df):
    ascended_df = df.sort_index(ascending=False).reset_index(drop=True)
    numbers_df = ascended_df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']]

    processed_number = []
    for index, row in numbers_df.iterrows():
        biniry_vector = np.zeros(46, dtype=int)
        for num in row:
            biniry_vector[num] = 1
        processed_number.append(biniry_vector)

    processed_number = np.array(processed_number)
    X, Y = moving_slide(SEQUENCE_LENGTH, numbers_df, processed_number)
    return X, Y

def cal_ave(some_list):
    if not some_list:
        return 0
    return sum(some_list) / len(some_list)

def main():
    print(f"TENSOR-FLOW Version: [{tf.__version__}]")
    
    gpu_devices = tf.config.list_physical_devices('GPU')
    mac_gpu = tf.config.list_physical_devices('macOS')
    if gpu_devices or mac_gpu:
        print("✅ Hardware Acceleration (GPU/MPS) is AVAILABLE")
    else:
        print("❌ Hardware Acceleration NOT AVAILABLE (Running on CPU only)")

    config = Config()
    df = pd.read_csv(config.JP_LOTO_FILE, sep=r'\s+')

    for i in range(config.ENSEMBLE_COUNT):
        print(f"\n")
        logger.info(f"[ENSEMBLE MODEL {i+1}/{config.ENSEMBLE_COUNT}] START TRAIN...")
        logger.info(f"  - SEQUENCE_LENGTH: {config.SEQUENCE_LENGTHS[i]}")
        logger.info(f"  - DROPOUT_RATE: {config.DROPOUT_RATES[i]}")
        logger.info(f"  - RANDOM_SEED: {config.RANDOM_STATES[i]}")
        print(f"\n")

        tf.random.set_seed(config.RANDOM_STATES[i])

        sequence_length = config.SEQUENCE_LENGTHS[i]
        sequence_for_predict = df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].sort_index(ascending=False).reset_index(drop=True)
        latest_sequence = sequence_for_predict.tail(sequence_length).values
        
        X, Y = preprocessing(sequence_length, df)

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        Y_train, Y_test = Y[:split_idx], Y[split_idx:]

        dropout_rate = config.DROPOUT_RATES[i]
        model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]), dropout_rate=dropout_rate)

        trained_model = train_model(model, config.epochs, config.patience, X_train, X_test, Y_train, Y_test)
        loss, accuracy = trained_model.evaluate(X_test, Y_test, verbose=0)

        logger.info(f"  - Model {i+1} Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

        result = predict_next(trained_model, latest_sequence)
        config.MODEL_ACCURACY.append(accuracy)
        config.MODEL_LOSS.append(loss)
        
        # [추가됨] Numpy array인 int64를 순수 Python int로 변환하여 JSON 직렬화 에러 방지
        config.MODEL_PREDICTIONS.append([int(num) for num in result])
        
        for num in result:
            config.RESULT_NUM[num] += 1

    result = np.argsort(config.RESULT_NUM)[::-1]

    print(f"\n\n{'='*100}")
    print(f"【最終結果 - アンサンブル投票結果】")
    print(f"{'='*100}\n")

    for i in range(15):
        number = result[i]
        frequency = config.RESULT_NUM[number]
        print(f'{i+1:2d}位　：　番号[{number:2d}], 出現頻度[{frequency}]')

    print(f"\n{'='*100}")
    print(f'\n最終おすすめ番号達（TOP 10）')
    print(f"{result[:10]}")
    print(f"\n{'='*100}")

    acc_ave = cal_ave(config.MODEL_ACCURACY)
    loss_ave = cal_ave(config.MODEL_LOSS)

    logger.info(f"平均 Accuracy: {acc_ave:.4f}")
    logger.info(f"平均 Loss: {loss_ave:.4f}")

    save_results(result, config, acc_ave, loss_ave, filepath="loto6_ver2.json")

if __name__ == '__main__':
    main()