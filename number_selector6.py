import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import sys
import logging
import json
import time
# 로거 기본 설정
logging.basicConfig(
    level=logging.INFO,  # INFO 이상 메시지를 출력
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# LSTM + moving-slide + Ensemble

class Config:
    def __init__(self):
        ROOT_DIR = os.getcwd()
        DB_DIR = os.path.join(ROOT_DIR,'database')
        self.JP_LOTO_FILE = os.path.join(DB_DIR,'japan_loto6.txt')
        self.SEQUENCE_LENGTH = 30 ## default = 35? 2037 ~ 2004회에 걸쳐서 07이 반복되서 보이는 경향
        self.SEQUENCE_LENGTHS = []
        self.SEQUENCE_LENGTH_COUNT = 10
        self.SEQUENCE_LENGTH_RANGE = 5
        self.SEQUENCE_LENGTH_VALUE = 3
        self.RESULT_NUM = np.zeros(44, dtype=int)
        self.epochs = 100
        self.patience = 60
        self.ENSEMBLE_COUNT = 5
        self.MODEL_ACCURACY = []
        self.MODEL_LOSS = []
        self.REST_POINT = 0.3
        self.VALUE_RANDOME_STATE = [7,15,777]


class MinimalLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # \r은 커서를 줄의 맨 앞으로 이동시켜 줍니다 (덮어쓰기 효과).
        # epoch 변수는 0부터 시작하므로, 사용자에게 보여줄 때는 +1을 합니다.
        print(f"-- Epoch {epoch + 1}/{self.params['epochs']}", end='\r')
        # 버퍼를 비워 출력이 즉시 업데이트되도록 합니다.
        sys.stdout.flush()

    def on_train_end(self, logs=None):
        # 훈련이 모두 끝난 후, 줄바꿈을 추가하여 터미널 줄이 깔끔하게 정리되도록 합니다.
        print()

def save_results(results, config,acc_ave,loss_ave, filepath="prediction_results.json"):
    """결과를 JSON 파일로 저장"""
    try:
        result_data = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "config": {
                "SEQUENCE_LENGTH": config.SEQUENCE_LENGTH,
                "EPOCHS": config.epochs,
                "ENSEMBLE_COUNT": config.ENSEMBLE_COUNT,
                "MODEL_ACCURACY": acc_ave,
                "MODEL_LOSS" : loss_ave,
            },
            "final_predictions": results.tolist(),
            "top_10_numbers": results[:10].tolist()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
            #json.dump(result_data, f, separators=(", ", ": "))
        logger.info(f"결과가 {filepath}에 저장되었습니다.")

    except Exception as e:
        logger.error(f"결과 저장 중 오류 발생: {e}")
        

def predict_next(model, latest_data):
    # 1. 모델 입력 형태로 차원 확장: (35, 6) -> (1, 35, 6)
    # 모델은 한 번에 여러 개(배치)의 데이터를 처리하도록 설계되었기 때문입니다.
    # 행은 axis0, 열은 axis1
    input_data = np.expand_dims(latest_data, axis=0)
    prediction = model.predict(input_data, verbose=0)

    #print(f"before : [{prediction}]")

    # 2. 예측 숫자 가공 바이너리벡터[0,1,0,0,0,1,0,0...]의 인덱스=>[0,1,2,3,4,5 ...]에서, 0번쩨 인덱스 (숫자0에 대응하는인덱스를 삭제)를 삭제
    probabilities = prediction[0][1:]

    # 4. 확률이 높은 순서대로 인덱스를 정렬
    # [::-1]을 붙여 내림차순(높은 것부터)으로 정렬합니다.
    # argsort는 인덱스(숫자)를 반환
    
    sorted_number = np.argsort(probabilities)[::-1]

    #print(" 次回の予測（上位１５位）")
    predicted_numbers = []
    for i in range(len(sorted_number)):
        number = sorted_number[i] + 1 # 인덱스는 0부터 시작하기에 +1을 해야 숫자가 표시된다.
        prob = probabilities[sorted_number[i]]
        predicted_numbers.append(number)
        print(f"順位{ i+1} : 番号 : {number}. 確率：{prob:.2%}")

    winner_numbers = []
    for num in predicted_numbers[:15]:
        winner_numbers.append(num)
    print("\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    print(f"おすすめ番号達：{predicted_numbers[:15]}")
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    return predicted_numbers[:15]


def train_model(model, epochs,patience, X_train, X_test, Y_train, Y_test):
    #print(f"\nstart model train")
    # 조기 종료(EarlyStopping) 콜백 설정: 10번 동안 성능 향상이 없으면 학습 중단
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)

    minimal_logger = MinimalLogger()

    history = model.fit(
        X_train, 
        Y_train, 
        epochs=epochs,           # 반복 학습
        batch_size=32,        # 한 번에 32개씩 데이터를 묶어 학습
        validation_data=(X_test, Y_test), # 각 epoch마다 테스트 데이터로 성능 검증
        verbose=0,
        callbacks=[early_stopping, minimal_logger]        # 조기 종료 기능 적용
        #callbacks=[minimal_logger]
    )

    return model




def create_model(input_shape, rest_point):
    model = Sequential([

        # UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.super().__init__(**kwargs)
        Input(shape=input_shape),
        
        # 입력층: LSTM 레이어. input_shape는 (시퀀스 길이, 피처 개수)
        # return_sequences=True는 다음 LSTM 레이어가 있기 때문에 추가
        LSTM(256, return_sequences=True),
        Dropout(rest_point),

        # 은닉층: LSTM 레이어
        LSTM(128),
        Dropout(rest_point),

        # 은닉층: 완전 연결 레이어
        Dense(64, activation='relu'),

        #출력층: 46개의 출력을 가집니다 (0~45번 숫자).
        # sigmoid 활성화 함수는 각 숫자가 나올 확률을 0과 1 사이로 예측
        # (다중 레이블 분류 문제이므로 sigmoid 사용)
        Dense(46, activation='sigmoid')
    ])

    # 2. 모델 컴파일
    # loss 함수로 'binary_crossentropy'를 사용합니다. 
    # 46개 각각에 대해 '나온다/안나온다'를 맞추는 이진 분류 문제의 합이기 때문입니다.
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model



def moving_slide(SEQUENCE_LENGTH, numbers_df, processed_number):
    '''
    일기예보를 예로들어서, 

    [월,화,수] => [목]의 날씨 예측,
    [화,수,목] => [금]의 날씨 예측,
    [수,목,금] => [토]의 날씨 예측
    '''
    X, Y = [], []
    X_source = numbers_df.values # 이진벡터로 학습시킬경우, 값이 너무 희소(sparse)해져서모델의 페턴학습이 어렵다.
    for i in range(len(X_source) - SEQUENCE_LENGTH):
        # 훈련 데이터
        X.append(X_source[i:i+SEQUENCE_LENGTH])
        # 정답 데이터
        Y.append(processed_number[i+SEQUENCE_LENGTH])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def preprocessing(SEQUENCE_LENGTH, df):
    ascended_df = df.sort_index(ascending=False)
    ascended_df = ascended_df.reset_index(drop=True) 
    numbers_df = ascended_df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']]
    #print(f"numbers_df :\n\n{numbers_df.head()}\n\n")

    processed_number = []
    for index, row in numbers_df.iterrows():
        biniry_vector = np.zeros(46, dtype=int)
        for num in row:
            biniry_vector[num] = 1
        processed_number.append(biniry_vector)
    #print(f'processed_number[2] :\n length : {len(processed_number[2])}\n{processed_number[2]}')

    processed_number = np.array(processed_number)
    
    X, Y = moving_slide(SEQUENCE_LENGTH, numbers_df, processed_number)
    return X, Y

def cal_ave(some_list):
    a = 0
    for value in some_list:
        a += value
    result = a / len(some_list)
    return result

def main():
    print(f"TENSOR-FLOW are loaded : Version[{tf.__version__}]")
    gpu = len(tf.config.list_physical_devices('GPU'))>0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")

    config = Config()
    #df = pd.read_csv(database, sep='\s+')
    df = pd.read_csv(config.JP_LOTO_FILE, sep=r'\s+') # if occured SyntaxWarning: invalid escape sequence '\s'
    for i in range(config.SEQUENCE_LENGTH_COUNT):
        config.SEQUENCE_LENGTHS.append(config.SEQUENCE_LENGTH+(config.SEQUENCE_LENGTH_VALUE*i))

    logger.info(f"CREATED SEQUENCE_LENGTHS : {config.SEQUENCE_LENGTHS}")
    for e_range in config.SEQUENCE_LENGTHS:
        for e_counter in range(config.ENSEMBLE_COUNT):
            print(f"\n")
            #logger.info(f"\nSEQUENCE LENGTH = {e_range} \n[ENSEMBLE MODEL {e_counter+1}/{config.ENSEMBLE_COUNT}] START TRAIN...")
            print(f"\n")

            # 1.preprocessing
            sequence_for_predict = df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].sort_index(ascending=False).reset_index(drop=True) 
            latest_sequence = sequence_for_predict.tail(e_range).values
            X, Y = preprocessing(e_range,df)

            # 2.train model
        
            logger.info(f"\n[SEQUENCE LENGTH : {e_range} in {config.SEQUENCE_LENGTHS}]\n[ENSEMBLE COUNTER : {e_counter+1}/{config.ENSEMBLE_COUNT}]\n[VALUE_RANDOME_STATE : {r_s} in {config.VALUE_RANDOME_STATE}]\n")

            # 시간 순서 유지 분할
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            Y_train, Y_test = Y[:split_idx], Y[split_idx:]

            model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]), rest_point=config.REST_POINT)
            #model.summary()
            trained_model = train_model(model, config.epochs, config.patience, X_train, X_test, Y_train, Y_test)
            loss, accuracy = trained_model.evaluate(X_test, Y_test, verbose=0)
            
            # 3.predict number
            result = predict_next(trained_model, latest_sequence)
            config.MODEL_ACCURACY.append(accuracy)
            config.MODEL_LOSS.append(loss)
            for num in result:
                config.RESULT_NUM[num] += 1

        result = np.argsort(config.RESULT_NUM)[::-1]
        
        for i in range(15):
            number = result[i]
            frequency = config.RESULT_NUM[number]
            print(f'{i+1}位　：　番号[{number}], 出現頻度[{frequency}]')

    print(f"\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    print(f'\n最終おすすめ番号達')
    print(f"{result[:10]}")
    print(f"\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    acc_ave = cal_ave(config.MODEL_ACCURACY)
    loss_ave = cal_ave(config.MODEL_LOSS)
    save_results(result, config, acc_ave, loss_ave,filepath="loto6.json")

if __name__ == '__main__':
    main()

#おすすめ番号達：[25, 43, 20, 12, 34, 29, 10, 19, 7, 14, 37, 33, 2, 3, 23]
#おすすめ番号達：[2, 32, 20, 40, 29, 23, 8, 24, 35, 7, 37, 13, 34, 31, 33]
#おすすめ番号達：[17, 18, 7, 29, 20, 23, 27, 33, 36, 43, 42, 8, 25, 26, 31]