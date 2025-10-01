import pandas as pd
import numpy as np
import os
import json
import logging
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, Bidirectional,
                                   BatchNormalization, MultiHeadAttention,
                                   LayerNormalization, GlobalAveragePooling1D, Add)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import sys

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    """설정 관리 클래스"""
    def __init__(self):
        self.SEQUENCE_LENGTH = 20
        self.EPOCHS = 100
        self.BATCH_SIZE = 32
        self.DROPOUT_RATE = 0.2
        self.LEARNING_RATE = 0.001
        self.ENSEMBLE_COUNT = 50
        self.TEST_SIZE = 0.2
        self.PATIENCE = 50
        self.MIN_DELTA = 0.001

class MinimalLogger(tf.keras.callbacks.Callback):
    """최소한의 로그 출력 콜백"""
    def on_epoch_end(self, epoch, logs=None):
        print(f"-- Epoch {epoch + 1}/{self.params['epochs']} - loss: {logs['loss']:.4f} - val_loss: {logs['val_loss']:.4f}", end='\r')
        sys.stdout.flush()

    def on_train_end(self, logs=None):
        print()

def create_improved_model(input_shape, config):
    """개선된 LSTM 모델 생성"""
    try:
        inputs = Input(shape=input_shape)

        # Bidirectional LSTM with BatchNorm
        x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(config.DROPOUT_RATE)(x)

        # Second Bidirectional LSTM layer
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = BatchNormalization()(x)
        x = Dropout(config.DROPOUT_RATE)(x)

        # Attention mechanism
        attention_output = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = Add()([x, attention_output])  # Residual connection
        x = LayerNormalization()(x)

        # Global pooling
        x = GlobalAveragePooling1D()(x)

        # Dense layers with residual connections
        dense1 = Dense(128, activation='swish')(x)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(config.DROPOUT_RATE * 1.5)(dense1)

        dense2 = Dense(64, activation='swish')(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(config.DROPOUT_RATE)(dense2)

        # Output layer
        outputs = Dense(46, activation='sigmoid')(dense2)

        model = Model(inputs=inputs, outputs=outputs)

        # 모델 컴파일 with custom learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='binary_crossentropy',
                     metrics=['accuracy', 'precision', 'recall'])

        return model

    except Exception as e:
        logger.error(f"모델 생성 중 오류 발생: {e}")
        raise

def create_time_aware_split(X, Y, test_size=0.2):
    """시계열 데이터에 적합한 분할 방식"""
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    return X_train, X_test, Y_train, Y_test

def train_model_with_callbacks(model, config, X_train, X_test, Y_train, Y_test):
    """콜백과 함께 모델 훈련"""
    try:
        # 콜백 설정
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=config.PATIENCE,
            min_delta=config.MIN_DELTA,
            restore_best_weights=True,
            verbose=0
        )

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=0
        )

        minimal_logger = MinimalLogger()

        callbacks = [early_stopping, lr_scheduler, minimal_logger]

        history = model.fit(
            X_train, Y_train,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            validation_data=(X_test, Y_test),
            callbacks=callbacks,
            verbose=0
        )

        return model, history

    except Exception as e:
        logger.error(f"모델 훈련 중 오류 발생: {e}")
        raise

def predict_with_confidence(model, latest_data, top_n=15):
    """신뢰도를 포함한 예측"""
    try:
        input_data = np.expand_dims(latest_data, axis=0)
        prediction = model.predict(input_data, verbose=0)

        # 0번 인덱스 제거 (숫자 0은 로또에서 사용되지 않음)
        probabilities = prediction[0][1:]

        # 확률 기준 정렬
        sorted_indices = np.argsort(probabilities)[::-1]

        predicted_numbers = []
        confidences = []

        for i in range(min(top_n, len(sorted_indices))):
            number = sorted_indices[i] + 1
            confidence = probabilities[sorted_indices[i]]
            predicted_numbers.append(number)
            confidences.append(confidence)

        return predicted_numbers, confidences

    except Exception as e:
        logger.error(f"예측 중 오류 발생: {e}")
        raise

def weighted_ensemble_prediction(models, model_scores, latest_sequences, top_n=10):
    """가중 앙상블 예측"""
    try:
        ensemble_scores = np.zeros(46)
        total_weight = sum(model_scores)

        # total_weight가 0이거나 매우 작은 경우 처리
        if total_weight <= 1e-10:
            logger.warning("모든 모델 점수가 0에 가까움. 균등 가중치 사용.")
            # 균등 가중치 사용
            weights = [1.0 / len(models)] * len(models)
        else:
            weights = [score / total_weight for score in model_scores]

        for i, (model, sequence, weight) in enumerate(zip(models, latest_sequences, weights)):
            predictions, _ = predict_with_confidence(model, sequence, 45)

            for rank, number in enumerate(predictions):
                # 순위에 따른 점수 부여 (상위일수록 높은 점수)
                rank_score = (len(predictions) - rank) / len(predictions)
                ensemble_scores[number] += weight * rank_score

        # 최종 순위 결정
        final_ranking = np.argsort(ensemble_scores)[::-1]
        return final_ranking[:top_n], ensemble_scores

    except Exception as e:
        logger.error(f"앙상블 예측 중 오류 발생: {e}")
        raise

def preprocessing_with_validation(sequence_length, df):
    """검증과 함께 데이터 전처리"""
    try:
        if df.empty:
            raise ValueError("데이터프레임이 비어있습니다.")

        # 데이터 정렬 및 초기화
        ascended_df = df.sort_index(ascending=False).reset_index(drop=True)
        numbers_df = ascended_df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']]

        # 이진 벡터 변환
        processed_numbers = []
        for _, row in numbers_df.iterrows():
            binary_vector = np.zeros(46, dtype=int)
            for num in row:
                if 1 <= num <= 45:  # 유효한 숫자 범위 검증
                    binary_vector[num] = 1
            processed_numbers.append(binary_vector)

        processed_numbers = np.array(processed_numbers)

        # 슬라이딩 윈도우 생성
        X, Y = [], []
        X_source = numbers_df.values

        for i in range(len(X_source) - sequence_length):
            X.append(X_source[i:i+sequence_length])
            Y.append(processed_numbers[i+sequence_length])

        return np.array(X), np.array(Y)

    except Exception as e:
        logger.error(f"데이터 전처리 중 오류 발생: {e}")
        raise

def save_results(results, model_performances, config, filepath="prediction_results.json"):
    """결과를 JSON 파일로 저장"""
    try:
        result_data = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "config": {
                "sequence_length": config.SEQUENCE_LENGTH,
                "epochs": config.EPOCHS,
                "ensemble_count": config.ENSEMBLE_COUNT
            },
            "model_performances": model_performances,
            "final_predictions": results.tolist(),
            "top_10_numbers": results[:10].tolist()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        logger.info(f"결과가 {filepath}에 저장되었습니다.")

    except Exception as e:
        logger.error(f"결과 저장 중 오류 발생: {e}")

def main():
    try:
        logger.info(f"TensorFlow 버전: {tf.__version__}")

        # 설정 초기화
        config = Config()

        # 파일 경로 설정
        ROOT_DIR = os.getcwd()
        DB_DIR = os.path.join(ROOT_DIR, 'database')
        JP_LOTO_FILE = os.path.join(DB_DIR, 'japan_loto6.txt')

        if not os.path.exists(JP_LOTO_FILE):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {JP_LOTO_FILE}")

        # 데이터 로드
        logger.info(f"데이터 로드: {JP_LOTO_FILE}")
        df = pd.read_csv(JP_LOTO_FILE, sep=r'\s+')

        # 앙상블을 위한 시퀀스 길이 설정
        sequence_lengths = [config.SEQUENCE_LENGTH + (3 * i) for i in range(config.ENSEMBLE_COUNT)]
        logger.info(f"시퀀스 길이: {sequence_lengths}")

        models = []
        model_scores = []
        latest_sequences = []
        model_performances = []

        # 각 시퀀스 길이별로 모델 훈련
        for seq_len in sequence_lengths:
            logger.info(f"\n시퀀스 길이 [{seq_len}]으로 모델 훈련 시작")

            # 예측용 최신 시퀀스 준비
            sequence_for_predict = df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].sort_index(ascending=False).reset_index(drop=True)
            latest_sequence = sequence_for_predict.tail(seq_len).values

            # 데이터 전처리
            X, Y = preprocessing_with_validation(seq_len, df)

            # 시계열 데이터 분할
            X_train, X_test, Y_train, Y_test = create_time_aware_split(X, Y, config.TEST_SIZE)

            # 모델 생성
            model = create_improved_model(input_shape=(X_train.shape[1], X_train.shape[2]), config=config)

            # 모델 훈련
            trained_model, history = train_model_with_callbacks(
                model, config, X_train, X_test, Y_train, Y_test
            )

            # 모델 평가
            test_loss, test_accuracy, test_precision, test_recall = trained_model.evaluate(X_test, Y_test, verbose=0)
            f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0

            logger.info(f"테스트 성능 - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, F1: {f1_score:.4f}")

            # 결과 저장
            models.append(trained_model)
            model_scores.append(f1_score)  # F1 스코어를 가중치로 사용
            latest_sequences.append(latest_sequence)
            model_performances.append({
                "sequence_length": seq_len,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "f1_score": f1_score
            })

        # 가중 앙상블 예측
        logger.info("\n가중 앙상블 예측 수행")
        final_predictions, ensemble_scores = weighted_ensemble_prediction(
            models, model_scores, latest_sequences, top_n=20
        )

        # 결과 출력
        print("\n" + "="*80)
        print("최종 예측 결과 (가중 앙상블)")
        print("="*80)

        for i in range(20):
            number = final_predictions[i]
            score = ensemble_scores[number]
            print(f"{i+1:2d}위 : 번호[{number:2d}], 앙상블 점수[{score:.4f}]")

        print("\n" + "="*80)
        print(f"최종 추천 번호 (상위 10개): {final_predictions[:10]}")
        print("="*80)

        # 결과 저장
        save_results(final_predictions, model_performances, config)

        return final_predictions[:10]

    except Exception as e:
        logger.error(f"메인 실행 중 오류 발생: {e}")
        raise

if __name__ == '__main__':
    main()