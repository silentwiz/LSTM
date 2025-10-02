# LSTM
ex for LSTM or mechine learning

## commit 2
### !!!중요!!! "solved : latest_sequence = sequence_for_predict.tail(e_range).values"
> e_range가 아니라, config.SEQUENCE_LENGTHS를 사용하고 있던 버그 
>>모델 학습에 중대한 영향을 미췄을 가능성 높음

## commit 3
### !!!중요!!! "DELETED : random state" 
> 모델 학습시에 train_test_split을 사용하여, "시간적 정보가 중요한" 이모델에서, 의도적으로 시간적 배열을 가지도록 생성된 학습 데이터를 randome state로 섞어버림 
>>모델 학습에 중대한 영향을 미쳤을 가능성 높음