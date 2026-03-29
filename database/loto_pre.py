import pandas as pd

def convert_jpy_to_num(text):
    if pd.isna(text) or str(text).strip() == '該当なし':
        return 0 
    
    text = str(text).replace('！', '').strip()
    text = text.replace('千万', '000万') 
    
    if '億' in text and not text.endswith('万') and not text.endswith('億'):
        if text[-1].isdigit():
            text += '万'
            
    total = 0
    text_copy = text
    
    if '億' in text_copy:
        parts = text_copy.split('億')
        if parts[0].isdigit():
            total += int(parts[0]) * 100000000
        text_copy = parts[1] 
        
    if '万' in text_copy:
        parts = text_copy.split('万')
        if parts[0].isdigit():
            total += int(parts[0]) * 10000
            
    if text_copy.isdigit() and total == 0:
        total += int(text_copy)
        
    return total

if __name__ == "__main__":
    # 1. 텍스트 파일 불러오기 (처음 올려주신 데이터를 보면 탭 또는 공백으로 구분된 것으로 보입니다)
    # 만약 에러가 난다면 sep='\t' 대신 sep='\s+' (연속된 공백)을 사용해 보세요.
    input_file = 'japan_loto6.txt'
    print(f"[{input_file}] 파일을 읽어오는 중...")
    df = pd.read_csv(input_file, sep='\s+')

    # 2. 특정 컬럼에 함수 적용하여 덮어쓰기 (새 컬럼을 만들지 않고 기존 컬럼을 수치형으로 변경)
    df['price'] = df['price'].apply(convert_jpy_to_num)
    df['carryover'] = df['carryover'].apply(convert_jpy_to_num)

    # 3. 데이터 확인용 상위 5개 출력
    print("\n--- 전처리 결과 미리보기 ---")
    print(df[['round', 'price', 'carryover']].head())

    # 4. LSTM 모델 학습을 위해 CSV 파일로 저장
    output_file = 'japan_loto6_preprocessed.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n전처리 완료! [{output_file}] 파일이 생성되었습니다.")
