import streamlit as st
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os
from datetime import datetime

st.title("다중 회귀 모델 비교 분석 (다항식/다중다항식/신경망)")

# 세션 상태 초기화
if "models" not in st.session_state:
    st.session_state.models = {
        'polynomial': None,
        'multipolynomial': None,
        'neural_network': None
    }
if "trained" not in st.session_state:
    st.session_state.trained = False
if "input_cols" not in st.session_state:
    st.session_state.input_cols = []
if "output_cols" not in st.session_state:
    st.session_state.output_cols = []
if "split_ratios" not in st.session_state:
    st.session_state.split_ratios = (0.7, 0.2, 0.1)
if "scaler_X" not in st.session_state:
    st.session_state.scaler_X = None
if "scaler_y" not in st.session_state:
    st.session_state.scaler_y = None
if "history" not in st.session_state:
    st.session_state.history = None
if "model_performance" not in st.session_state:
    st.session_state.model_performance = {
        'forward': None,
        'reverse': None
    }

# Step 1. 데이터 업로드
st.subheader("1. 데이터 업로드")
upload_method = st.radio("데이터 업로드 방법 선택", ["엑셀 복사/붙여넣기", "CSV 파일 업로드"])

if upload_method == "엑셀 복사/붙여넣기":
    data_text = st.text_area("엑셀에서 복사한 데이터를 여기에 붙여넣어 주세요.", height=300)
    if data_text:
        try:
            data = pd.read_csv(io.StringIO(data_text), sep="\t")
        except Exception as e:
            st.error(f"데이터 변환 중 에러 발생: {e}")
            st.stop()
else:
    uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=['csv'])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"파일 읽기 중 에러 발생: {e}")
            st.stop()

if 'data' in locals():
    # 결측치 처리
    if data.isnull().sum().sum() > 0:
        st.warning("데이터에 결측치가 있습니다. 평균값으로 대체합니다.")
        data = data.fillna(data.mean())
    
    st.success("데이터 변환 성공!")
    st.dataframe(data)

    # 데이터 시각화
    st.subheader("데이터 분포 확인")
    viz_col = st.selectbox("시각화할 변수 선택", options=data.columns)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(data=data, x=viz_col, kde=True, ax=ax)
    st.pyplot(fig)

    # Step 2. 입력/출력 설정
    st.subheader("2. 모델 입출력 설정")
    columns = data.columns.tolist()
    
    # 입력 변수 선택 UI 개선
    st.markdown("**입력 변수(Features) 선택**")
    input_cols = []
    for col in columns:
        if st.checkbox(f"{col} (min: {data[col].min():.2f}, max: {data[col].max():.2f}, avg: {data[col].mean():.2f})", key=f"input_checkbox_{col}"):
            input_cols.append(col)
    
    # 출력 변수 선택 UI 개선
    st.markdown("**출력 변수(Targets) 선택**")
    output_cols = []
    for col in columns:
        if st.checkbox(f"{col} (min: {data[col].min():.2f}, max: {data[col].max():.2f}, avg: {data[col].mean():.2f})", key=f"output_checkbox_{col}"):
            output_cols.append(col)

    if not input_cols or not output_cols:
        st.error("입력 변수와 출력 변수를 모두 선택해주세요.")
        st.stop()

    # Step 3. 데이터 분할 비율 설정
    st.subheader("3. 데이터 분할 비율 설정")
    train_ratio = st.slider("Train 데이터 비율 (%)", 10, 90, 70)
    val_ratio = st.slider("Validation 데이터 비율 (%)", 5, 45, 20)
    test_ratio = 100 - train_ratio - val_ratio
    st.write(f"Test 데이터 비율: {test_ratio}%")

    if test_ratio <= 0:
        st.error("Train + Validation 비율 합이 100%를 초과했습니다. 조정해 주세요.")
        st.stop()
    else:
        st.session_state.split_ratios = (train_ratio/100, val_ratio/100, test_ratio/100)

    # Step 4. 모델 파라미터 설정
    st.subheader("4. 모델 파라미터 설정")
    
    # 다항식 차수 설정
    st.markdown("**다항식 모델 설정**")
    poly_degree = st.number_input("다항식 차수", min_value=1, max_value=10, value=2, step=1)
    
    # 다중다항식 차수 설정
    st.markdown("**다중다항식 모델 설정**")
    multi_poly_degree = st.number_input("다중다항식 차수", min_value=1, max_value=5, value=2, step=1)
    
    # 신경망 설정
    st.markdown("**신경망 모델 설정**")
    num_hidden_layers = st.number_input("은닉층 수", min_value=1, max_value=10, value=2, step=1)
    
    hidden_layers = []
    for i in range(num_hidden_layers):
        st.markdown(f"**{i+1}번째 은닉층 설정**")
        units = st.number_input(f"{i+1}번째 층 뉴런 수", min_value=1, max_value=512, value=64, step=1, key=f"units_{i}")
        activation = st.selectbox(f"{i+1}번째 층 활성화 함수", options=["relu", "tanh", "sigmoid", "linear"], index=0, key=f"activation_{i}")
        dropout_rate = st.slider(f"{i+1}번째 층 드롭아웃 비율", 0.0, 0.5, 0.2, step=0.1, key=f"dropout_{i}")
        hidden_layers.append((units, activation, dropout_rate))
    
    epochs = st.number_input("학습 에폭 수", min_value=1, max_value=10000, value=100, step=10)
    batch_size = st.number_input("배치 크기", min_value=1, max_value=1024, value=32, step=1)
    learning_rate = st.number_input("학습률", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f")

    if st.button("모델 학습 시작"):
        # Forward 학습
        st.write("Forward 모델 학습 중...")
        X = data[input_cols].values
        y = data[output_cols].values

        # 데이터 정규화
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        
        st.session_state.scaler_X = scaler_X
        st.session_state.scaler_y = scaler_y

        # 데이터 분할
        train_ratio, val_ratio, test_ratio = st.session_state.split_ratios
        X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=(1-train_ratio), random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(test_ratio/(val_ratio + test_ratio)), random_state=42)

        st.success(f"Forward 데이터 분할 완료: Train {len(X_train)}개 / Val {len(X_val)}개 / Test {len(X_test)}개")

        # Forward 모델 학습
        # 1. 다항식 모델
        st.write("Forward 다항식 모델 학습 중...")
        poly = PolynomialFeatures(degree=poly_degree)
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)
        X_test_poly = poly.transform(X_test)
        
        poly_model = LinearRegression()
        poly_model.fit(X_train_poly, y_train)
        st.session_state.models['polynomial'] = (poly_model, poly)

        # 2. 다중다항식 모델
        st.write("Forward 다중다항식 모델 학습 중...")
        multi_poly = PolynomialFeatures(degree=multi_poly_degree, include_bias=False)
        X_train_multi = multi_poly.fit_transform(X_train)
        X_val_multi = multi_poly.transform(X_val)
        X_test_multi = multi_poly.transform(X_test)
        
        multi_poly_model = LinearRegression()
        multi_poly_model.fit(X_train_multi, y_train)
        st.session_state.models['multipolynomial'] = (multi_poly_model, multi_poly)

        # 3. 신경망 모델
        st.write("Forward 신경망 모델 학습 중...")
        def build_model(input_dim, output_dim):
            model = keras.Sequential()
            model.add(layers.Input(shape=(input_dim,)))
            
            for units, activation, dropout_rate in hidden_layers:
                model.add(layers.Dense(units, activation=activation))
                model.add(layers.BatchNormalization())
                model.add(layers.Dropout(dropout_rate))
            
            model.add(layers.Dense(output_dim))
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
                        loss='mse', 
                        metrics=['mae'])
            return model

        nn_model = build_model(X.shape[1], y.shape[1])
        history = nn_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        st.session_state.models['neural_network'] = nn_model
        st.session_state.history = history

        # Reverse 학습
        st.write("\nReverse 모델 학습 중...")
        X_rev = data[output_cols].values
        y_rev = data[input_cols].values

        # 데이터 정규화
        scaler_X_rev = StandardScaler()
        scaler_y_rev = StandardScaler()
        X_rev_scaled = scaler_X_rev.fit_transform(X_rev)
        y_rev_scaled = scaler_y_rev.fit_transform(y_rev)
        
        st.session_state.scaler_X_rev = scaler_X_rev
        st.session_state.scaler_y_rev = scaler_y_rev

        # 데이터 분할
        X_rev_train, X_rev_temp, y_rev_train, y_rev_temp = train_test_split(X_rev_scaled, y_rev_scaled, test_size=(1-train_ratio), random_state=42)
        X_rev_val, X_rev_test, y_rev_val, y_rev_test = train_test_split(X_rev_temp, y_rev_temp, test_size=(test_ratio/(val_ratio + test_ratio)), random_state=42)

        st.success(f"Reverse 데이터 분할 완료: Train {len(X_rev_train)}개 / Val {len(X_rev_val)}개 / Test {len(X_rev_test)}개")

        # Reverse 모델 학습
        # 1. 다항식 모델
        st.write("Reverse 다항식 모델 학습 중...")
        poly_rev = PolynomialFeatures(degree=poly_degree)
        X_rev_train_poly = poly_rev.fit_transform(X_rev_train)
        X_rev_val_poly = poly_rev.transform(X_rev_val)
        X_rev_test_poly = poly_rev.transform(X_rev_test)
        
        poly_rev_model = LinearRegression()
        poly_rev_model.fit(X_rev_train_poly, y_rev_train)
        st.session_state.models['polynomial_rev'] = (poly_rev_model, poly_rev)

        # 2. 다중다항식 모델
        st.write("Reverse 다중다항식 모델 학습 중...")
        multi_poly_rev = PolynomialFeatures(degree=multi_poly_degree, include_bias=False)
        X_rev_train_multi = multi_poly_rev.fit_transform(X_rev_train)
        X_rev_val_multi = multi_poly_rev.transform(X_rev_val)
        X_rev_test_multi = multi_poly_rev.transform(X_rev_test)
        
        multi_poly_rev_model = LinearRegression()
        multi_poly_rev_model.fit(X_rev_train_multi, y_rev_train)
        st.session_state.models['multipolynomial_rev'] = (multi_poly_rev_model, multi_poly_rev)

        # 3. 신경망 모델
        st.write("Reverse 신경망 모델 학습 중...")
        nn_rev_model = build_model(X_rev.shape[1], y_rev.shape[1])
        history_rev = nn_rev_model.fit(
            X_rev_train, y_rev_train,
            validation_data=(X_rev_val, y_rev_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        st.session_state.models['neural_network_rev'] = nn_rev_model
        st.session_state.history_rev = history_rev

        st.session_state.trained = True
        st.session_state.input_cols = input_cols
        st.session_state.output_cols = output_cols

        st.success("Forward/Reverse 모델 학습 완료!")

        # 모델 저장
        if not os.path.exists('models'):
            os.makedirs('models')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 모델 정보 생성
        model_info = {
            'timestamp': timestamp,
            'input_cols': input_cols,
            'output_cols': output_cols,
            'poly_degree': poly_degree,
            'multi_poly_degree': multi_poly_degree,
            'num_hidden_layers': num_hidden_layers,
            'hidden_layers': hidden_layers,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        
        # Forward 모델 저장
        joblib.dump(poly_model, f'models/poly_model_{timestamp}.pkl')
        joblib.dump(poly, f'models/poly_features_{timestamp}.pkl')
        joblib.dump(multi_poly_model, f'models/multi_poly_model_{timestamp}.pkl')
        joblib.dump(multi_poly, f'models/multi_poly_features_{timestamp}.pkl')
        nn_model.save(f'models/nn_model_{timestamp}.h5')
        joblib.dump(scaler_X, f'models/scaler_X_{timestamp}.pkl')
        joblib.dump(scaler_y, f'models/scaler_y_{timestamp}.pkl')
        
        # Reverse 모델 저장
        joblib.dump(poly_rev_model, f'models/poly_model_rev_{timestamp}.pkl')
        joblib.dump(poly_rev, f'models/poly_features_rev_{timestamp}.pkl')
        joblib.dump(multi_poly_rev_model, f'models/multi_poly_model_rev_{timestamp}.pkl')
        joblib.dump(multi_poly_rev, f'models/multi_poly_features_rev_{timestamp}.pkl')
        nn_rev_model.save(f'models/nn_model_rev_{timestamp}.h5')
        joblib.dump(scaler_X_rev, f'models/scaler_X_rev_{timestamp}.pkl')
        joblib.dump(scaler_y_rev, f'models/scaler_y_rev_{timestamp}.pkl')
        
        # 모델 정보 저장
        joblib.dump(model_info, f'models/model_info_{timestamp}.pkl')

        # Step 5. 학습 결과 출력
        st.subheader("5. 학습 결과 그래프")
        
        # Forward 신경망 학습 곡선
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history.history['loss'], label='Forward Train Loss')
        ax.plot(history.history['val_loss'], label='Forward Validation Loss')
        ax.set_title('Forward Neural Network Training History')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Reverse 신경망 학습 곡선
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history_rev.history['loss'], label='Reverse Train Loss')
        ax.plot(history_rev.history['val_loss'], label='Reverse Validation Loss')
        ax.set_title('Reverse Neural Network Training History')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # 예측값 vs 실제값 그래프
        st.subheader("예측값 vs 실제값 비교")
        
        # Forward 예측
        y_pred_poly = poly_model.predict(X_test_poly)
        y_pred_multi = multi_poly_model.predict(X_test_multi)
        y_pred_nn = nn_model.predict(X_test)
        
        # Reverse 예측
        y_rev_pred_poly = poly_rev_model.predict(X_rev_test_poly)
        y_rev_pred_multi = multi_poly_rev_model.predict(X_rev_test_multi)
        y_rev_pred_nn = nn_rev_model.predict(X_rev_test)

        # 역정규화
        y_test_original = scaler_y.inverse_transform(y_test)
        y_pred_poly = scaler_y.inverse_transform(y_pred_poly)
        y_pred_multi = scaler_y.inverse_transform(y_pred_multi)
        y_pred_nn = scaler_y.inverse_transform(y_pred_nn)

        y_rev_test_original = scaler_y_rev.inverse_transform(y_rev_test)
        y_rev_pred_poly = scaler_y_rev.inverse_transform(y_rev_pred_poly)
        y_rev_pred_multi = scaler_y_rev.inverse_transform(y_rev_pred_multi)
        y_rev_pred_nn = scaler_y_rev.inverse_transform(y_rev_pred_nn)

        # Forward 예측 결과 그래프
        st.markdown("**Forward 예측 결과**")
        for idx, col in enumerate(output_cols):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(y_test_original[:, idx], y_pred_poly[:, idx], alpha=0.5, label='Polynomial')
            ax.scatter(y_test_original[:, idx], y_pred_multi[:, idx], alpha=0.5, label='Multipolynomial')
            ax.scatter(y_test_original[:, idx], y_pred_nn[:, idx], alpha=0.5, label='Neural Network')
            
            min_val = min(y_test_original[:, idx].min(), y_pred_poly[:, idx].min(), 
                         y_pred_multi[:, idx].min(), y_pred_nn[:, idx].min())
            max_val = max(y_test_original[:, idx].max(), y_pred_poly[:, idx].max(), 
                         y_pred_multi[:, idx].max(), y_pred_nn[:, idx].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')
            
            ax.set_xlabel(f'Actual {col}')
            ax.set_ylabel(f'Predicted {col}')
            ax.set_title(f'Forward {col} - Model Comparison')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        # Reverse 예측 결과 그래프
        st.markdown("**Reverse 예측 결과**")
        for idx, col in enumerate(input_cols):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(y_rev_test_original[:, idx], y_rev_pred_poly[:, idx], alpha=0.5, label='Polynomial')
            ax.scatter(y_rev_test_original[:, idx], y_rev_pred_multi[:, idx], alpha=0.5, label='Multipolynomial')
            ax.scatter(y_rev_test_original[:, idx], y_rev_pred_nn[:, idx], alpha=0.5, label='Neural Network')
            
            min_val = min(y_rev_test_original[:, idx].min(), y_rev_pred_poly[:, idx].min(), 
                         y_rev_pred_multi[:, idx].min(), y_rev_pred_nn[:, idx].min())
            max_val = max(y_rev_test_original[:, idx].max(), y_rev_pred_poly[:, idx].max(), 
                         y_rev_pred_multi[:, idx].max(), y_rev_pred_nn[:, idx].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')
            
            ax.set_xlabel(f'Actual {col}')
            ax.set_ylabel(f'Predicted {col}')
            ax.set_title(f'Reverse {col} - Model Comparison')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        # Step 6. 모델 성능 평가
        st.subheader("6. 모델 성능 평가")

        def evaluate_model(y_true, y_pred, name="Model"):
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            return {
                'Model': name,
                'MSE': f"{mse:.4f}",
                'MAE': f"{mae:.4f}",
                'R²': f"{r2:.4f}"
            }

        # Forward 모델 성능 평가
        st.markdown("**Forward 모델 성능 평가**")
        forward_results = []
        forward_results.append(evaluate_model(y_test_original, y_pred_poly, "Forward Polynomial Model"))
        forward_results.append(evaluate_model(y_test_original, y_pred_multi, "Forward Multipolynomial Model"))
        forward_results.append(evaluate_model(y_test_original, y_pred_nn, "Forward Neural Network Model"))
        
        forward_df = pd.DataFrame(forward_results)
        st.session_state.model_performance['forward'] = forward_df
        st.table(forward_df)

        # Reverse 모델 성능 평가
        st.markdown("**Reverse 모델 성능 평가**")
        reverse_results = []
        reverse_results.append(evaluate_model(y_rev_test_original, y_rev_pred_poly, "Reverse Polynomial Model"))
        reverse_results.append(evaluate_model(y_rev_test_original, y_rev_pred_multi, "Reverse Multipolynomial Model"))
        reverse_results.append(evaluate_model(y_rev_test_original, y_rev_pred_nn, "Reverse Neural Network Model"))
        
        reverse_df = pd.DataFrame(reverse_results)
        st.session_state.model_performance['reverse'] = reverse_df
        st.table(reverse_df)

# Step 7. 예측
if st.session_state.trained:
    st.subheader("7. 예측")
    
    # 모델 성능 평가 결과 표시
    if st.session_state.model_performance['forward'] is not None:
        st.markdown("**현재 모델 성능 평가**")
        
        # Forward 모델 성능 평가
        st.markdown("**Forward 모델 성능 평가**")
        st.table(st.session_state.model_performance['forward'])

        # Reverse 모델 성능 평가
        st.markdown("**Reverse 모델 성능 평가**")
        st.table(st.session_state.model_performance['reverse'])
    
    # 예측 모드 선택
    prediction_mode = st.radio("예측 모드 선택", ["현재 학습된 모델 사용", "저장된 모델 로드"])
    
    if prediction_mode == "저장된 모델 로드":
        # 모델 로드 옵션
        st.markdown("**저장된 모델 로드**")
        if os.path.exists('models'):
            model_files = [f for f in os.listdir('models') if f.endswith('.h5')]
            if model_files:
                # 모델 정보 표시
                st.markdown("**저장된 모델 목록**")
                model_list = []
                for model_file in model_files:
                    timestamp = model_file.split('_')[-1].split('.')[0]
                    info_file = f'models/model_info_{timestamp}.pkl'
                    if os.path.exists(info_file):
                        model_info = joblib.load(info_file)
                        model_list.append({
                            '파일명': model_file,
                            '학습일시': datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S"),
                            '입력변수': ', '.join(model_info['input_cols']),
                            '출력변수': ', '.join(model_info['output_cols']),
                            '다항식차수': model_info['poly_degree'],
                            '다중다항식차수': model_info['multi_poly_degree'],
                            '은닉층수': model_info['num_hidden_layers']
                        })
                
                # 모델 정보를 DataFrame으로 변환하여 표시
                if model_list:
                    model_df = pd.DataFrame(model_list)
                    st.table(model_df)
                
                # 모델 선택
                selected_model = st.selectbox("로드할 모델 선택", model_files)
                if st.button("모델 로드"):
                    try:
                        timestamp = selected_model.split('_')[-1].split('.')[0]
                        
                        # 필요한 파일들이 모두 존재하는지 확인
                        required_files = [
                            f'models/poly_model_{timestamp}.pkl',
                            f'models/poly_features_{timestamp}.pkl',
                            f'models/multi_poly_model_{timestamp}.pkl',
                            f'models/multi_poly_features_{timestamp}.pkl',
                            f'models/nn_model_{timestamp}.h5',
                            f'models/scaler_X_{timestamp}.pkl',
                            f'models/scaler_y_{timestamp}.pkl',
                            f'models/model_info_{timestamp}.pkl'
                        ]
                        
                        missing_files = [f for f in required_files if not os.path.exists(f)]
                        if missing_files:
                            st.error(f"다음 모델 파일들이 없습니다: {', '.join(missing_files)}")
                            st.stop()
                        
                        # 모델 정보 로드
                        model_info = joblib.load(f'models/model_info_{timestamp}.pkl')
                        st.session_state.input_cols = model_info['input_cols']
                        st.session_state.output_cols = model_info['output_cols']
                        
                        # 모든 파일이 존재하면 로드
                        st.session_state.models['polynomial'] = (
                            joblib.load(f'models/poly_model_{timestamp}.pkl'),
                            joblib.load(f'models/poly_features_{timestamp}.pkl')
                        )
                        st.session_state.models['multipolynomial'] = (
                            joblib.load(f'models/multi_poly_model_{timestamp}.pkl'),
                            joblib.load(f'models/multi_poly_features_{timestamp}.pkl')
                        )
                        st.session_state.models['neural_network'] = keras.models.load_model(f'models/nn_model_{timestamp}.h5')
                        st.session_state.scaler_X = joblib.load(f'models/scaler_X_{timestamp}.pkl')
                        st.session_state.scaler_y = joblib.load(f'models/scaler_y_{timestamp}.pkl')
                        
                        st.session_state.trained = True
                        st.success("모델 로드 완료!")
                        
                    except Exception as e:
                        st.error(f"모델 로드 중 오류가 발생했습니다: {str(e)}")
                        st.stop()
    else:
        st.info("현재 학습된 모델을 사용합니다.")

    # 예측 방향 선택
    prediction_direction = st.radio("예측 방향 선택", ["Forward (입력 → 출력)", "Reverse (출력 → 입력)"])
    
    if prediction_direction == "Forward (입력 → 출력)":
        st.markdown("**입력값을 입력하세요 (Input Features):**")
        user_input = []
        for col in st.session_state.input_cols:
            min_val = data[col].min()
            max_val = data[col].max()
            avg_val = data[col].mean()
            val = st.number_input(
                f"{col} (min: {min_val:.2f}, max: {max_val:.2f}, avg: {avg_val:.2f})",
                value=float(avg_val),
                key=f"input_number_{col}"
            )
            user_input.append(val)
    else:
        st.markdown("**입력값을 입력하세요 (Output Features):**")
        user_input = []
        for col in st.session_state.output_cols:
            min_val = data[col].min()
            max_val = data[col].max()
            avg_val = data[col].mean()
            val = st.number_input(
                f"{col} (min: {min_val:.2f}, max: {max_val:.2f}, avg: {avg_val:.2f})",
                value=float(avg_val),
                key=f"input_number_{col}"
            )
            user_input.append(val)

    if st.button("예측하기"):
        user_input_array = np.array(user_input).reshape(1, -1)
        
        if prediction_direction == "Forward (입력 → 출력)":
            # Forward 예측
            user_input_scaled = st.session_state.scaler_X.transform(user_input_array)
            
            # 각 모델별 예측
            poly_model, poly_features = st.session_state.models['polynomial']
            user_input_poly = poly_features.transform(user_input_scaled)
            pred_poly = poly_model.predict(user_input_poly)
            pred_poly = st.session_state.scaler_y.inverse_transform(pred_poly)
            
            multi_poly_model, multi_poly_features = st.session_state.models['multipolynomial']
            user_input_multi = multi_poly_features.transform(user_input_scaled)
            pred_multi = multi_poly_model.predict(user_input_multi)
            pred_multi = st.session_state.scaler_y.inverse_transform(pred_multi)
            
            nn_model = st.session_state.models['neural_network']
            pred_nn = nn_model.predict(user_input_scaled)
            pred_nn = st.session_state.scaler_y.inverse_transform(pred_nn)
            
            # 결과 출력
            st.success("Forward 예측 결과")
            prediction_results = []
            for idx, target in enumerate(st.session_state.output_cols):
                prediction_results.append({
                    'Target': target,
                    '다항식 모델': f"{pred_poly[0][idx]:.2f}",
                    '다중다항식 모델': f"{pred_multi[0][idx]:.2f}",
                    '신경망 모델': f"{pred_nn[0][idx]:.2f}"
                })
        else:
            # Reverse 예측
            user_input_scaled = st.session_state.scaler_X_rev.transform(user_input_array)
            
            # 각 모델별 예측
            poly_model, poly_features = st.session_state.models['polynomial_rev']
            user_input_poly = poly_features.transform(user_input_scaled)
            pred_poly = poly_model.predict(user_input_poly)
            pred_poly = st.session_state.scaler_y_rev.inverse_transform(pred_poly)
            
            multi_poly_model, multi_poly_features = st.session_state.models['multipolynomial_rev']
            user_input_multi = multi_poly_features.transform(user_input_scaled)
            pred_multi = multi_poly_model.predict(user_input_multi)
            pred_multi = st.session_state.scaler_y_rev.inverse_transform(pred_multi)
            
            nn_model = st.session_state.models['neural_network_rev']
            pred_nn = nn_model.predict(user_input_scaled)
            pred_nn = st.session_state.scaler_y_rev.inverse_transform(pred_nn)
            
            # 결과 출력
            st.success("Reverse 예측 결과")
            prediction_results = []
            for idx, target in enumerate(st.session_state.input_cols):
                prediction_results.append({
                    'Target': target,
                    '다항식 모델': f"{pred_poly[0][idx]:.2f}",
                    '다중다항식 모델': f"{pred_multi[0][idx]:.2f}",
                    '신경망 모델': f"{pred_nn[0][idx]:.2f}"
                })
        
        # 예측 결과를 DataFrame으로 변환하여 표시
        results_df = pd.DataFrame(prediction_results)
        st.table(results_df)
        
        # 입력값 정보도 표로 표시
        input_info = []
        for col, val in zip(st.session_state.input_cols if prediction_direction == "Forward (입력 → 출력)" else st.session_state.output_cols, user_input):
            input_info.append({
                'Feature': col,
                '입력값': f"{val:.2f}",
                '최소값': f"{data[col].min():.2f}",
                '최대값': f"{data[col].max():.2f}",
                '평균값': f"{data[col].mean():.2f}"
            })
        
        st.markdown("**입력값 정보**")
        input_df = pd.DataFrame(input_info)
        st.table(input_df)
        
        # 예측 결과 저장
        if st.button("예측 결과 저장"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_df = pd.DataFrame({
                'timestamp': [timestamp],
                'prediction_direction': [prediction_direction],
                **{f'input_{col}': [f"{val:.2f}"] for col, val in zip(st.session_state.input_cols if prediction_direction == "Forward (입력 → 출력)" else st.session_state.output_cols, user_input)},
                **{f'output_{col}_poly': [results[col]['polynomial']] for col in (st.session_state.output_cols if prediction_direction == "Forward (입력 → 출력)" else st.session_state.input_cols)},
                **{f'output_{col}_multi': [results[col]['multipolynomial']] for col in (st.session_state.output_cols if prediction_direction == "Forward (입력 → 출력)" else st.session_state.input_cols)},
                **{f'output_{col}_nn': [results[col]['neural_network']] for col in (st.session_state.output_cols if prediction_direction == "Forward (입력 → 출력)" else st.session_state.input_cols)}
            })
            if not os.path.exists('predictions'):
                os.makedirs('predictions')
            results_df.to_csv(f'predictions/prediction_{timestamp}.csv', index=False)
            st.success("예측 결과가 저장되었습니다!")