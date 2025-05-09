# 다중 회귀 모델 비교 분석 웹 앱

이 웹 앱은 다항식, 다중다항식, 신경망 모델을 사용하여 다중 회귀 분석을 수행하고 결과를 비교할 수 있는 도구입니다.

## 주요 기능

1. 데이터 업로드 (엑셀 복사/붙여넣기 또는 CSV 파일)
2. 입력/출력 변수 설정
3. 모델 파라미터 설정
4. Forward/Reverse 모델 학습
5. 학습 결과 시각화
6. 모델 성능 평가
7. 예측 수행

## 로컬에서 실행하기

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. 앱 실행:
```bash
streamlit run method.py
```

## 웹 배포하기

### Streamlit Cloud 사용하기

1. [Streamlit Cloud](https://streamlit.io/cloud)에 가입합니다.
2. GitHub에 코드를 업로드합니다.
3. Streamlit Cloud에서 "New app"을 클릭합니다.
4. GitHub 저장소를 선택하고 `method.py`를 메인 파일로 지정합니다.
5. "Deploy"를 클릭하여 배포를 시작합니다.

### 다른 클라우드 서비스 사용하기

#### Heroku 배포
1. Heroku 계정 생성
2. Heroku CLI 설치
3. 다음 파일들을 생성:

`Procfile`:
```
web: streamlit run method.py --server.port $PORT
```

`runtime.txt`:
```
python-3.9.18
```

4. 배포 명령어:
```bash
heroku create your-app-name
git add .
git commit -m "Initial commit"
git push heroku main
```

#### AWS Elastic Beanstalk 배포
1. AWS 계정 생성
2. Elastic Beanstalk CLI 설치
3. `.ebextensions/python.config` 파일 생성:
```yaml
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: method:app
```

4. 배포 명령어:
```bash
eb init
eb create
eb deploy
```

## 주의사항

1. 모델 파일과 예측 결과는 서버의 임시 저장소에 저장되므로, 중요한 데이터는 별도로 백업해야 합니다.
2. 대용량 데이터 처리 시 메모리 사용량에 주의해야 합니다.
3. 보안을 위해 민감한 데이터는 업로드하지 않도록 주의해야 합니다. 