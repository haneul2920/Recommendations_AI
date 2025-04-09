from flask import Flask, jsonify, Response
import pandas as pd
from sqlalchemy import create_engine
from surprise import SVD, Dataset, Reader
from transformers import pipeline
import os
import pickle
import json

# DB 설정
db_config = {
    "user": "pandas13",
    "password": "cookit%40012",
    "host": "pandas13.cafe24.com",
    "port": 3306,
    "database": "pandas13"
}

# 감성 분석 클래스
class ProductReviewAnalyzerFromDB:
    def __init__(self, db_config):
        self.db_config = db_config
        self.df = None
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

    def load_and_clean_data(self):
        engine = create_engine(
            f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )
        query = "SELECT * FROM comment"
        self.df = pd.read_sql(query, engine)
        self.df = self.df.dropna(subset=['text'])
        self.df.rename(columns={
            'text': '리뷰 내용',
            'user_id': 'user',
            'product_name': '상품 이름'
        }, inplace=True)

    def analyze_sentiment(self):
        review_texts = self.df['리뷰 내용'].tolist()
        results = [self.sentiment_pipeline(text)[0] for text in review_texts]

        self.df.loc[self.df.index[:len(results)], '감성 결과'] = [r['label'] for r in results]
        self.df.loc[self.df.index[:len(results)], '감성 점수'] = [r['score'] for r in results]
        self.df['감성 점수 정제'] = self.df['감성 결과'].map(
            lambda x: 1.0 if 'positive' in x.lower()
            else 0.5 if 'neutral' in x.lower()
            else 0.0
        )

    def get_dataframe(self):
        return self.df

# 추천 시스템 클래스
class HybridRecommender:
    def __init__(self, df):
        self.df = df
        self.model = SVD()
        self.trainset = None
        self.trained = False

    def prepare_data(self):
        reader = Reader(rating_scale=(0, 5))
        data = Dataset.load_from_df(self.df[['user', '상품 이름', 'user_rating']], reader)
        self.trainset = data.build_full_trainset()

    def train(self):
        self.model.fit(self.trainset)
        self.trained = True

    def predict_cf_scores(self, user_id):
        all_items = self.df['상품 이름'].unique()
        rated_items = self.df[self.df['user'] == user_id]['상품 이름'].unique()
        unrated_items = [item for item in all_items if item not in rated_items]
        predictions = {item: self.model.predict(user_id, item).est for item in unrated_items}
        return predictions

    def hybrid_recommend(self, user_id, top_n=5, alpha=0.7):
        cf_scores = self.predict_cf_scores(user_id)
        senti_scores = self.df.groupby('상품 이름')['감성 점수 정제'].mean().to_dict()
        hybrid_scores = {
            item: alpha * cf_scores[item] + (1 - alpha) * senti_scores.get(item, 0.5) * 5
            for item in cf_scores
        }
        sorted_recommendations = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:top_n]

# Flask 앱 생성
app = Flask(__name__)

# 파일 경로
sentiment_path = 'sentiment_data.pkl'
model_path = 'svd_model.pkl'

# 데이터 및 모델 로딩
if os.path.exists(sentiment_path) and os.path.exists(model_path):
    print("🔁 캐시에서 감성 데이터와 모델 불러오는 중...")
    with open(sentiment_path, 'rb') as f:
        df = pickle.load(f)
    with open(model_path, 'rb') as f:
        recommender = pickle.load(f)
else:
    print("🚀 최초 실행: 감성 분석 및 모델 학습 중...")
    analyzer = ProductReviewAnalyzerFromDB(db_config)
    analyzer.load_and_clean_data()
    analyzer.analyze_sentiment()
    df = analyzer.get_dataframe()

    # 평점 생성
    if 'user_rating' not in df.columns:
        df['user_rating'] = df['감성 점수 정제'] * 5

    recommender = HybridRecommender(df)
    recommender.prepare_data()
    recommender.train()

    # 캐시 저장
    with open(sentiment_path, 'wb') as f:
        pickle.dump(df, f)
    with open(model_path, 'wb') as f:
        pickle.dump(recommender, f)

    print("✅ 캐싱 완료!")

# API 라우트
@app.route('/api/recommend/<user_id>', methods=['GET'])
def recommend(user_id):
    try:
        user_id = user_id.strip()
        if user_id in df['user'].unique():
            recommendations = recommender.hybrid_recommend(user_id)

            product_info = df.drop_duplicates(subset='상품 이름')[['상품 이름', 'product_id', 'product_image', 'price']]
            result = []
            for item, score in recommendations:
                product_row = product_info[product_info['상품 이름'] == item]
                if not product_row.empty:
                    product_data = product_row.iloc[0]
                    result.append({
                        "product_id": int(product_data["product_id"]),
                        "product_name": str(item),
                        "product_image": str(product_data["product_image"]),
                        "price": float(product_data["price"]),
                        "score": round(float(score), 2)
                    })
            json_result = json.dumps(result, ensure_ascii=False, indent=2)
            return Response(json_result, content_type='application/json; charset=utf-8')
        else:
            return jsonify({"error": f"사용자 '{user_id}'는 데이터에 없습니다."}), 404
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        return jsonify({"error": str(e)}), 500

# 서버 실행
if __name__ == '__main__':
    app.run(debug=True, port=5000)
