from flask import Flask, Response
import pandas as pd
import json
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'sentiment_data.pkl')

# 데이터 로딩
df = pd.read_pickle(data_path)

class SimpleRecommender:
    def __init__(self, df):
        self.df = df

    def recommend(self, user_id, top_n=8):
        seen_items = self.df[self.df['user'] == user_id]['상품 이름'].unique()
        unseen_df = self.df[~self.df['상품 이름'].isin(seen_items)]

        score_df = (
            unseen_df.groupby('상품 이름')
            .agg({
                'user_rating': 'mean',
                '감성 점수 정제': 'mean',
                'product_id': 'first',
                'product_image': 'first',
                'price': 'first'
            })
            .reset_index()
        )
        score_df['score'] = 0.7 * score_df['user_rating'] + 0.3 * score_df['감성 점수 정제'] * 5
        return score_df.sort_values(by='score', ascending=False).head(top_n)

recommender = SimpleRecommender(df)

@app.route('/')
def home():
    return '🚀 Flask 추천 시스템 (무료 버전용) 실행 중!'

@app.route('/api/recommend/<user_id>', methods=['GET'])
def recommend(user_id):
    user_id = user_id.strip()
    if user_id not in df['user'].unique():
        return Response(json.dumps({"error": f"사용자 '{user_id}'는 데이터에 없습니다."}, ensure_ascii=False),
                        content_type="application/json; charset=utf-8", status=404)

    top_items = recommender.recommend(user_id)
    result = []
    for _, row in top_items.iterrows():
        result.append({
            "product_id": int(row['product_id']),
            "product_name": row['상품 이름'],
            "product_image": row['product_image'],
            "price": float(row['price']),
            "score": round(float(row['score']), 2)
        })

    return Response(json.dumps(result, ensure_ascii=False), content_type="application/json; charset=utf-8")

if __name__ == '__main__':
    app.run(debug=True)
