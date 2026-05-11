import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# 1. データの読み込み
df = pd.read_csv('cloud_reviews_full.csv')

session = requests.Session()
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Referer": "https://cloud-jp.net/"
}

def get_full_content(url):
    """詳細ページからsectionタグを指定して本文を丸ごと抽出する"""
    try:
        time.sleep(2.0) # サーバー負荷軽減のため2秒待機
        res = session.get(url, headers=headers, timeout=15)
        if res.status_code != 200:
            return "取得失敗(Status Error)"
        
        soup = BeautifulSoup(res.text, "html.parser")
        
        # HTMLソースから判明した、本文を包む正確なタグ
        # <section class="content partsH2-4 partsH3-61"> を探す
        content_section = soup.find("section", class_="content")
        
        if content_section:
            # 記事内の不要な「目次」や「広告」を除外したい場合はここで調整可能
            # 今回は分析用にすべてテキストとして抽出します
            return content_section.get_text(separator="\n", strip=True)
        else:
            return "本文セクションが見つかりませんでした"
            
    except Exception as e:
        return f"エラー: {str(e)}"

# --- 実行 ---
print(f"全 {len(df)} 件の本文取得を開始します。完了まで時間がかかります...")

contents = []
for i, url in enumerate(df['レビューURL']):
    print(f"[{i+1}/{len(df)}] 取得中: {url}")
    body = get_full_content(url)
    contents.append(body)
    
    # 50件ごとに中間保存（万が一のクラッシュ対策）
    if (i + 1) % 50 == 0:
        temp_df = df.head(i + 1).copy()
        temp_df['レビュー本文'] = contents
        temp_df.to_csv('../data/cloud_reviews_intermediate.csv', index=False, encoding='utf-8-sig')
        print(f">>> {i + 1}件まで中間保存しました。")

# 最終保存
df['レビュー本文'] = contents
df.to_csv('../data/cloud_reviews_final.csv', index=False, encoding='utf-8-sig')

print("\nすべて完了しました！ 'cloud_reviews_final.csv' を確認してください。")