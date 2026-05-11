import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

all_reviews = []
session = requests.Session()

# 擬態用ヘッダー
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Referer": "https://cloud-jp.net/"
}

# 現在のページ数に合わせて調整（多めに25ページまで設定）
MAX_PAGE = 25 

print("CLOUD 全レビューデータの取得を開始します...")

for page in range(1, MAX_PAGE + 1):
    url = f"https://cloud-jp.net/category/flavor-review/page/{page}/"
    print(f"現在 {page} ページ目を処理中...", end=" ")
    
    try:
        res = session.get(url, headers=headers, timeout=15)
        
        # ページが存在しない（最後のページを超えた）場合は終了
        if res.status_code == 404:
            print("\n全ページの取得が完了しました。")
            break
        elif res.status_code != 200:
            print(f"\nアクセス失敗（Code: {res.status_code}）。次のページへ進みます。")
            continue

        soup = BeautifulSoup(res.text, "html.parser")
        articles = soup.find_all("article", class_="archive__item")
        
        # 記事が見つからない場合も終了
        if not articles:
            print("\n記事が見つからないため、終了します。")
            break
            
        page_count = 0
        for art in articles:
            title_tag = art.find("h2", class_="heading-secondary")
            if not title_tag: continue
            
            title_link = title_tag.find("a")
            date_tag = art.find("li", class_="icon-clock")
            summary_tag = art.find("p", class_="phrase-secondary")
            
            if title_link:
                all_reviews.append({
                    "レビュータイトル": title_link.get_text(strip=True),
                    "更新日": date_tag.get_text(strip=True) if date_tag else "不明",
                    "概要": summary_tag.get_text(strip=True) if summary_tag else "",
                    "レビューURL": title_link["href"]
                })
                page_count += 1
        
        print(f"({page_count}件取得)")
        
        # サーバー負荷軽減とブロック防止のため、少し長めに待機
        time.sleep(2.0)
        
    except Exception as e:
        print(f"\nエラー発生: {e}")
        # 途中で止まっても、それまでのデータを保存するためにループを抜ける
        break

# 取得したデータを保存
df_cloud_all = pd.DataFrame(all_reviews)

if not df_cloud_all.empty:
    # 保存。後の結合のためにブランド名を分離する準備もしておくと楽です
    df_cloud_all.to_csv("../data/cloud_reviews_full.csv", index=False, encoding="utf-8-sig")
    print(f"\n--- 最終結果 ---")
    print(f"合計取得件数: {len(df_cloud_all)} 件")
    print("ファイル名: cloud_reviews_full.csv で保存しました。")
else:
    print("\nデータが取得できませんでした。")