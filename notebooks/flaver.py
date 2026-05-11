import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# --- 設定 ---
all_data = []
MAX_PAGE = 12  # 1〜11ページまで取得

# --- 1. 全商品の基本情報を取得 ---
for page in range(1, MAX_PAGE):
    url = f"https://www.aslaj.com/view/category/ct6?page={page}"
    print(f"ページ {page} の基本情報を取得中...")
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)
        res.encoding = "utf-8"
        soup = BeautifulSoup(res.text, "html.parser")
        
        items = soup.select("li.itemList__unit")
        for item in items:
            name_tag = item.select_one(".itemName")
            price_tag = item.select_one(".itemPrice")
            link_tag = item.select_one("a.itemWrap")
            
            if name_tag and price_tag and link_tag:
                full_name = name_tag.get_text(strip=True)
                brand = "不明"
                flavor = full_name
                if " - " in full_name:
                    parts = full_name.rsplit(" - ", 1)
                    flavor = parts[0]
                    brand = parts[1]
                
                price_text = price_tag.get_text(strip=True).split("円")[0].replace(",", "")
                
                all_data.append({
                    "ブランド": brand,
                    "フレーバー名": flavor,
                    "価格": int(price_text) if price_text.isdigit() else price_text,
                    "詳細URL": "https://www.aslaj.com" + link_tag["href"]
                })
        time.sleep(1)
    except Exception as e:
        print(f"エラー: {e}")

df_master = pd.DataFrame(all_data)

# --- 2. 詳細ページから説明文を取る関数 ---
def get_description(url):
    try:
        # 連続アクセスしすぎないよう少し待つ
        time.sleep(0.5) 
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        res.encoding = "utf-8"
        soup = BeautifulSoup(res.text, "html.parser")
        
        # ASLAJのソースを見ると、説明文は .item-detail または .item-exp に入っていることが多いです
        # 複数の候補で探します
        desc_tag = soup.select_one(".item-detail") or soup.select_one(".item-exp") or soup.select_one("#itemDetail")
        return desc_tag.get_text(strip=True) if desc_tag else "説明なし"
    except:
        return "取得エラー"

# --- 3. 最初の10件だけで「説明文」の取得テスト ---
print("\n最初の10件だけ詳細説明を取得します（テスト）...")
# 最初の10件をコピー
#df_test = df_master.head(10).copy()
df_test = df_master.copy()
df_test["説明文"] = df_test["詳細URL"].apply(get_description)

# 結果を表示
print("\n--- テスト結果 ---")
print(df_test[["ブランド", "フレーバー名", "説明文"]])

# --- 4. 保存 ---
# 基本情報全件
df_master.to_csv("../data/aslaj_master_list.csv", index=False, encoding="utf-8-sig")
# テスト結果（説明文入り）
df_test.to_csv("../data/aslaj_test_with_desc.csv", index=False, encoding="utf-8-sig")

print("\nCSVを保存しました：aslaj_master_list.csv, aslaj_test_with_desc.csv")