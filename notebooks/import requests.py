import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

def scrape_2026_full_data(month_num):
    m_str = str(month_num).zfill(2)
    list_url = f"https://npb.jp/games/2026/schedule_{m_str}_detail.html"
    
    print(f"--- 2026年{month_num}月の解析を開始します ---")
    res = requests.get(list_url)
    res.encoding = 'utf-8'
    soup = BeautifulSoup(res.text, 'html.parser')
    
    # 詳細リンクの抽出
    links = [a['href'] for a in soup.find_all('a', href=re.compile(r'/scores/2026/'))]
    unique_links = sorted(list(set(links))) 
    
    print(f"合計 {len(unique_links)} 試合が見つかりました。")
    game_data = []

    for link in unique_links:
        top_url = f"https://npb.jp{link}"
        # 試合経過ページのURLを作成
        pbp_url = top_url.replace("index.html", "") + "playbyplay.html"
        
        print(f"解析中: {top_url}")
        time.sleep(1) # サーバー負荷対策
        
        try:
            # 1. 試合TOPページから基本スタッツを取得
            res_top = requests.get(top_url)
            res_top.encoding = 'utf-8'
            soup_top = BeautifulSoup(res_top.text, 'html.parser')
            
            score_table = soup_top.find('table', id='tablefix_ls')
            rows = score_table.find_all('tr')
            v_cells = [td.get_text(strip=True) for td in rows[1].find_all(['td', 'th'])]
            h_cells = [td.get_text(strip=True) for td in rows[2].find_all(['td', 'th'])]

            info_text = soup_top.find('p', class_='game_info').get_text()
            attendance = re.search(r'入場者\s*([\d,]+)人', info_text)
            
            game_tit = soup_top.find('div', class_='game_tit')
            date = game_tit.find('time').get_text()
            stadium = game_tit.find('span', class_='place').get_text()

            # 2. 試合経過ページからテキストを取得
            res_pbp = requests.get(pbp_url)
            res_pbp.encoding = 'utf-8'
            soup_pbp = BeautifulSoup(res_pbp.text, 'html.parser')
            
            # id="progress" 内のテキストを抽出
            progress_div = soup_pbp.find('div', id='progress')
            pbp_lines = []
            if progress_div:
                # イニング(h5)とプレー内容(td.w2)を順番に取得
                elements = progress_div.find_all(['h5', 'td'], class_=['', 'w2'])
                for el in elements:
                    text = el.get_text(strip=True)
                    if text and text != "結果": # 見出しの「結果」は除外
                        pbp_lines.append(text)
            
            full_pbp_text = "\n".join(pbp_lines)

            # データを辞書に格納
            game_data.append({
                "日付": date,
                "球場": stadium,
                "ホーム球団": h_cells[0],
                "ビジター球団": v_cells[0],
                "ホーム得点": h_cells[-3],
                "ビジター得点": v_cells[-3],
                "ホーム安打": h_cells[-2],
                "ビジター安打": v_cells[-2],
                "ホーム失策": h_cells[-1],
                "ビジター失策": v_cells[-1],
                "観客動員数": int(attendance.group(1).replace(',', '')) if attendance else 0,
                "試合経過": full_pbp_text,
                "URL": top_url
            })
            
        except Exception as e:
            print(f"  エラー発生 ({link}): {e}")
            
    return pd.DataFrame(game_data)

# --- メイン処理 ---
df_april = scrape_2026_full_data(4)

# CSVに保存
output_file = "npb_2026_04_full.csv"
df_april.to_csv(output_file, index=False, encoding="utf-8-sig")

print("-" * 30)
print(f"完了しました！ データを '{output_file}' に保存しました。")