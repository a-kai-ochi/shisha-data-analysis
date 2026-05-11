# shisha-data-analysis
# シーシャフレーバーの市場価格とユーザーレビューの相関分析データ

## 1. データの出典
- 販売データ: [ASLAJ](https://www.aslaj.com/)
- レビューデータ: [CLOUD](https://cloud-jp.net/)

## 2. 内容概要
国内最大級のシーシャ通販サイトとレビューサイトから、約500件のフレーバー情報と、それに対する詳細なレビュー本文をスクレイピングにより収集した。
- `aslaj_all_flavors.csv`: ブランド、フレーバー名、価格、詳細URL
- `cloud_reviews_final.csv`: レビュータイトル、更新日、レビュー本文、URL

## 3. 分析の目的・期待
感性的な嗜好品である「シーシャ」を、データサイエンスの視点から客観的に分析する。
- ブランド別ポジショニング: 価格帯とレビューで使われる語彙（シルキー、重い等）の相関。
- フレーバー推薦: レビュー本文の共起ネットワークを用いた「相性の良いミックス」の自動抽出。

## 4. 収集方法
Python (Requests, BeautifulSoup) を用いたWebスクレイピング。
ボット対策への対応として、Session管理やUser-Agentの設定、適切な待機時間（2秒）を実装している。

## 5. ディレクトリ構成

```
├── README.md                # 課題の目的・データ出典・分析の展望を記述
├── requirements.txt         # 実行に必要なライブラリ (pandas, requests, bs4等)
├── data/                    # 収集したデータセットを格納
│   ├── aslaj_master_list.csv    # ASLAJから取得した販売価格・商品リスト
│   └── cloud_reviews_final.csv   # CLOUDから取得したレビュー本文を含む最終データ
└── notebooks/               # スクレイピング実行スクリプト
    ├── flaver.py                # ASLAJ販売データ取得用コード
    └── flavor-review-all.py     # CLOUDレビュー全文取得用コード
```
