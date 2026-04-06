import pandas as pd
from utils.fact_checker import extract_keywords, search_online, compare_articles

df = pd.read_csv('data/labeled_news.csv')
kannada_df = df[df['source'].isin(['Prajavani', 'Vijayakarnataka'])]
if len(kannada_df) > 0:
    title = kannada_df['title'].iloc[0]
else:
    title = 'ದಾವಣಗೆರೆ ಬಳಿ ಭೀಕರ ಅಪಘಾತ 5 ಸಾವು'

print(f'Input: {title}')
kw = extract_keywords(title)
print(f'Extracted Keywords: {kw}')

res = search_online(kw)
print(f'Search Results Count: {len(res)}')
if len(res) > 0:
    print(f'Top match title: {res[0]["title"]}')

comp = compare_articles(title, res)
print(f'Status: {comp["status"]} | Score: {comp["score"]}')
