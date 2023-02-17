import snscrape
import snscrape.modules.twitter as sntwitter
import pandas as pd
import itertools
raw_content = []
result = sntwitter.TwitterHashtagScraper('macbook_2022').get_items()
sliced_scraped_tweets = itertools.islice(result, 200)
df = pd.DataFrame(sliced_scraped_tweets)[['date', 'rawContent','lang']]
data = df[df['lang'] == 'en']
print(data)
# print(result)
# counter = 0
# for i, tweet in enumerate(result):
#     if(tweet.lang == 'en'):
#         print(tweet.rawContent)
#         counter += 1
#         if counter > 50:
#             break
