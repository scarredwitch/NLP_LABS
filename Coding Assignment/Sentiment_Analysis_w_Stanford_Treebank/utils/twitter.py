import twint

# Configure Twint to search for tweets containing a certain keyword

def scraper(keyword):
    c = twint.Config()
    c.Search = keyword
    c.Limit = 50
    c.Store_object = True
    c.Hide_output = True # don't show output in terminal
    c.Lang = "en" # english
    twint.run.Search(c)
    tweets = twint.output.tweets_list

    return tweets

if __name__ =="__main__":
    tweets = scraper("hogwarts legacy")
    print('--'*30)
    for tw in tweets:
        print(tw.tweet)
        break