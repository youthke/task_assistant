import word_dividor
import pandas as pd
#Bag of Words作る
from word_dividor import WordDividor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


class Assistant:
    def __init__(self):
       df = pd.read_csv('./data/input_word.csv')
       cv = CountVectorizer(analyzer=word_dividor.WordDividor().extract_words, max_features=500)
       X = cv.fit_transform(df['texts'])
       Y = df['func_type']
       clf = MultinomialNB()
       clf.fit(X, Y)
       self.cv = cv
       self.clf = clf

    def predict_proba(self, text):
        inputX = self.cv.transform([text])
        return self.clf.predict_proba(inputX)
    
    def predict(self, text):
        inputX = self.cv.transform([text])
        return self.clf.predict(inputX)

    def is_clearly(self, text):
        predict_num = self.predict(text)
        proba = self.predict_proba(text)
        return proba[0][predict_num] > 0.5

    def returnWord(self, text):
        print(f"inputword: {text}")
        predict_num = self.predict(text)[0]
        if not self.is_clearly(text)[0]:
            return "すみません. よくわかりません"
        elif predict_num == 0:
            return "課題の登録ですね！どんな課題ですか？"
        elif predict_num == 1:
            return "課題の表示ですね！こちらが残りの課題です！"
        elif predict_num == 2:
            return "課題の更新ですね！どの課題が終了しましたか？"
        elif predict_num == 3:
            return "課題の削除ですね！どの課題の削除ですか？"
        else:
            return "error!"



        



if "__main__":
    switcher = Assistant()
    df = pd.read_csv('./data/input_word.csv')
    for index, row in df.iterrows():
        print(f"{index}:{switcher.returnWord(row['texts'])}")
