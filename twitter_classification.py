import pandas as pd  
import re
from nltk.stem.porter import *
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data=pd.read_csv("train_twitter.csv")
stemmer = PorterStemmer()
print(data.columns)
arr=[]

# temp='@user when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.'
# print(len(data['tweet']))
arr=[]
for i in data['tweet']:

	r=re.findall('@[\w]*',i)
	temp=i
	for j in r:
		temp=i.replace(j,"_")

	arr.append(temp)

# print(len(arr))

data['tweet']=arr

# print(arr)

data['tweet']=data['tweet'].str.replace("[^a-zA-Z#]"," ")

data['tweet']=data['tweet'].apply(lambda x:' '.join([w for w in x.split() if len(w)>3]))

token_tweet=[]
token_tweet=data['tweet'].apply(lambda x: x.split())

# print(token_tweet.head())


token_tweet=token_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

temp=[]


for i in token_tweet:
	temp.append(' '.join(i))

# print(temp[:10])	

data['tweet']=temp


words=' '.join(w for w in data['tweet'])

# print(words)

print("ALL WORDS")
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
# plt.show()


temp_data=data.loc[data['label']==1,['tweet']]

# print(temp_data.head())

negative_words=' '.join(text for text in temp_data['tweet'])

temp_data=[]

temp_data=data.loc[data['label']==0,['tweet']]
positive_words=' '.join(text for text in temp_data['tweet'])

print("NEGATIVE WORDS")
wordcloud1 = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud1, interpolation="bilinear")
plt.axis('off')
# plt.show()


print("POSITIVE WORDS")
wordcloud2 = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(positive_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud2, interpolation="bilinear")
plt.axis('off')
# plt.show()

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(data['tweet'])

print(tfidf.shape)

train_tfidf = tfidf[:31962,:]
print(data['label'].shape)


print(train_tfidf.shape)
xtrain, xvalid, ytrain, yvalid = train_test_split(train_tfidf, data['label'], random_state=42, test_size=0.3)
print(ytrain.index)
# print("xtrain",xtrain.shape)
# print("xvalid",xvalid.shape)
# print("ytrain",ytrain.shape)
# print("yvalid",yvalid.shape)

xtrain_tfidf=train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]
lreg = LogisticRegression()
lreg.fit(xtrain_tfidf, ytrain)

print(lreg.score(xvalid_tfidf,yvalid))