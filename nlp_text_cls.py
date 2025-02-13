# sentiment analysis simple example using NLTK.
import random
from numpy.lib.function_base import vectorize
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize  # tokenization
from nltk.corpus import stopwords, wordnet  # preprocessing (remove stopwords,punctuation)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# labeled dataset we need to define
data = [
    # Positive Sentences
    ("The food was absolutely delicious", 1),
    ("I loved the atmosphere", 1),
    ("The waiter was very friendly and helpful", 1),
    ("This place has an amazing vibe", 1),
    ("I had a wonderful experience", 1),
    ("The service was quick and efficient", 1),
    ("Everything was perfect, I highly recommend it", 1),
    ("The staff was extremely polite", 1),
    ("I enjoyed every bite of my meal", 1),
    ("The music was soothing and enjoyable", 1),
    ("The ambiance is just fantastic", 1),
    ("The drinks were refreshing", 1),
    ("It was an amazing experience overall", 1),
    ("The dessert was heavenly", 1),
    ("Best dining experience I have had", 1),
    ("The flavors were perfectly balanced", 1),
    ("This place exceeded my expectations", 1),
    ("I am definitely coming back", 1),
    ("It was worth every penny", 1),
    ("The chef did an outstanding job", 1),
    ("The presentation of the food was stunning", 1),
    ("I loved the variety in the menu", 1),
    ("The seats were comfortable", 1),
    ("I was treated with great hospitality", 1),
    ("The location is perfect", 1),
    ("Everything tasted fresh", 1),
    ("Highly recommended for family dinners", 1),
    ("I felt very welcomed", 1),
    ("The portions were generous", 1),
    ("The cleanliness was top-notch", 1),
    ("The live music made the evening special", 1),
    ("The place had a relaxing atmosphere", 1),
    ("Customer service was excellent", 1),
    ("The staff went above and beyond", 1),
    ("It was a delightful surprise", 1),
    ("Perfect place for a date night", 1),
    ("I loved the attention to detail", 1),
    ("The seasoning was spot on", 1),
    ("The decor was aesthetically pleasing", 1),
    ("The restaurant had a great selection of wines", 1),
    ("The pasta was cooked to perfection", 1),
    ("It felt like a home away from home", 1),
    ("Everything was fresh and well-prepared", 1),
    ("The burger was juicy and flavorful", 1),
    ("The portions were more than enough", 1),
    ("The lighting created a cozy atmosphere", 1),
    ("They have the best coffee in town", 1),
    ("I had a great time with my friends", 1),
    ("It was a five-star experience", 1),
    ("This place is a hidden gem", 1),

    # Negative Sentences
    ("The food was bland and tasteless", 0),
    ("The service was extremely slow", 0),
    ("I had a terrible experience", 0),
    ("The waiter was rude and unprofessional", 0),
    ("The place was too noisy", 0),
    ("The chairs were very uncomfortable", 0),
    ("The food was overpriced and not worth it", 0),
    ("The restaurant was overcrowded", 0),
    ("I waited for an hour to get my food", 0),
    ("The portion sizes were too small", 0),
    ("The food was served cold", 0),
    ("The restroom was dirty", 0),
    ("The steak was overcooked and dry", 0),
    ("The drinks were watered down", 0),
    ("There was a hair in my food", 0),
    ("The atmosphere felt dull and lifeless", 0),
    ("The bread was stale", 0),
    ("The soup was too salty", 0),
    ("The food quality was very poor", 0),
    ("I had food poisoning after eating here", 0),
    ("The customer service was awful", 0),
    ("The music was too loud", 0),
    ("I was ignored by the staff", 0),
    ("It was way too expensive for the quality", 0),
    ("The place had a bad smell", 0),
    ("They got my order completely wrong", 0),
    ("The menu was very limited", 0),
    ("The rice was undercooked", 0),
    ("The fish tasted off", 0),
    ("The waiter never refilled my water", 0),
    ("The food looked better than it tasted", 0),
    ("The place felt outdated", 0),
    ("There were flies around the table", 0),
    ("The coffee was burnt", 0),
    ("The fries were soggy", 0),
    ("They forgot my side dish", 0),
    ("The ice cream was melted when served", 0),
    ("The food had no seasoning", 0),
    ("The soup was served lukewarm", 0),
    ("The restaurant was too dark", 0),
    ("I got a stomach ache after eating here", 0),
    ("The waiter had an attitude", 0),
    ("The place was not well-maintained", 0),
    ("The sauce was too sweet", 0),
    ("The salad was wilted", 0),
    ("The food was greasy", 0),
    ("The service was disappointing", 0),
    ("I will not be coming back", 0),
    ("Overall, a bad experience", 0)
]

# Text data augmentation techniques: synonym_replacement,random_deletion,word_order_shuffling
# synonym replacement
def synonym_replacement(sentence,n=1):
      words=word_tokenize(sentence)
      new_words=words.copy()  # make a copy of the words to modify
      count=0 # counter to track the number of replaced words
      for i, word in enumerate(words):
            synonyms=wordnet.synsets(word)
            if synonyms and count < n:
                  synonym=synonyms[0].lemmas()[0].name().replace("_"," ")
                  new_words[i]=synonym
                  count+=1
      return " ".join(new_words)

# Random word deletion function
def random_deletion(sentence,p=0.3):
      words=word_tokenize(sentence)
      if len(words)==1:
            return sentence
      new_words=[word for word in words if random.uniform(0,1) >p]
      return " ".join(new_words) if new_words else words[0]

# word order shuffling function
def word_order_shuffling(sentence):
      words=word_tokenize(sentence)
      random.shuffle(words)
      return " ".join(words)


# Apply data augmentation to dataset
augmented_data=[]
for text,label in data:
      augmented_data.append((text,label))
      augmented_data.append((synonym_replacement(text),label))
      augmented_data.append((random_deletion(text),label))
      augmented_data.append((word_order_shuffling(text),label))

print("\nOriginal dataset size:", len(data))
print("Augmented dataset size:", len(augmented_data))










# preprocessing and tokenization using NLTK
stop_words=set(stopwords.words('english'))  # store stopwords once
def preprocess_text(text):
      text=text.lower()
      text=re.sub(r'[^\w\s]','',text)
      words=word_tokenize(text)
      words=[word for word in words if word not in stop_words]
      return ' '.join(words)
# now we have tokens we need to feature engineering
# Machine learning models does not work with text directly
# so we need to convert into numerical vectors
from sklearn.feature_extraction.text import TfidfVectorizer
# process dataset
texts=[preprocess_text(text) for text, label in augmented_data]
labels=np.array([label for text, label in augmented_data])
print(labels)
vectorizer=TfidfVectorizer()
print(vectorizer)
X=vectorizer.fit_transform(texts)
print(X)



# Model training
# split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test=train_test_split(X,labels,test_size=0.2,random_state=42)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
# train the model
model.fit(X_train,y_train)
# make prediction
y_pred=model.predict(X_test)
# accuracy of the model
from sklearn.metrics import accuracy_score
print(f"Accuracy: {accuracy_score(y_test,y_pred) * 100:.2f}%")


# Testing the model
new_text = ["I felt very welcomed"]
new_text_vectorized=vectorizer.transform([preprocess_text(text) for text in new_text])
prediction=model.predict(new_text_vectorized)
print("Positive" if prediction[0]==1 else "negative")

