try:
    import pandas as pd
    import pickle
    import nltk
    import string
    from nltk.stem import WordNetLemmatizer
    from nltk import word_tokenize
    from nltk.corpus import stopwords
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import classification_report
    import sys
    import subprocess




except:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'nltk'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sklearn'])
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')



class EmotionDetection:
    def __init__(self):
        self.df = None
        self.replace_words = pickle.load(open("save_models_and_dataset\\rep_words.pkl","rb"))
        self.stop_words = list(stopwords.words("english"))
        self.stop_words.remove("not")
        self.lemmatizer = WordNetLemmatizer()
        
    def read_file(self):

        self.df = pd.read_csv("save_models_and_dataset\\emotion_dataset_raw.csv")
    
    def clean_data(self,text):
        text = str(text)

        text = text.lower()

        for key, value in self.replace_words.items():
            text = re.sub(key, value,text)

        text = re.sub('@[^\s]+',' ', text)

        text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', ' ',text)

        text = re.sub(r'#([^\s]+)', r'\1',text)

        text = text.translate(str.maketrans('', '', string.punctuation))

        text = re.sub(r'\d+'," ", text)
        removed_stop_lemma = []
        tokens = word_tokenize(text)
        for i in range(len(tokens)):
            if tokens[i] not in self.stop_words:
                if len(tokens[i]) > 2:
                    removed_stop_lemma.append(self.lemmatizer.lemmatize(tokens[i]))
        return " ".join(removed_stop_lemma)
    
    def Vectorization(self):
        vect_tfidf = TfidfVectorizer()
        return vect_tfidf
    
    def get_model(self):
        model = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(30,), verbose=True,max_iter=100,activation="relu",learning_rate="invscaling"))
        return model

    def training(self):
        self.read_file()

        self.df['Text'] = self.df['Text'].apply(lambda x: self.clean_data(x))
        tfidf = self.Vectorization()
        X_matrix = tfidf.fit_transform(self.df["Text"])
        X = X_matrix
        y = self.df['Emotion'].tolist()

        pickle.dump(tfidf, open("save_models_and_dataset\\tfidf_weights.pkl","wb"))
        model = self.get_model()
        model.fit(X,y)

        pickle.dump(model,open("save_models_and_dataset\\model_weights.pkl","wb"))
        print(classification_report(model.predict(X),y))
    
    def predict_text(self,text):

        vect_tfidf = pickle.load(open("save_models_and_dataset\\tfidf_weights.pkl","rb"))

        model = pickle.load(open("save_models_and_dataset\\model_weights.pkl","rb"))
        clean_text = self.clean_data(text)
        x_vector = vect_tfidf.transform([clean_text])
        emotion = model.predict(x_vector)[0]
        prob = max(model.predict_proba(x_vector)[0])
        return emotion,prob


if __name__ == "__main__":
    inst = EmotionDetection()

    text = input("Enter the text: ")
    emotion, prob = inst.predict_text(text)
    print("Emotion: {}\nScore: {}".format(emotion,prob))



