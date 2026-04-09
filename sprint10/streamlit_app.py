import pandas as pd
import streamlit as st
import altair as alt
from st_wordcloud import st_wordcloud

from tensorflow.keras import Sequential, preprocessing
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import load_model

MODEL_PATH="sprint10/word2vec.keras"
CSV_PATH="sprint10/MovieReview.csv"
#model.load_weights("word2vec.weights.h5")
model = load_model(MODEL_PATH)

# extraire la matrice d'embeddings
vectors = model.layers[0].trainable_weights[0].numpy()
import numpy as np
from sklearn.preprocessing import Normalizer

# définir les fonctions de similitude.
def dot_product(vec1, vec2):
    return np.sum((vec1*vec2))

def cosine_similarity(vec1, vec2):
    return dot_product(vec1, vec2)/np.sqrt(dot_product(vec1, vec1)*dot_product(vec2, vec2))

@st.cache_data
def find_closest(word_index, vectors = vectors, number_closest=10):
    list1=[]
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def compare(index_word1, index_word2, index_word3, vectors, number_closest):
    list1=[]
    query_vector = vectors[index_word1] - vectors[index_word2] + vectors[index_word3]
    normalizer = Normalizer()
    query_vector =  normalizer.fit_transform([query_vector], 'l2')
    query_vector= query_vector[0]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def print_closest(word2idx, idx2word, word, number=10):
    index_closest_words = find_closest(word2idx[word], vectors, number)
    for index_word in index_closest_words :
        print(idx2word[index_word[1]]," -- ",index_word[0])

st.title("Modèle Word2Vec with streamlit")
df = pd.read_csv(CSV_PATH)

import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

@st.cache_data
def preprocess_sentence(w,stop_words):
    w = unicode_to_ascii(w.lower().strip())
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!]+", " ", w)
    w = re.sub(r'\b\w{0,2}\b', '', w)

    # remove stopword
    mots = word_tokenize(w.strip())
    mots = [mot for mot in mots if mot not in stop_words]
    return ' '.join(mots).strip()

@st.cache_data
def init_tokenizer(data,num_words=10000):
    #nltk.download_shell()
    nltk.download('all')
    stop_words = stopwords.words('english')
    temp_data = data.apply(lambda x :preprocess_sentence(x, stop_words))
    tokenizer = preprocessing.text.Tokenizer(num_words)
    tokenizer.fit_on_texts(temp_data)
    # Créer une liste pour accéder rapidement au mot par index
    #idx2word = {v: k for k, v in tokenizer.word_index.items()}
    return tokenizer.word_index, tokenizer.index_word, tokenizer.word_counts, tokenizer.num_words

word2idx, idx2word, word_counts, vocab_size = init_tokenizer(data=df["review"],num_words=10000)

st.sidebar.title("Sommaire")

pages=["Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
    st.dataframe(df.head(10))
    st.write(df.shape)
    st.dataframe(df.describe())
    #st.write(moviesreview.find_closest('zombie'))

if page == pages[1] : 
    cols = st.columns([3, 2])
    with cols[1].container(border=True, height="stretch"):
        "### Répartition par sentiments"
        st.altair_chart(
            alt.Chart(df)
            .mark_bar()
            .encode(
                alt.X("sentiment:N"),
                #alt.X("sentiment:N").title("sentiment"),
                alt.Y("count()"),
                #alt.Y("count()").title("nb messages"),
                alt.Color("sentiment:N").title("sentiment"),
            )
            .configure_legend(orient="bottom")
        )
    with cols[0].container(border=True, height="stretch"):
        "### Top 10 des mots les plus utilisés"
        # Trier les mots par fréquence décroissante
        top_10_words = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)[:10]
        #st.write(top_10_words)
        # Afficher les 10 mots les plus fréquents
        list=[]
        for word, count in top_10_words:
            #print(f"{word}: {count}")
            list.append(f"{word} (score: {count:.2f})")
        st.html("<br>".join(list))
        #st_wordcloud(words, width=800, height=600)
        
if page == pages[2] : 
    #mot = st.text_input("Entrez un mot :", "king",  key="mot")
    # Accéder à la couche d’embedding
    embedding_layer = model.layers[0]
    embeddings = embedding_layer.get_weights()[0]
    #st.write("Taille de la matrice d'embedding :", embeddings.shape)
    mot = st.selectbox("Choose a category", word2idx)
    st.write(f"Selected: {mot}")

    # Vérifier si le mot est dans le vocabulaire
    if mot in word2idx:
        # Obtenir l’indice d’un mot
        word_index = word2idx[mot]
        #st.write("Index du mot {mot}:", word_index)
        if 0 < word_index < embeddings.shape[0]:
            # Récupérer le vecteur du mot
            vector = embeddings[word_index -1]
            #st.write("vector du mot {mot}:", vector)
            # Jouer avec les propriétés sémantiques
            similar_mots = find_closest(word_index, vectors = embeddings, number_closest=10)

            words=[{"text":mot,"value":100, "topic": "lol"}]
            list=[]
            for score, idx in similar_mots:
                #mot_sim = tokenizer.index_word[idx +1]
                mot_sim = idx2word[idx +1]
                words.append({"text": mot_sim, "value":(score * 100)})
                list.append(f"{mot_sim} (score: {score:.2f})")
            
            st.write(f"Les mots similaires à '{mot}':")
            cols = st.columns([3, 2])
            with cols[0].container(border=True, height="stretch"):
                st_wordcloud(words, width=600, height=400)
            with cols[1].container(border=True, height="stretch"):
                #for idx, score in similar_mots:
                st.html("<br>".join(list))
        else:
            st.write("Index hors limites ou mot non dans le vocabulaire.")
    else:
        st.write("Mot non trouvé dans le vocabulaire.")
