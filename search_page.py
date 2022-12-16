import streamlit as st
import pandas as pd
import nltk
nltk.download('punkt')
import gensim
import os
current_dir = os.getcwd()
init_model_dir = os.path.join(current_dir, "Data","updated_data_v01.pkl")
st.header('Anybody Can Classify ')

# df = pd.read_csv("C:/Users/marathoy/Desktop/AMAZON_FINAL_DEPLOY/Vinay_Kadel_BTI_upd.csv")
df =  pd.read_pickle(init_model_dir)
selected_text = st.text_input(
         "Type Keywords for your product",
    )

def fetch_poster(url):

    if(url == "nan"):
        full_path = 'https://upload.wikimedia.org/wikipedia/commons/d/de/Amazon_icon.png'
        return full_path
    else:
        full_path = url
        return full_path


def search_function(selected_text):
     
    # df =  pd.read_pickle("C:/Users/marathoy/Desktop/AMAZON_FINAL_DEPLOY/updated_data_v01.pkl")
    #   st.markdown(df.iloc[9])
    text_columns = ["Ruling reference", "Description", "Keywords","Expiry","Code","Images"]
    documents = df[text_columns].apply(lambda x: " ".join(x), axis=1)

# Tokenize the documents using NLTK's word_tokenize() function
    tokens = [nltk.word_tokenize(doc) for doc in documents]

# Create a dictionary from the tokens
    dictionary = gensim.corpora.Dictionary(tokens)

# Convert the tokens to a bag-of-words representation using the dictionary
    corpus = [dictionary.doc2bow(token) for token in tokens]

    query = selected_text

# Convert the query to a bag-of-words representation
    query_bow = dictionary.doc2bow(query.split())

# Perform semantic search using the bag-of-words representation
    results = gensim.similarities.MatrixSimilarity(corpus)[query_bow]

# Print the top 10 most similar documents to the query
    for i, similarity in sorted(enumerate(results), key=lambda x: -x[1]):
         st.markdown("******************************")
         st.image(fetch_poster(df.iloc[i].Images))
         st.markdown(f"{df.iloc[i].Keywords}")
         st.markdown(f"{df.iloc[i].Description}")
         st.markdown(f"Document {i}: {df.iloc[i].Code}")
         st.markdown("******************************")

if st.button('Show Results'):
    search_function(selected_text)
    




# text_columns = ["Ruling reference", "Description", "Keywords","Expiry","Code","Images"]
# documents = df[text_columns].apply(lambda x: " ".join(x), axis=1)

# # Tokenize the documents using NLTK's word_tokenize() function
# tokens = [nltk.word_tokenize(doc) for doc in documents]

# # Create a dictionary from the tokens
# dictionary = gensim.corpora.Dictionary(tokens)

# # Convert the tokens to a bag-of-words representation using the dictionary
# corpus = [dictionary.doc2bow(token) for token in tokens]

# # Define a query to search for
# query = selected_text

# # Convert the query to a bag-of-words representation
# query_bow = dictionary.doc2bow(query.split())

# # Perform semantic search using the bag-of-words representation
# results = gensim.similarities.MatrixSimilarity(corpus)[query_bow]

# # Print the top 10 most similar documents to the query
# for i, similarity in sorted(enumerate(results), key=lambda x: -x[1])[:10]:
#     st.caption(f"Document {i}: {documents[i]}")

# def fetch_poster(url):
#         if(url == "nan"):
#             full_path = 'https://upload.wikimedia.org/wikipedia/commons/d/de/Amazon_icon.png'
#             return full_path
#         else:
#              full_path = url
#              return full_path


# def search_function(selected_text):
     
#     df =  pd.read_pickle("C:/Users/marathoy/Desktop/Amazon_Cops_V02_Backup/updated_data_v01.pkl")
#     text_columns = ["Ruling reference", "Description", "Keywords","Expiry","Code","Images"]
#     documents = df[text_columns].apply(lambda x: " ".join(x), axis=1)

# # Tokenize the documents using NLTK's word_tokenize() function
#     tokens = [nltk.word_tokenize(doc) for doc in documents]

# # Create a dictionary from the tokens
#     dictionary = gensim.corpora.Dictionary(tokens)

# # Convert the tokens to a bag-of-words representation using the dictionary
#     corpus = [dictionary.doc2bow(token) for token in tokens]

#     query = selected_text

# # Convert the query to a bag-of-words representation
#     query_bow = dictionary.doc2bow(query.split())

# # Perform semantic search using the bag-of-words representation
#     results = gensim.similarities.MatrixSimilarity(corpus)[query_bow]

# # Print the top 10 most similar documents to the query
#     for i, similarity in sorted(enumerate(results), key=lambda x: -x[1]):
#          st.caption(f"Document {i}: {documents[i]}")




# st.header('Anybody Can Classify ')

# selected_text = st.text_input(
#         "Type Keywords for your product",
#     )

# if st.button('Show Results'):
        
#          recommended_results = search_function(selected_text)
#          st.caption(recommended_results)
#         #  with st.container():
#         #     container = st.expander("SEE SEARCH RESULTS 1")
#         #     with container:
#         #             cols = st.columns(7)
#         #             # cols[0].write('###### Similarity Score')
#         #             # cols[0].caption("model_desc")
#         #             for img, col, cod in zip(recommended_results.iloc[:7].Images, cols[:7], recommended_results.iloc[:7].Code):
#         #                 with col:
#         #                     # st.caption('{}'.format(score))
#         #                     st.image(fetch_poster(img), use_column_width=True)
            
#         #                     # if model == 'Similar items based on text embeddings':
#         #                     st.caption(cod)
#         #                     # st.caption(desc)
        
#         #  with st.container():
#         #     container = st.expander("SEE SEARCH RESULTS 2")
#         #     with container:
#         #             cols = st.columns(7)
#         #             # cols[0].write('###### Similarity Score')
#         #             # cols[0].caption("model_desc")
#         #             for img, col, cod , desc in zip(recommended_results.iloc[7:14].Images, cols[0:7], recommended_results.iloc[7:14].Code , recommended_results.iloc[7:14].Ruling_reference):
#         #                 with col:
#         #                     # st.caption('{}'.format(score))
#         #                     st.image(fetch_poster(img), use_column_width=True)
            
#         #                     # if model == 'Similar items based on text embeddings':
#         #                     st.caption(cod)
#         #                     st.caption(desc)
