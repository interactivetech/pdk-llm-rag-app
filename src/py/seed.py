import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
import argparse

from chromadb.utils import embedding_functions

from chromadb.api.types import (
    Document,
    Documents,
    Embedding,
    Image,
    Images,
    EmbeddingFunction,
    Embeddings,
    is_image,
    is_document,
)
from tqdm import tqdm
from typing import Any, Dict, List, Mapping, Union, cast

#sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
import os

def main(csv_path,path_to_db):
    settings = chromadb.get_settings()
    settings.allow_reset = True
    print(f"creating/resetting db at {path_to_db}...")
    db = chromadb.PersistentClient(path=path_to_db, settings=settings)
    print("Done!")
    db.reset()
    model_path=args.emb_model_path
    print("Loading {}...".format(model_path))
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_path, device="cpu"
    )
    print("Done!")
    #model_path='/mnt/efs/shared_fs/determined/all-MiniLM-L6-v2/'
    #emb_fn.models['all-MiniLM-L6-v2'].save(model_path)
    #print("Model saved at:{} ".format(model_path))
    collection = db.create_collection(name="HPE_press_releases", embedding_function=emb_fn)

    # data_path = csv_path
    # df = pd.read_csv(data_path)
    # LEN=df.shape[0]
    # collection.add(
    #     documents=[df.iloc[i]['Content'] for i in range(LEN)],
    #     metadatas=[{'Title':df.iloc[i]['Title'],'Content':df.iloc[i]['Content'],'Date':df.iloc[i]['Date']} for i in range(LEN)],
    #     ids=[f'id{str(i)}' for i in range(LEN)]
    # )
    '''Chunking strategy'''
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=100)
    def chunk_text(text_splitter,text):
        return text_splitter.split_text(text)

    data_path = csv_path
    df = pd.read_csv(data_path)
    LEN=df.shape[0]
    chunks = []
    inds = []
    print("Number of docs: ",LEN)
    for i in tqdm(range(LEN)):
        ch = chunk_text(text_splitter,df.iloc[i]['Content'])
        chunks+=ch
        inds+=[i]*len(ch)
    print("Number of chunks: ",len(chunks))

    for i in tqdm(range(len(chunks))):
        collection.add(
            documents=[chunks[i]],
            metadatas=[{'Title':df.iloc[inds[i]]['Title'],'Content':df.iloc[inds[i]]['Content'],'Date':df.iloc[inds[i]]['Date']}],
            ids=[f'id{str(i)}']
        )

    query = "How were HPE's earnings in 2022?"
    results = collection.query(query_texts=[query], n_results=5)
    print("query: ",query, "results: ",results['documents'])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_db',type=str, default='/mnt/efs/shared_fs/determined/rag_db/', help='path to csv containing press releases')
    parser.add_argument('--emb_model_path',type=str, default=None, help='path to locally saved sentence transformer model')

    parser.add_argument('--csv_path',type=str, default='/mnt/efs/shared_fs/determined/nb_fs/dev-llm-rag-app/data/HPE_2023_Press_Releases_qa.csv', help='path to csv containing press releases')
    args = parser.parse_args()
    main(args.csv_path,args.path_to_db)
