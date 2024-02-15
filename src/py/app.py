import chainlit as cl 
import chromadb 
import requests 
import os
from chromadb.config import Settings
from datetime import datetime
from chromadb.utils import embedding_functions

class ConversationQueue:
    def __init__(self, max_length=3):
        self.queue = []
        self.max_length = max_length

    def add_conversation(self, question, response=None):
        if response:
            conversation = f'[INST] {question} [/INST] "{response}"'
        else:
            conversation = f'[INST] {question} [/INST] '
        if len(self.queue) == self.max_length:
            self.queue.pop(0)
        self.queue.append(conversation)

    def get_conversations(self):
        return ''.join(self.queue)

    def generate_prompt(self, doc, query):
        base_prompt = self.get_conversations()
        prompt = f"{base_prompt}[INST]Document:`{doc}`. Using the text in Document, answer the following question factually: {query}. Answer concisely at most in three sentences. Respond in a natural way, like you are having a conversation with a friend.[/INST]"
        return prompt
    
api_host = os.environ.get("API_HOST")
api_port = os.environ.get("API_PORT")
RAG_DB_PATH=os.environ.get("DB_PATH")
EMB_PATH=os.environ.get("EMB_PATH")
print(api_host,api_port,RAG_DB_PATH,EMB_PATH)
#parser = argparse.ArgumentParser(description="Process a file.") 
#path = '/mnt/efs/shared_fs/determined/rag_db/' 
#db = chromadb.PersistentClient(path=path) 
#collection = db.get_collection("HPE_press_releases") 
#EXTERNAL_IP="10.239.100.80"
#parser.add_argument("external_api_ip", default=None,help="External IP for API") 
#parser.add_argument("path_to_db", default='/mnt/efs/shared_fs/determined/rag_db',help="Path to vector db in container")
#parser.add_argument("host",default="0.0.0.0")
#parser.add_argument("port", default=8085)

#titan_url = "http://{}:8006/generate_stream".format(EXTERNAL_IP)
#args = parser.parse_args()


#path = '/mnt/efs/shared_fs/determined/rag_db/'
path = RAG_DB_PATH
settings = chromadb.get_settings()
settings.allow_reset = True
db = chromadb.PersistentClient(path=path,settings=settings)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMB_PATH, device="cpu"
)
collection = db.get_collection("HPE_press_releases",embedding_function=emb_fn)
#parser.add_argument("filepath", help="Path to the XML file")
titan_url = "http://{}:{}/generate_stream".format(api_host,api_port)
print("Titan URL: ",titan_url)

@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: cl.Message):
    # Send the final answer.

    msg = cl.Message(
        content="",
    )

    results = collection.query(query_texts=[message.content], n_results=5)
    
    date_strings = [i['Date'] for i in results['metadatas'][0]]
    # Your list of datetime strings
    # date_strings = ['2017-11-21', '2018-03-19', '2022-01-28', '2023-06-20', '2022-04-27']
    # Step 1: Parse strings into datetime objects
    date_objects = [datetime.fromisoformat(date_str) for date_str in date_strings]

    # Step 2: Extract year, month, and day
    formatted_dates = [dt.strftime('%Y-%m-%d') for dt in date_objects]

    # Step 3: Sort datetime objects while keeping track of original indices
    sorted_dates_with_indices = sorted(enumerate(zip(date_objects, formatted_dates)),
                                       key=lambda x: x[1][0], reverse=True)

    # Extract sorted dates and original indices
    sorted_dates = [date_str for _, (dt, date_str) in sorted_dates_with_indices]
    original_indices = [index for index, _ in sorted_dates_with_indices]

    # Print the result
    print("Sorted Dates:", sorted_dates)
    print("Original Indices:", original_indices)
    results_x = [results["documents"][0][original_indices[0]],
                 results["documents"][0][original_indices[1]],
                 results["documents"][0][original_indices[2]]]# get the first three document
    await show_sources(results)
    # print("results: ",results)`
    '''
    2/6/24 (Andrew): Add limit to ensure that any press release does not exceed >14k. 
    This assumes TitanML API deployed on A100
    This will decrease when API is deployed no T4.
    '''
    results2 = "\n".join(results_x)
    results2 = results2[:8500]
    print("len(results2): ",len(results2))
    print("results2: ",results2)
    prompt = f"[INST]`{results2}`. Using the above information, answer the following question: {message.content}.Answer factually and concisely, answer at most in three sentences. Respond in a natural way, like you are having a conversation with a friend.[/INST]"
    print("=========prompt=============: ")
    print(prompt)
    print("=========end_of_prompt=============")
    params={ 'generate_max_length': 1000,
        'no_repeat_ngram_size': 0,
        'sampling_topk': 50,
        'sampling_topp': 0.95,
        'sampling_temperature': 0.3,
        'repetition_penalty': 1.0}
    json = {"text": prompt,
            **params}
    response = requests.post(titan_url, json=json, stream=True)
    response.encoding = "utf-8"
    print("response: ", response.content)
    for text in response.iter_content(chunk_size=1, decode_unicode=True):
        await msg.stream_token(text)

    await msg.send()


async def show_sources(results):
    # elements = [
    #     cl.Text(name=f"Source {i+1}", content=r, display="inline")
    #     for i, r in enumerate(results)
    # ]
    # elements = [
    #     cl.Text(content=r, display="inline")
    #     for i, r in enumerate(results)
    # ]
    # await cl.Message(content="I found these sources:", elements=elements).send()
    # await cl.Message(content="", elements=elements).send()
    await cl.Message(content="").send()

