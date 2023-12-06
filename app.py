from http.client import HTTPException

from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from datetime import datetime
import datetime
import plaid
from plaid.api import plaid_api
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.model.products import Products
from plaid.model.country_code import CountryCode
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions
from pathlib import Path
from langchain.document_loaders import JSONLoader
from collections import Counter


from typing import List
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import openai
from openai.error import OpenAIError
from dotenv import load_dotenv

# LLM and database imports
from langchain.document_loaders import UnstructuredHTMLLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import requests
import pinecone
import sqlite3
import pandas as pd
import os

# id counter
count = 0

load_dotenv()

# load API keys
PINECONE_API_KEY = ''
PINECONE_API_ENV = ''

OPENAI_API_KEY = ""
PLAID_CLIENT_ID = ''
PLAID_SANDBOX = ''
user_name = ""
CHUNK_SIZE = 2000

# set up openai embedding
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "personal-database"  # put in the name of your pinecone index here
index = pinecone.Index('personal-database')

# set up plaid api
host = plaid.Environment.Sandbox
configuration = plaid.Configuration(
    host=host,
    api_key={
        'clientId': PLAID_CLIENT_ID,
        'secret': PLAID_SANDBOX,
    }
)
api_client = plaid.ApiClient(configuration)
client = plaid_api.PlaidApi(api_client)
access_token = None
item_id = None


# initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods
    allow_headers=["*"],  # allow all headers
)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")
openai.api_key = os.getenv("OPENAI_API_KEY")

# designate llm model for chat bot
llm = ChatOpenAI(model="gpt-4",temperature=1, openai_api_key=OPENAI_API_KEY)
text_field = 'text'
vector_store = Pinecone.from_existing_index(index_name=index_name,embedding=embeddings,text_key=text_field)

# implement conversational memory
conversation_memory = ConversationBufferMemory()
template = """
You are a helpful and honest AI assistant that will always help human. Consider the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) in your response:
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""

# add history, context, question in the prompt
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)

# set rag pipeline with retriever
rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=vector_store.as_retriever(search_kwargs={"k": 8}), # ADJUST NUMBER OF RAG RESULT
    verbose=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": prompt,
        "memory": ConversationBufferMemory(
            memory_key="history",
            input_key="question"),
    }
)


class TokenRequest(BaseModel):
    public_token: str

class Message(BaseModel):
    role: str
    content: str

class ServiceSelection(BaseModel):
    services: List[str]

# data source selection page
selected_services = []
@app.post("/selected_services")
async def handle_selected_services(selection: ServiceSelection):
    global selected_services
    selected_services = selection.services

# retrieve gmail and upload to database
@app.post("/email_services")
async def email_services(selection: ServiceSelection):
    if "service2" not in selection.services:
        if "service3" in selection.services:
            extract_gmail_api_start()
        if "service1" in selected_services:
            insert_bookmark_to_pinecone()
        if "service4" in selected_services:
            insert_pdf_to_pinecone()

# if no update on email or transactions
@app.post("/no_email_nor_transaction")
async def no_email_nor_transaction(selection: ServiceSelection):
    if "service2" not in selected_services and "service3" not in selected_services:
        if "service1" in selected_services:
            insert_bookmark_to_pinecone()
        if "service4" in selected_services:
            insert_pdf_to_pinecone()

# landing page
@app.get("/", response_class=HTMLResponse)
def get_homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# dashboard page
@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard(request: Request):
    return templates.TemplateResponse("index2.html", {"request": request})

# chat page
@app.get("/chat")
def get_index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

# set_access token for plaid api
@app.post("/set_access_token")
async def set_access_token(token_request: TokenRequest):
    public_token = token_request.public_token
    global access_token
    global item_id
    try:
        public_token_exchange_request = ItemPublicTokenExchangeRequest(
            public_token=public_token
        )
        response = client.item_public_token_exchange(public_token_exchange_request)
        access_token = response['access_token']
        item_id = response['item_id']
        if "service2" in selected_services:
            make_json()
            insert_transaction_to_pinecone()
            if "service3" in selected_services:
                extract_gmail_api_start()
            if "service1" in selected_services:
                insert_bookmark_to_pinecone()
            if "service4" in selected_services:
                insert_pdf_to_pinecone()

        return JSONResponse(content=jsonable_encoder(response.to_dict()))
    except plaid.ApiException as e:
        return json.loads(e.body)


# create link token
@app.post("/create_link_token")
async def create_link_token():
    link_token_request = LinkTokenCreateRequest(
        products=[Products("auth"), Products("transactions")],
        client_name="Banking_Proto",
        country_codes=[CountryCode('US')],
        language='en',
        user=LinkTokenCreateRequestUser(
            client_user_id='user_good'
        )
    )
    response = client.link_token_create(link_token_request)
    return JSONResponse(content=jsonable_encoder(response.to_dict()))

# get transaction from plaid api
def get_transactions_from_plaid():
    request = TransactionsGetRequest(
        access_token=access_token,
        start_date=datetime.date(2023, 1, 1),
        end_date=datetime.date(2023, 6, 1),
    )
    response = client.transactions_get(request)
    transactions = response['transactions']

    # Iterating through paginated transactions in case there is more
    while len(transactions) < response['total_transactions']:
        request = TransactionsGetRequest(
            access_token=access_token,
            start_date=datetime.date(2023, 1, 1),
            end_date=datetime.date(2023, 6, 1),
            options=TransactionsGetRequestOptions(
                offset=len(transactions)
            )
        )
    response = client.transactions_get(request)
    transactions.extend(response['transactions'])

    return response.to_dict()


# dump transactions to json
async def process_transactions():
    make_json()
    insert_transaction_to_pinecone()

# helper function for json read
def read_ith_entry_from_json(file_path, i):
    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    # Access the ith entry
    ith_entry = data["chunk"][i]
    return ith_entry

# helper function for json write
def write_to_json(file_path, data):
    # Write data to a JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# determine the type of feedback for evidence
def find_type(metadata):
    source = metadata['source']

    try:
        if source[-4:] == ".pdf":
            metadata_type = 'pdf'
            return metadata_type
    except Exception as e:
        pass
    try:
        sequence = metadata['seq_num']
        if source[-10:] == "email.json":
            metadata_type = "email"
            return metadata_type
        else:
            metadata_type = "transaction"
            return metadata_type
    except Exception as e:
        pass
    try:
        metadata_type = 'link'
        return metadata_type
    except Exception as e:
        pass


# generate evidence and logic-checker
def generate_evidence(evidence, response):
    counter_evidence_prompt = f"Given the evidence '{evidence}', is this statement accurate: '{response['result']}'? Provide explanation within 30 words. " \
                              "Return in json format in the form of {\"trust_type\":\"yes/no\", \"explanation\": \"explanation\"}"

    counter_evidence = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": counter_evidence_prompt}],
        max_tokens=100
    )

    counter_evidence_dict =counter_evidence["choices"][0]["message"]["content"]

    return counter_evidence_dict


# differentiate command type (avoid redundant checks on greetings)
def check_greetings(response):
    counter_evidence_prompt = f"Can you check whether this is an answer to greetings (like how are you), chitchat, giving/replying to thanks or include anything saying that I'm just an AI? '{response['result']}'? " \
                              "Return only yes or no."
    counter_evidence = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": counter_evidence_prompt}],
        max_tokens=100
    )
    answer = counter_evidence["choices"][0]["message"]["content"]
    print("yesyes")

    return answer.lower().strip() == "yes"


# find the most frequent metadata
def most_common(metadata_types):
    count = Counter(metadata_types)
    most_common = sorted(count, key=lambda x: (-count[x], metadata_types.index(x)))[0]
    return most_common

# filter the most frequent metadata
def filter_by_most_common(metadata_container, metadata_types, source_type):
    filtered_metadata = [metadata for metadata, type_ in zip(metadata_container, metadata_types) if type_ == source_type]
    return filtered_metadata

# filter and save transaction and dump to json
def filter_and_save_transactions(file_path, output_file_name, metadata_container_filtered):
    with open(file_path, 'r') as file:
        data = json.load(file)
    seq_nums = {int(item['seq_num']) - 1 for item in metadata_container_filtered}
    if "chunk" in data:
        filtered_transactions = [data["chunk"][i] for i in seq_nums if i < len(data["chunk"])]
    else:
        filtered_transactions = []

    output_file_name_real = "./static/" + output_file_name

    with open(output_file_name_real, 'w') as outfile:
        json.dump(filtered_transactions, outfile, indent=4)

    return {output_file_name: output_file_name_real}

# find the title for json data entry
def find_title(metadata_container_filtered):
    titles_paths = {source.split('/')[-1].replace('.pdf', ''): 'static/' + source for item in metadata_container_filtered for source in [item['source']]}
    json_output = json.dumps(titles_paths, indent=None)
    return json_output

# process the format of response
def create_response_object(response_result, metadata_type, source_value, counter_evidence_response):
    return {
        "response": response_result,
        "source": {
            "source_type": metadata_type,
            "source_value": source_value
        },
        "trust": counter_evidence_response
    }

# generate chat message
async def generate(messages: List[Message], model_type: str):
    global rag_pipeline
    try:
        current_query = messages[-1].content
        if current_query[0:3] == "///":
            custom_upsert(current_query[3:])
            response = "Glad to remember what you said."
            yield response
        else:
            response = rag_pipeline(current_query)

            if(check_greetings(response)):
                source_type = "greeting"
                source_value = "greeting"
                explain = '{"trust_type": "yes", "explanation": "This is a greeting."}'
                is_greeting = create_response_object(response['result'], source_type, source_value, explain)
                yield is_greeting
                return

            results = vector_store.similarity_search(current_query, k=8)
            evidence = ""
            metadata_container = []
            metadata_type = []
            for result in results:
                content = result.page_content
                evidence += content
                metadata = result.metadata
                metadata_container.append(metadata)
                metadata_type.append(find_type(metadata))

            source_type = most_common(metadata_type)
            metadata_container_filtered = filter_by_most_common(metadata_container, metadata_type, source_type)
            if source_type == "transaction":
                source_value = filter_and_save_transactions('transactions.json', 'my_transaction.json', metadata_container_filtered)
            if source_type == "email":
                source_value = filter_and_save_transactions('email.json', 'my_email.json', metadata_container_filtered)
            if source_type == "pdf":
                source_value = find_title(metadata_container_filtered)
            if source_type == "link":
                source_value = {item['title']: item['url'] for item in metadata_container_filtered}

            counter_evidence_response = generate_evidence(evidence, response)
            final_response = create_response_object(response['result'], source_type, source_value, counter_evidence_response)

            print(final_response)

            yield final_response

    except OpenAIError as e:
        yield f"{type(e).__name__}: {str(e)}"


# GPT request
class Gpt4Request(BaseModel):
    messages: List[Message]
    model_type: str

@app.post("/gpt4")
async def gpt4(request: Gpt4Request):
    try:
        async for response in generate(request.messages, request.model_type):
            assistant_response = response
        return assistant_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# dump transaction to json
def make_json():
    data = get_transactions_from_plaid()

    transactions_list = data["transactions"]
    TRANSACTION_DICT_OUT = dict()
    TRANSACTION_DICT = dict()
    TRANSACTION_LIST = []
    for i in range(len(transactions_list)):
        TRANSACTION_DICT["name"] = transactions_list[i]["name"]
        temp_category = ""
        if transactions_list[i]["category"] != None:
            for cat in transactions_list[i]["category"]:
                temp_category += cat
                temp_category += ", "
        TRANSACTION_DICT["category"] = temp_category
        TRANSACTION_DICT["amount"] = str(transactions_list[i]["amount"])
        date_ = transactions_list[i]["date"]
        TRANSACTION_DICT["date"] = date_.strftime("%m/%d/%Y, %H:%M:%S")
        TRANSACTION_DICT["merchant_name"] = transactions_list[i]["merchant_name"]
        TRANSACTION_DICT["payment_channel"] = transactions_list[i]["payment_channel"]
        TRANSACTION_LIST.append(TRANSACTION_DICT)
        TRANSACTION_DICT = dict()

    TRANSACTION_DICT_OUT["chunk"] = TRANSACTION_LIST

    with open("transactions.json", "w") as outfile:
        json.dump(TRANSACTION_DICT_OUT, outfile)

# upsert transaction data to pinecone database
def insert_transaction_to_pinecone():
    file_path = os.path.join('transactions.json')

    loader = JSONLoader(
        file_path=file_path,
        jq_schema='.chunk[]',
        text_content=False)

    data = loader.load()

    no_of_transactions = len(data)

    def upsert_transaction_to_pinecone(data):
        global count, embeddings, index
        ### Chunk your data up into smaller documents
        CHUNK_SIZE = 2000
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
        texts = text_splitter.split_documents([data])

        # populate vectors
        batch_num = len(texts)
        vectors = []
        for i in range(batch_num):
            batch = texts[i].page_content
            ids = "transaction_" + str(count)
            embeds = embeddings.embed_query(batch)
            # get metadata to store in Pinecone
            metadata = texts[i].metadata
            metadata["text"] = batch
            # add to Pinecone
            vectors.append((ids, embeds, metadata))
            count += 1
        # Upsert into Pinecone index
        try:
            index.upsert(vectors=vectors)
        except Exception as e:
            pass

    for no in range(no_of_transactions):
        upsert_transaction_to_pinecone(data[no])


#==========================================================================================
#==========================================================================================
#==========================================================================================

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import os.path
import base64
import json

EMAIL_LIST = []
EMAIL_DICT = dict()
EMAIL_DICT_OUT = dict()

# gmail authentication
def gmail_authenticate():
    SCOPES = ['https://mail.google.com/']
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

# clean text
def clean(text):
    # clean text for creating a folder
    return "".join(c if c.isalnum() else "_" for c in text)


# parse email data
def parse_parts(service, parts, folder_name, message):
    """
    Utility function that parses the content of an email partition
    """
    if parts == None:
        EMAIL_DICT['content'] = ''
    if parts:
        for part in parts:
            filename = part.get("filename")
            mimeType = part.get("mimeType")
            body = part.get("body")
            data = body.get("data")
            file_size = body.get("size")
            part_headers = part.get("headers")
            if part.get("parts"):
                # recursively call this function when we see that a part
                # has parts inside
                parse_parts(service, part.get("parts"), folder_name, message)
            if data:
                text = base64.urlsafe_b64decode(data).decode()
                if mimeType != "text/html":
                    EMAIL_DICT['content'] = text


# read email message
def read_message(service, number_of_emails):
    """
    This function takes Gmail API `service` and the given `message_id` and does the following:
        - Downloads the content of the email
        - Prints email basic information (To, From, Subject & Date) and plain/text parts
        - Creates a folder for each email based on the subject
        - Downloads text/html content (if available) and saves it under the folder created as index.html
        - Downloads any file that is attached to the email and saves it in the folder created
    """
    global EMAIL_DICT
    result = service.users().messages().list(userId='me', maxResults=number_of_emails).execute()
    messages = result.get("messages")
    for i, temp in enumerate(messages):

        msg = service.users().messages().get(userId='me', id=temp['id'], format='full').execute()
        # parts can be the message body, or attachments
        payload = msg['payload']
        headers = payload.get("headers")
        parts = payload.get("parts")
        folder_name = "email"
        has_subject = False
        if headers:
            # this section prints email basic info & creates a folder for the email
            for header in headers:
                name = header.get("name")
                value = header.get("value")
                if name.lower() == 'from':
                    EMAIL_DICT['from'] = str(value)
                if name.lower() == "to":
                    EMAIL_DICT['to'] = str(value)
                if name.lower() == "subject":
                    # make our boolean True, the email has "subject"
                    has_subject = True
                    # make a directory with the name of the subject
                    folder_name = clean(value)
                    EMAIL_DICT['subject'] = str(value)
                if name.lower() == "date":
                    EMAIL_DICT['date'] = str(value)
        parse_parts(service, parts, folder_name, temp)
        EMAIL_LIST.append(EMAIL_DICT)
        EMAIL_DICT = dict()

# extract gmail api
def extract_gmail_api_start():
    service = gmail_authenticate()
    read_message(service, 20)
    EMAIL_DICT_OUT['chunk'] = EMAIL_LIST
    with open("email.json", "w") as outfile:
        json.dump(EMAIL_DICT_OUT, outfile)

    insert_email_to_pinecone()

# upsert email to pinecone database
def insert_email_to_pinecone():
    file_path = os.path.join('email.json')
    data = json.loads(Path(file_path).read_text())


    loader = JSONLoader(
        file_path=file_path,
        jq_schema='.chunk[]',
        text_content=False)

    data = loader.load()
    no_of_emails = len(data)
    # email_counter = 0

    def upsert_email_json_to_pinecone(data):
        global count, index, embeddings

        ### Chunk your data up into smaller documents
        CHUNK_SIZE = 2000
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
        texts = text_splitter.split_documents([data])


        # populate vectors
        batch_num = len(texts)
        vectors = []
        for i in range(batch_num):
            batch = texts[i].page_content
            ids = "email_" + str(count)
            embeds = embeddings.embed_query(batch)
            # get metadata to store in Pinecone
            metadata = texts[i].metadata
            metadata["text"] = batch
            # add to Pinecone
            vectors.append((ids, embeds, metadata))
            count += 1

        # Upsert into Pinecone index
        try:
            index.upsert(vectors=vectors)
        except Exception as e:
            pass

    for no in range(no_of_emails):
        upsert_email_json_to_pinecone(data[no])


#===========================================================================
#===========================================================================
#===========================================================================


load_dotenv()

CHUNK_SIZE = 2000


########################################################################
##  LOAD BOOKMARK
########################################################################

# upsert bookmark/ web history to pinecone database
def insert_bookmark_to_pinecone():
    ### Load your bookmark data
    CHROME_BOOKMARK_PATH = f"/Users/{user_name}/Library/Application Support/Google/Chrome/Default/History"
    # print(CHROME_BOOKMARK_PATH)
    conn = sqlite3.connect(CHROME_BOOKMARK_PATH)
    # Query the database to retrieve data
    query = "SELECT * FROM urls"
    df_bookmark = pd.read_sql_query(query, conn)
    # Close the connection
    conn.close()

    ### create unique pinecone vector ids
    # count = 0
    html_id_header = "bookmark_html_"

    # helper function to upload one html file in the df
    def upload_html_to_pinecone_helper(row):
        global count,html_id_header
        global index, embeddings
        url = row["url"]
        query_parameters = {"downloadformat": "html"}
        response = requests.get(url, params=query_parameters)
        file_name = f"{row['id']}.html"

        # Define the subfolder and file name
        subfolder = 'html_file'
        full_path = os.path.join(subfolder, file_name)
        # Open the file and write the content
        with open(full_path, mode="wb") as file:
            file.write(response.content)

        loader = UnstructuredHTMLLoader(full_path)
        data = loader.load()

        ### Chunk your data up into smaller documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
        texts = text_splitter.split_documents(data)

        # populate vectors
        batch_num = len(texts)
        vectors = []
        for i in range(batch_num):
            batch = texts[i].page_content
            ids = "bookmark_html_" + str(count)
            embeds = embeddings.embed_query(batch)
            # get metadata to store in Pinecone
            metadata = texts[i].metadata
            metadata["text"] = batch
            metadata["url"] = row["url"]
            metadata["title"] = row["title"]
            metadata["visit_count"] = row["visit_count"]
            metadata["last_visit_time"] = row["last_visit_time"]
            # add to Pinecone
            vectors.append((ids, embeds, metadata))
            count += 1
        # Upsert into Pinecone index
        try:
            index.upsert(vectors=vectors)
        except Exception as e:
            pass
        # Done Uploading

    ### Upload html process
    page_count = 0
    for i, row in df_bookmark.iterrows():
        page_count += 1
        if page_count > 10:
            break
        page_id = row["id"]
        url = row["url"]
        title = row["title"]
        visit_count = row["visit_count"]
        last_visit_time = row["last_visit_time"]
        upload_html_to_pinecone_helper(row)

# upsert local documents to pinecone database
def insert_pdf_to_pinecone():
    path = "pdf"
    dir_list = os.listdir(path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)


    for file_name in dir_list:
        global count
        # populate vectors
        vectors = []
        full_path = os.path.join(path, file_name)
        loader = PyPDFLoader(full_path)
        try:
            data = loader.load()
        except Exception as e:
            continue
        texts = text_splitter.split_documents(data)
        batch_num = len(texts)
        for i in range(batch_num):
            batch = texts[i].page_content
            ids = "pdf_"+str(count)
            embeds = embeddings.embed_query(batch)
            # get metadata to store in Pinecone
            metadata = texts[i].metadata
            metadata["text"] = batch
            # source, page, text

            # add to Pinecone
            vectors.append((ids, embeds, metadata))
            count += 1
        # Upsert into Pinecone index
        try:
            index.upsert(vectors=vectors)
        except Exception as e:
            pass

# define a custom upsert for users to input note
def custom_upsert(data):
    global count
    global index, embeddings
    vectors = []
    batch = data
    ids = "note_" + str(count)
    embeds = embeddings.embed_query(batch)
    # get metadata to store in Pinecone
    metadata = {}
    metadata["text"] = batch
    metadata["type"] = "note"
    # add to Pinecone
    vectors.append((ids, embeds, metadata))
    count += 1
    try:
        index.upsert(vectors=vectors)
    except Exception as e:
        pass


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
