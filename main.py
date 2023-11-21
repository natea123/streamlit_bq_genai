# First
import streamlit as st
import google.auth

credentials, project_id = google.auth.default()
PROJECT_ID = ""
MODEL = "text-bison@001"
REGION = 'us-central1'
EXPERIMENT = 'bq-citibikes'
SERIES = 'applied-genai'

BQ_PROJECT = 'bigquery-public-data'
BQ_DATASET = 'new_york'
BQ_TABLES = ['citibike_trips', 'citibike_stations']

from os.path import basename
from typing import Any, Mapping, List, Dict, Optional, Tuple, Sequence, Union
import json, re
from google.protobuf.json_format import MessageToDict


import vertexai.language_models
from google.cloud import aiplatform
from google.cloud import bigquery

import pandas as pd


# vertex ai clients
vertexai.init(project = PROJECT_ID, location = REGION)
aiplatform.init(project = PROJECT_ID, location = REGION)

# bigquery client
bq = bigquery.Client(project = PROJECT_ID)


question = "What are the top 5 stations that had the most rides by riders over the age of 40 and the ride count?"

# create links to model: embedding api and text generation
textgen_model = vertexai.language_models.TextGenerationModel.from_pretrained('text-bison')
codegen_model = vertexai.language_models.CodeGenerationModel.from_pretrained('code-bison')

query = f"""
    SELECT *
    FROM `{BQ_PROJECT}.{BQ_DATASET}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`
    WHERE table_name in ({','.join([f'"{table}"' for table in BQ_TABLES])})
"""
#print(query)
schema_columns = bq.query(query = query).to_dataframe()

######
def initial_query(question, schema_columns):

    # code generation model
    codegen_model = vertexai.language_models.CodeGenerationModel.from_pretrained('code-bison')

    # initial request for query:
    query_response = codegen_model.predict(f"""Write a Google SQL query for BigQuery that answers the following question while correctly refering to BigQuery tables and the needed column names.  When joining tables use coersion to ensure all join columns are the same data type. Output column names should include the units when applicable.  Tables should be refered to using a fully qualified name include project and dataset along with table name.

Question: {question}

Context:
{schema_columns.to_markdown(index = False)}
""")

    # extract query from response
    if query_response.text.find("```") >= 0:
        query = query_response.text.split("```")[1]
        if query.startswith('sql'):
            query = query[3:]
        print('First try:\n', query)
    else:
        print('No query provided (first try) - unforseen error, printing out response to help with editing this funcntion:\n', query_response.text)

    return query


def codechat_start(question, query, schema_columns):

    # code chat model
    codechat_model = vertexai.language_models.CodeChatModel.from_pretrained('codechat-bison@001')

    # start a code chat session and give the schema for columns as the starting context:
    codechat = codechat_model.start_chat(
        context = f"""This session is trying to troubleshoot a Google BigQuery SQL query that is being writen to answer a question.
Question: {question}

BigQuery SQL Query: {query}

information_schema:
{schema_columns.to_markdown(index = False)}

Instructions:
As the user provides versions of the query and the errors returned by BigQuery, offer suggestions that fix the errors but it is important that the query still answer the original question.
"""
    )

    return codechat


def fix_query(query, max_fixes):

    # iteratively run query, and fix it using codechat until success (or max_fixes reached):
    fix_tries = 0
    answer = False
    while fix_tries < max_fixes:
        if not query:
            return
        # run query:
        query_job = bq.query(query = query)
        # if errors, then generate repair query:
        if query_job.errors:
            fix_tries += 1

            if fix_tries == 1:
                codechat = codechat_start(question, query, schema_columns)

            # construct hint from error
            hint = ''
            for error in query_job.errors:
                # detect error message
                if 'message' in list(error.keys()):
                    # detect index of error location
                    if error['message'].rindex('[') and error['message'].rindex(']'):
                        begin = error['message'].rindex('[') + 1
                        end = error['message'].rindex(']')
                        # verify that it looks like an error location:
                        if end > begin and error['message'][begin:end].index(':'):
                            # retrieve the two parts of the error index: query line, query column
                            query_index = [int(q) for q in error['message'][begin:end].split(':')]
                            hint += query.split('\n')[query_index[0]-1].strip()
                            break

            # construct prompt to request a fix:
            fix_prompt = f"""This query:\n{query}\n\nReturns these errors:\n{query_job.errors}\n\nPlease fix it and make sure it matches the schema."""

            #if hint != '':
            #    fix_prompt += f"""Hint, the error appears to be in this line of the query:\n{hint}"""

            query_response = codechat.send_message(fix_prompt)
            query_response = codechat.send_message('Respond with only the corrected query that still answers the question as a markdown code block.')
            if query_response.text.find("```") >= 0:
                query = query_response.text.split("```")[1]
                if query.startswith('sql'):
                    query = query[4:]
                print(f'Fix #{fix_tries}:\n', query)
            # response did not have a query????:
            else:
                query = ''
                print('No query in response...')

        # no error, break while loop
        else:
            break

    return query, query_job, fix_tries, codechat

def answer_question(question, query_job):

    # text generation model
    textgen_model = vertexai.language_models.TextGenerationModel.from_pretrained('text-bison')

    result = query_job.to_dataframe()
    # answer question
    question_prompt = f"""Answer the following question using the provided context.  Note that the context is a tabular result returned from a BigQuery query.  Do not repeat the question or the context when responding.

question:
{question}
context:
{result.to_markdown(index = False)}
"""
    question_response = textgen_model.predict(question_prompt, max_output_tokens = 500)

    return question_response.text

def BQ_QA(question, max_fixes = 7, schema_columns = schema_columns):

    query = initial_query(question, schema_columns)
    # run query:
    query_job = bq.query(query = query)
    # if errors, then generate repair query:
    if query_job.errors:
        query, query_job, fix_tries, codechat = fix_query(query, max_fixes)

    # respond with outcome:
    if query_job.errors:
        print(f'No answer generated after {fix_tries} tries.')
        return codechat
    else:
        question_response = answer_question(question, query_job)
        print(question_response)
        try:
            return question_response
        except:
            return None




from langchain.callbacks.base import BaseCallbackHandler

from langchain.schema import ChatMessage
import streamlit as st

st.title("💬 Chatbot") 
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]



if prompt := st.chat_input():
    

    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg = BQ_QA(st.session_state.messages)
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg)
