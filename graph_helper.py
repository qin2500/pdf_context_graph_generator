import pandas as pd
import numpy as np
import os
import json
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


def get_concepts(prompt:pd.DataFrame, chunk_id:uuid):
    print("POOOOOOOO")
    prompt = prompt[0]
    assert isinstance(prompt, str), "prompt must be a string"
    print(prompt)
    load_dotenv()
    chat = ChatOpenAI(temperature=0, openai_api_key = os.getenv("OPENAI_API_KEY"),  model="gpt-3.5-turbo")
    SYS_PROMPT = (
            "You are a network graph maker who extracts terms and their relations from a given context. "
            
        )
    
    USER_PROMPT = ("You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
            "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
            "It is very important that you perform your task only on the text provided within the context chunk and nothing else."
            "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
                "\tTerms may include object, entity, location, organization, person, \n"
                "\tcondition, acronym, documents, service, concept, etc.\n"
                "\tTerms should be as atomistic as possible\n\n"
            "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
                "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
                "\tTerms can be related to many other terms\n\n"
            "Thought 3: Find out the relation between each such related pair of terms. \n\n"
            "Format your output as a list of json. Each element of the list contains a pair of terms"
            "and the relation between them, like the follwing: \n"
            "[\n"
            "   {\n"
            '       "node_1": "A concept from extracted ontology",\n'
            '       "node_2": "A related concept from extracted ontology",\n'
            '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
            "   }, {...}\n"
            "]"
            f"context: ```{prompt}``` \n\n ")
    messages = [
        SystemMessage(
            content=SYS_PROMPT
        ),
        HumanMessage(
            content=USER_PROMPT
        )
    ]
   
    response = chat(messages).content
    try:
        result = json.loads(response)
        result = [dict(item, **chunk_id) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result
    

def formatDF(nodes_list: pd.DataFrame) -> pd.DataFrame:
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
    graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(lambda x: x.lower())
    graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(lambda x: x.lower())

    return graph_dataframe