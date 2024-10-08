import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
import streamlit as st

llm = ChatOpenAI(model = "gpt-3.5-turbo")

examples = [
    {
        "query" : "Good Morning",
        "output" : """Ohayou gozaimasu. (おはようございます) 
                      \nOhayou (おはよう) is a casual way to say good morning among friends and family.  
                      \nAdding gozaimasu (ございます) makes it more polite and formal, suitable for use in professional or respectful settings.""" 
    },{
        "query" : "My name is Jacob",
        "output" : """Watashi no namae wa Jeikobu desu.(わたしのなまえはじぇいこぶです)  
                      \nWatashi no namae (わたしのなまえ)  means my name. 
                      \nwa (は) is the topic particle.  
                      \nJeikobu (じぇいこぶ) is Jacob in hiragana.  
                      \ndesu (です) is a polite sentence-ending particle."""
    },{
        "query" : "How are you",
        "output" : """Ogenki desu ka? (おげんきですか)  
                      \nOgenki (おげんき) means well-being or health. 
                      \ndesu (です)  is a polite form of is.  
                      \nka (か) is a question particle"""
    }
]

example_template = """
User : {query}
Output : {output}
"""

prompt_template = PromptTemplate(
    template = example_template,
    input_varaibles = ["query", "output"]
)

prefix = """
The following are the translation of english words to japanese , along with thier meanings here are some of the examples 
"""

suffix = """
User : {query}
Output :
"""

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt= prompt_template,
    prefix = prefix,
    suffix= suffix, 
    input_variables= ["query"],
    example_separator= "\n\n"
)

#chain = LLMChain(llm=llm, prompt= few_shot_prompt_template)

#stramlit framework
st.title("learn Nihongo (japanese)")
input_text = st.text_input("Lets go! Type your word")

llm = ChatOpenAI(model = "gpt-3.5-turbo")


if input_text:
    chain = few_shot_prompt_template | llm 
    result = chain.invoke({"query": input_text})

    st.write(result.content)