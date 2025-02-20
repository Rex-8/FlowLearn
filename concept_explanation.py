import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))

optimize_query_prompt_template = """
You are an expert educational assistant. Your goal is to optimize the user's question for a knowledge base search and identify key topics, taking into account the conversation context.

Optimized Query: Refine the question to be more specific and focused. If the question is already clear, return the original question unchanged. Ensure that the optimized_query is a concise rephrasing of the original question. Do not add any information that the user did not ask for.

Topics: Identify all possible topics of the question. Return a list of keywords related to the question.

Output Format: Return a JSON object with "optimized_query" and "topics" keys.

Example:
Question: How does acceleration relate to velocity?
Output:
{{
"optimized_query": "What is the relationship between acceleration and velocity?",
"topics": ["acceleration", "velocity", "acceleration velocity relationship"]
}}

Conversation Context: {conversation_context}

Question: {user_prompt}

Output:
"""

persist_directory = './chroma.db'

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

collection = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_function
)

optimize_query_prompt = PromptTemplate(input_variables=["user_prompt", "subject", "conversation_context"], template=optimize_query_prompt_template)
optimize_query_chain = optimize_query_prompt | llm | StrOutputParser()

def concept_explanation(subject, user_prompt, conversation_context=""):
    try:
        llm_output = optimize_query_chain.invoke({"user_prompt": user_prompt, "subject": subject, "conversation_context": conversation_context})
        data = json.loads(llm_output)
        optimized_query = data['optimized_query']
        topics = data['topics']
        print(optimized_query, topics)
        
        context = []
        for topic in topics:
            search_results = collection.similarity_search(
                topic,
                k=3,
                filter={"subject": subject}
            )
            context.extend([doc.page_content for doc in search_results])
        
        if context:
            gemini_prompt = f"""
            You are an expert in {subject}. Based on the following information, explain: {optimized_query}. Cite the specific documents from the Context that support your explanation. Prioritize answering the user's question directly before providing extra details , but also do give detailed answers. Take into account the previous conversation

            Context:
            {chr(10).join(context)}

            Explanation:
            """
            explanation = llm.invoke(gemini_prompt).content
        else:
            explanation = "I'm sorry, I couldn't find relevant information in the knowledge base. Please try rephrasing your question or providing more context."
    
    except Exception as e:
        explanation = f"An error occurred during the concept explanation workflow: {e}"
        print(f"Error during concept_explanation: {e}")
    
    return explanation

if __name__ == "__main__":
    explanation = concept_explanation(subject="Phy", user_prompt="What exactly is dimensional analysis and how does it link with Units")
    print(explanation)
