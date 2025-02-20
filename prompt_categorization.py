import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))

classify_prompt_template = """
You are an expert educational assistant. Analyze the user prompt by thinking step by step, considering the conversation context, and determine:

1. **Category:** Classify the prompt into "E" (Explain), "P" (Problem Solve), "R" (Revise), "S"(Self Assess) or "None":

    *   **Explain (E):** User wants a detailed explanation or teaching of a concept.
        *   Example: "Explain the concept of quantum entanglement."
    *   **Problem Solve (P):** User presents a specific problem or question (not a general topic).
        *   Example: "What is the probability of getting two heads when flipping a coin twice?"
    *   **Revise (R):** User needs a review or summary of a previously discussed topic.
        *   Example: "Summarize our previous discussion on the American Civil War."
    *   **Self Assess (S):** User asks for help checking their understanding.
        *   Example: "Quiz me on the different types of chemical bonds."
    *   **None (None):** The prompt is unrelated to education or the specified subjects.
        *   Example: "What is the weather like today?"

2.  **Subject:** Identify the primary subject: "Phy" (Physics), "Chem" (Chemistry), "Math" (Mathematics), or "None".

**Output Format:** Return category and subject separated by a comma (e.g., "E,Phy"). If you do not know the answer say NONE

**Conversation Context:** {conversation_context}

**Prompt:** {prompt}

**Output:**
"""

classification_prompt = PromptTemplate(input_variables=["prompt", "conversation_context"], template=classify_prompt_template)
classification_chain = classification_prompt | llm | StrOutputParser()

def classify_prompt(user_prompt, conversation_context=""):
    llm_output = classification_chain.invoke({"prompt": user_prompt, "conversation_context": conversation_context})
    try:
        category, subject = llm_output.strip().upper().split(",")
        category = category.strip()
        subject = subject.strip()
        if category not in ("E", "P", "R", "S", "NONE"):
            print(f"Warning: LLM returned unexpected category: '{category}'. Defaulting to 'E'.")
            category = "E"

        if subject not in ("Phy", "Chem", "Math", "None"):
            print(f"Warning: LLM returned unexpected subject: '{subject}'. Defaulting to 'None'.")
            subject = "None"

        return category, subject
    except ValueError:
        print(f"Warning: LLM returned unexpected format: '{llm_output}'. Defaulting to 'E,None'.")
        return "E", "None"