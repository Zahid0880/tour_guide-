from openai_service import call_open_ai
from chromadb_service import retriever
from csv_loader import CSVLoader  # Assuming CSVLoader is your new class for loading CSV data

PROMPT = """You are a Tourism Recommendation Bot. Provide detailed recommendations based on the questions asked about tourism destinations, activities, and travel tips. 
If an off-topic question is asked, you should answer 'I can't answer topic-unrelated questions.'"""

def recommender_bot(question):
    try:
        # Retrieve similar documents
        texts = []
        docs = retriever(question)

        for doc in docs:
            texts.append(doc)

        VECTOR_DB_DATA = "\n".join(texts)
        
        # Log the query and data for debugging
        print(f"Query: {question}")
        #print(f"VECTOR_DB_DATA: {VECTOR_DB_DATA}")

        # Prepare message for OpenAI API
        message = [
            {
                "role": "system",
                "content": PROMPT
            },
            {
                "role": "system",
                "content": f"Use this data to answer: {VECTOR_DB_DATA}"
            },
            {
                "role": "user",
                "content": question
            }
        ]
        
        # Call OpenAI API
        response = call_open_ai(message)
        print(response)  # Log the raw response

    except Exception as e:
        print("An error occurred in recommender_bot function:", str(e))