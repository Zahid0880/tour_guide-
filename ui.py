from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv
from mongoservice import MongoService  # Import your MongoService class

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize MongoDB service
mongo_service = MongoService()

# Pydantic model for query input
class QueryInput(BaseModel):
    question: str
    user_id: str  # Added user_id to associate questions with a specific user
    stream: bool = False

# Utility functions
def load_chunks(docs):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory="./chromadb",
        embedding_function=embeddings,
        collection_name="tourism",
    )
    print("Adding documents to vectorstore...")
    vectorstore.add_documents(docs)
    print("Documents added to vectorstore.")

def get_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = Chroma(
        persist_directory="./chromadb",
        embedding_function=embeddings,
        collection_name="tourism",
    )
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def format_docs(docs):
    if not docs:
        raise ValueError("No documents to format.")
    if not all(isinstance(doc, Document) and hasattr(doc, "page_content") for doc in docs):
        raise ValueError("Documents must be instances of 'Document' and have a 'page_content' attribute.")
    return "\n\n".join(doc.page_content for doc in docs)

def run_chain(question):
    LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = get_retriever()

    docs = retriever.get_relevant_documents(question)
    formatted_data = format_docs(docs)

    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert AI assistant specializing in tourism recommendations. Answer the question using the provided data 
        and also you have previous question knowledge use it too.

        Context:
        {data}

        Question:
        {question}

        Your Answer:
        """
    )

    inputs = prompt.format_prompt(data=formatted_data, question=question).to_messages()
    result = LLM.invoke(inputs)
    return result.content

# Main endpoint
@app.post("/ask")
async def handle_question(input: QueryInput):
    """
    Process user queries and provide answers, storing relevant questions in MongoDB.
    """
    try:
        # Check if the question is already in MongoDB
        saved_response = mongo_service.fetch_data(input.question)
        if saved_response:
            return JSONResponse(content={"response": saved_response['response']})

        # Run chatbot for the current question
        bot_response = run_chain(input.question)

        # Save the question-answer pair in MongoDB
        mongo_service.save_chat({
            "user_id": input.user_id,
            "message": input.question,
            "response": bot_response
        })

        # Return response
        if input.stream:
            response_generator = (line for line in bot_response.splitlines())
            return StreamingResponse(response_generator, media_type="text/plain")
        else:
            return JSONResponse(content={"response": bot_response})

    except Exception as e:
        error_message = f"Error processing question: {e}"
        print(error_message)  # Debugging purposes
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/questions")
async def get_all_questions():
    """
    Retrieve all questions stored in the MongoDB database.
    """
    try:
        # Fetch all chat records from the database
        chats = mongo_service.fetch_all_chats()

        # Extract only the questions
        questions = [chat["message"] for chat in chats]

        # Return the questions
        return JSONResponse(content={"questions": questions})

    except Exception as e:
        error_message = f"Error retrieving questions: {e}"
        print(error_message)  # Debugging purposes
        raise HTTPException(status_code=500, detail=error_message)
