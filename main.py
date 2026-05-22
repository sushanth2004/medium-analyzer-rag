import os 
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"],
    embedding= embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based on the following context :
    {context}
    
    Question : {question}

    Provide a detailed answer.
    """
)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

#----------------------------------------------------------------------
# Implementaion - 1 : Without LCEL (simple function based approach)
#----------------------------------------------------------------------

def retrieval_chain_without_Lcel (query : str):
    """
    Simple retrieval chain without Lcel.
    Manually retrieves documents, formats them and generates the response.

    Limitations :
    1. Manual step by execution.
    2. No built-in streaming support.
    3. No async support without additional code.
    4. Harder to compose with other chains.
    5. More verbose and error prone.
    6. Difficult to trace output.
    """
    # step-1 : Retrieve relavant documents
    documents = retriever.invoke(query)

    # step-2 : format documents into context string
    context = format_docs(documents)

    # step-3 : Format prompt with context and query
    messages = prompt_template.format(context=context, question=query)

    # step-4 : Invoke LLM with formated messages 
    response = llm.invoke(messages)
    
    # step-5 : return the content
    return response.content


#---------------------------------------------------------------------------------
# Implementaion - 2 : With LCEL (langchain Expression language) - good approach 
#--------------------------------------------------------------------------------

def create_retrieval_chain_with_lcel() :
    """
    creates a retrieval chain with lcel
    return a chain that can be invoked with {"question" : ""}
    
    Advantages over non-lcel approach :
    - Declarative and composible : easy to chain operations with pipe(|) operator
    - Built-in streaming : chain.stream() works out of the box 
    - Built-in async : chain.ainvoke() and chain.astream() are available 
    - batch processing : chain.batch() for multiple inputs 
    - Type safety : better integration with langchain type system
    - less code : More concise and readable 
    - reusable : chain can be saved , shared and comeposed with other chains 
    - better debugging : can be traceble with langsmith
    
    """

    retrieval_chain = (
        # runnablePassthrough 
        # i/p - {"question" : "what is pinecone ?"}
        # o/p -
        #{
        # "question" : "what is pinecone ?",
        # "context"  : "........."
        #} 

        RunnablePassthrough.assign(
            context = itemgetter("question") | retriever | format_docs 
        )
        | prompt_template
        | llm 
        | StrOutputParser()
    )

    return retrieval_chain




if __name__ == "__main__" :
    print("retrieving.....")

    #query
    query = "what is Pinecone in machine learning?"
    
    #-----Raw LLM implementation without RAG----

    print("\n"+"="*70)
    print("Implementation 0 : Raw Invocation without RAG")
    print("="*70)

    result_raw = llm.invoke([HumanMessage(content=query)])

    print("\nAnswer :")
    print(result_raw.content)


    print("\n"+"="*70)
    print("Implementation 1 : Implementation without LCEL")
    print("="*70)

    result_without_lcel = retrieval_chain_without_Lcel(query)

    print("\nAnswer :")
    print(result_without_lcel)

    print("\n"+"="*70)
    print("Implementation 2 : Implementation with LCEL")
    print("="*70)

    chain_with_lcel = create_retrieval_chain_with_lcel()

    result_with_lcel = chain_with_lcel.invoke({"question" : query})

    print("\nAnswer :")
    print(result_with_lcel)

