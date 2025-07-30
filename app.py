import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import cohere

# Page config
st.set_page_config(
    page_title="UChicago MS-ADS Q&A",
    page_icon="🎓",
    layout="wide"
)

# Title and description
st.title("🎓 UChicago MS-ADS Program Q&A")
st.markdown("""
This application helps answer questions about the University of Chicago's Master of Science in Applied Data Science program.
Simply enter your OpenAI and Cohere API keys, then ask any question about the program!
""")

# Sidebar for API keys
with st.sidebar:
    st.header("API Keys")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    cohere_api_key = st.text_input("Cohere API Key", type="password")
    
    st.markdown("---")
    st.markdown("""
    ### About
    This app uses RAG (Retrieval-Augmented Generation) to provide accurate answers about the MS-ADS program by:
    1. Loading pre-scraped program data
    2. Finding relevant context using FAISS
    3. Reranking results with Cohere
    4. Generating accurate answers with GPT
    """)

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Function to create QA chain
def create_qa_chain(openai_api_key, cohere_api_key):
    try:
        # Set API keys
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["COHERE_API_KEY"] = cohere_api_key
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Load the pre-created FAISS index
        vectorstore = FAISS.load_local("./faiss_index", embeddings)
        
        # Create prompt template
        prompt_template = """You are a precise information system for the University of Chicago's MS in Applied Data Science program.

CORE REQUIREMENTS:
1. ALWAYS start with "Based on the program materials..."
2. Include specific details from the context (dates, costs, contact info, URLs) as available
3. Be specific about program types (Online vs In-Person) when relevant
4. Use exact quotes and numbers from the context
5. If information seems incomplete, state what you found and note limitations

RESPONSE RULES:
- NO speculation beyond provided context
- NO approximations unless explicitly quoted
- NO hedging language (might, maybe, probably) unless in quotes
- If asked about visa sponsorship, be explicit about which programs are eligible
- If asked about appointments/advising, mention specific contact methods available

Context: {context}

Question: {question}

Complete and accurate answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create QA chain
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.0
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

        return qa_chain

    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None

# Main interface
if openai_api_key and cohere_api_key:
    if not st.session_state.qa_chain:
        with st.spinner("Initializing the Q&A system..."):
            st.session_state.qa_chain = create_qa_chain(openai_api_key, cohere_api_key)
            if st.session_state.qa_chain:
                st.success("System initialized successfully!")

    # Question input
    question = st.text_input("Ask a question about the MS-ADS program:", placeholder="e.g., What is the tuition cost?")

    if question and st.session_state.qa_chain:
        with st.spinner("Finding answer..."):
            try:
                # Get answer
                result = st.session_state.qa_chain({"query": question})
                answer = result['result']
                source_docs = result['source_documents']

                # Display answer
                st.markdown("### Answer:")
                st.markdown(answer)

                # Display sources
                st.markdown("### Sources:")
                for i, doc in enumerate(source_docs, 1):
                    with st.expander(f"Source {i}"):
                        st.markdown(f"**Title:** {doc.metadata.get('title', 'Unknown')}")
                        st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                        st.markdown("**Content Preview:**")
                        st.markdown(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)

            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

else:
    st.warning("Please enter your OpenAI and Cohere API keys in the sidebar to start.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit, LangChain, and FAISS") 