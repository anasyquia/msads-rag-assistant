import os
import time
import numpy as np
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever, Document
from pydantic import Field
import cohere
from typing import List, Any

# Configuration Parameters (matching original notebook)
CHUNK_SIZE = 600  # Size of text chunks
CHUNK_OVERLAP = 100  # Overlap between chunks
RETRIEVAL_K = 5  # Number of chunks to retrieve
TEMPERATURE = 0.0  # Temperature for LLM
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"

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

# Response Validator Class
class ResponseValidator:
    """Enhanced response validation with multiple checks"""
    
    def __init__(self):
        self.required_prefixes = [
            "Based on the program materials",
            "According to the program materials",
            "The program materials indicate",
            "The program materials state"
        ]
        
        self.hallucination_phrases = [
            "typically", "usually", "generally", "often",
            "in most cases", "commonly", "traditionally",
            "tends to", "approximately", "around"
        ]
        
        self.uncertainty_phrases = [
            "might", "may", "could", "possibly", "perhaps",
            "probably", "likely", "seems", "appears"
        ]
    
    def validate_response(self, response: str, context: str = None) -> dict:
        """Run all validation checks on a response"""
        try:
            # Source attribution
            has_attribution = any(response.lower().startswith(prefix.lower()) 
                                for prefix in self.required_prefixes)
            
            # Hallucination check
            found_hallucination_phrases = [phrase for phrase in self.hallucination_phrases 
                                         if phrase in response.lower()]
            hallucination_risk = len(found_hallucination_phrases) / len(self.hallucination_phrases)
            
            # Uncertainty check
            found_uncertainty_phrases = [phrase for phrase in self.uncertainty_phrases 
                                       if phrase in response.lower()]
            uncertainty_score = len(found_uncertainty_phrases) / len(self.uncertainty_phrases)
            
            # Length check
            words = response.split()
            word_count = len(words)
            is_appropriate_length = 10 <= word_count <= 150
            
            # Context usage check
            context_score = 1.0
            if context:
                response_words = response.lower().split()
                response_phrases = [' '.join(response_words[i:i+3]) 
                                  for i in range(len(response_words)-2)]
                found_phrases = [phrase for phrase in response_phrases 
                               if phrase in context.lower()]
                context_score = len(found_phrases) / len(response_phrases) if response_phrases else 0
            
            # Overall validation
            is_valid = all([
                has_attribution,
                hallucination_risk < 0.2,
                uncertainty_score < 0.2,
                is_appropriate_length,
                context_score > 0.2
            ])
            
            return {
                "is_valid": is_valid,
                "attribution": {"has_attribution": has_attribution},
                "hallucination_risk": {
                    "score": hallucination_risk,
                    "phrases": found_hallucination_phrases,
                    "is_safe": hallucination_risk < 0.2
                },
                "uncertainty": {
                    "score": uncertainty_score,
                    "phrases": found_uncertainty_phrases,
                    "is_confident": uncertainty_score < 0.2
                },
                "length": {
                    "word_count": word_count,
                    "is_appropriate": is_appropriate_length
                },
                "context_usage": {
                    "score": context_score,
                    "is_grounded": context_score > 0.2
                }
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "error": str(e)
            }

# Document Reranker Class
class DocumentReranker:
    """Enhanced reranker with retry logic and fallback"""
    
    def __init__(self, api_key: str):
        self.reranker = cohere.Client(api_key=api_key)
        self.max_retries = 2
        self.retry_delay = 10
    
    def rerank_documents(self, query: str, documents: list, top_k: int = 5) -> list:
        """Rerank with retry logic and better error handling"""
        if not documents:
            return documents
        
        for attempt in range(self.max_retries + 1):
            try:
                docs_for_rerank = [{"text": doc.page_content} for doc in documents]
                
                results = self.reranker.rerank(
                    model="rerank-english-v3.0",
                    query=query,
                    documents=docs_for_rerank,
                    top_n=min(top_k, len(documents)),
                    return_documents=True
                )
                
                reranked_docs = []
                for result in results.results:
                    original_doc = documents[result.index]
                    original_doc.metadata["rerank_score"] = round(result.relevance_score, 3)
                    reranked_docs.append(original_doc)
                
                return reranked_docs
                
            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    self.retry_delay *= 2  # Exponential backoff
                else:
                    return documents[:top_k]

# Enhanced Retriever Class
class EnhancedRetriever(BaseRetriever):
    """Retriever with dynamic sizing based on query complexity"""
    
    vectorstore: Any = Field()
    reranker: DocumentReranker = Field()
    base_initial_k: int = Field(default=15)  # Matching notebook
    base_final_k: int = Field(default=8)     # Matching notebook
    max_context_tokens: int = Field(default=11000)  # Matching notebook
    
    def _estimate_tokens(self, text: str) -> int:
        """Quick token estimation"""
        return len(text.split()) * 1.3
    
    def _adjust_retrieval_size(self, query: str) -> tuple[int, int]:
        """Dynamically adjust retrieval based on query"""
        query_lower = query.lower()
        
        # Increase retrieval for complex queries
        if any(word in query_lower for word in 
               ['compare', 'difference', 'vs', 'versus', 'both', 'all']):
            return self.base_initial_k + 5, self.base_final_k + 2
        
        # Standard retrieval for simple queries
        return self.base_initial_k, self.base_final_k
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Enhanced retrieval with token-aware sizing"""
        
        # Adjust retrieval size based on query
        initial_k, final_k = self._adjust_retrieval_size(query)
        
        # Step 1: Initial retrieval
        initial_docs = self.vectorstore.similarity_search(query, k=initial_k)
        
        # Step 2: Rerank
        reranked_docs = self.reranker.rerank_documents(query, initial_docs, top_k=final_k)
        
        # Step 3: Token-aware final selection
        selected_docs = []
        total_tokens = 0
        
        for doc in reranked_docs:
            doc_tokens = self._estimate_tokens(doc.page_content)
            if total_tokens + doc_tokens > self.max_context_tokens:
                break
            selected_docs.append(doc)
            total_tokens += doc_tokens
        
        return selected_docs

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
    5. Validating responses for accuracy
    """)

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'validator' not in st.session_state:
    st.session_state.validator = None
if 'reranker' not in st.session_state:
    st.session_state.reranker = None

# Function to create QA chain
def create_qa_chain(openai_api_key, cohere_api_key):
    try:
        # Set API keys
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["COHERE_API_KEY"] = cohere_api_key
        
        # Initialize components
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = FAISS.load_local(
            "./faiss_index",
            embeddings,
            allow_dangerous_deserialization=True  # Allow since this is our own trusted index
        )
        validator = ResponseValidator()
        reranker = DocumentReranker()
        
        # Create enhanced retriever with exact notebook parameters
        retriever = EnhancedRetriever(
            vectorstore=vectorstore,
            reranker=reranker,
            base_initial_k=15,  # Matching notebook
            base_final_k=8,     # Matching notebook
            max_context_tokens=11000  # Matching notebook
        )
        
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

        # Create QA chain with exact notebook parameters
        llm = ChatOpenAI(
            model=CHAT_MODEL,      # Using config variable
            temperature=TEMPERATURE # Using config variable
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

        return qa_chain, validator, reranker

    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None, None, None

# Main interface
if openai_api_key and cohere_api_key:
    if not st.session_state.qa_chain:
        with st.spinner("Initializing the Q&A system..."):
            qa_chain, validator, reranker = create_qa_chain(openai_api_key, cohere_api_key)
            if qa_chain and validator and reranker:
                st.session_state.qa_chain = qa_chain
                st.session_state.validator = validator
                st.session_state.reranker = reranker
                st.success("System initialized successfully!")

    # Question input
    question = st.text_input("Ask a question about the MS-ADS program:", placeholder="e.g., What is the tuition cost?")

    if question and st.session_state.qa_chain:
        with st.spinner("Finding answer..."):
            try:
                # Get initial results
                result = st.session_state.qa_chain({"query": question})
                answer = result['result']
                source_docs = result['source_documents']

                # Rerank sources
                reranked_docs = st.session_state.reranker.rerank_documents(
                    query=question,
                    documents=source_docs,
                    top_k=5
                )

                # Validate response
                validation_result = st.session_state.validator.validate_response(
                    response=answer,
                    context=" ".join(doc.page_content for doc in reranked_docs)
                )

                # Display answer
                st.markdown("### Answer:")
                st.markdown(answer)

                # Display validation results
                st.markdown("### Response Validation:")
                validation_status = "✅ VALID" if validation_result["is_valid"] else "❌ INVALID"
                st.markdown(f"**Status:** {validation_status}")
                
                with st.expander("View Validation Details"):
                    st.markdown("- **Source Attribution:** " + ("✅" if validation_result["attribution"]["has_attribution"] else "❌"))
                    st.markdown("- **Hallucination Risk:** " + ("✅ Low" if validation_result["hallucination_risk"]["is_safe"] else "❌ High"))
                    if validation_result["hallucination_risk"]["phrases"]:
                        st.markdown("  - Risky phrases: " + ", ".join(validation_result["hallucination_risk"]["phrases"]))
                    st.markdown("- **Uncertainty Level:** " + ("✅ Low" if validation_result["uncertainty"]["is_confident"] else "❌ High"))
                    st.markdown(f"- **Length:** {validation_result['length']['word_count']} words")
                    st.markdown("- **Context Usage:** " + ("✅ Good" if validation_result["context_usage"]["is_grounded"] else "❌ Poor"))

                # Display sources
                st.markdown("### Sources:")
                for i, doc in enumerate(reranked_docs, 1):
                    with st.expander(f"Source {i} (Relevance: {doc.metadata.get('rerank_score', 'N/A')})"):
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