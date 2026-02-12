"""
LLM Service with RAG for Heart Disease Prediction App
This module handles the AI chat assistant with retrieval-augmented generation.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Union

from dotenv import load_dotenv

# LangChain imports
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document

# Vector Store (optional - will be imported when needed)
try:
    from langchain_community.vectorstores import Chroma, FAISS
    VECTORSTORE_AVAILABLE = True
except ImportError:
    Chroma = None
    FAISS = None
    VECTORSTORE_AVAILABLE = False

# Document Loaders (optional)
try:
    from langchain_community.document_loaders import (
        TextLoader,
        DirectoryLoader,
        PyPDFLoader,
        Docx2txtLoader,
        UnstructuredHTMLLoader,
        CSVLoader
    )
    LOADERS_AVAILABLE = True
except ImportError:
    TextLoader = None
    DirectoryLoader = None
    PyPDFLoader = None
    Docx2txtLoader = None
    UnstructuredHTMLLoader = None
    CSVLoader = None
    LOADERS_AVAILABLE = False

# Text Processing
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitters import RecursiveCharacterTextSplitter

# ============================================================================
# SETUP LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# Model Configuration
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_RETRIES = 3

# RAG Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.7

# Paths
KNOWLEDGE_BASE_PATH = "backend/knowledge_base"
VECTOR_STORE_PATH = "backend/vector_store"

# System Prompt with RAG context
SYSTEM_PROMPT = """You are a helpful medical assistant specializing in heart disease and cardiovascular health.

Your role is to:
1. Provide educational information about heart disease and cardiovascular health
2. Explain medical terms in simple, understandable language
3. Offer general health and lifestyle advice
4. Help interpret prediction results while emphasizing the importance of professional medical consultation
5. Answer questions about symptoms, risk factors, and preventive measures

Important guidelines:
- Always remind users that AI predictions and advice are NOT substitutes for professional medical consultation
- Be empathetic, clear, and supportive in your responses
- If asked about specific medical decisions, always recommend consulting a healthcare provider
- Provide evidence-based information when possible
- Be concise but thorough in your explanations
- If you're unsure about something, admit it and suggest consulting a doctor
- Use the provided context from the knowledge base to give accurate information
- If the context doesn't contain relevant information, say so and provide general guidance

Remember: You're here to educate and support, not to diagnose or prescribe treatment."""

RAG_PROMPT_TEMPLATE = """Use the following pieces of context from our medical knowledge base to answer the user's question. 
If the context doesn't contain relevant information, acknowledge this and provide general guidance while recommending professional consultation.

CONTEXT FROM KNOWLEDGE BASE:
{context}

USER QUESTION: {question}

Please provide a helpful, accurate, and empathetic response based on the above context."""

# ============================================================================
# DOCUMENT PROCESSOR CLASS
# ============================================================================

class DocumentProcessor:
    """Handles document loading and processing for RAG"""
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Supported file types and their loaders
        self.loader_mapping = {}
        if LOADERS_AVAILABLE:
            self.loader_mapping = {
                ".txt": TextLoader,
                ".pdf": PyPDFLoader,
                ".docx": Docx2txtLoader,
                ".html": UnstructuredHTMLLoader,
                ".csv": CSVLoader,
            }
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a single document"""
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        extension = path.suffix.lower()
        
        if extension not in self.loader_mapping:
            logger.warning(f"Unsupported file type: {extension}")
            return []
        
        try:
            loader_class = self.loader_mapping[extension]
            loader = loader_class(str(path))
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata["source"] = str(path)
                doc.metadata["file_type"] = extension
            
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """Load all documents from a directory"""
        path = Path(directory_path)
        
        if not path.exists():
            logger.error(f"Directory not found: {directory_path}")
            return []
        
        all_documents = []
        
        for extension in self.loader_mapping.keys():
            pattern = f"**/*{extension}"
            matching_files = list(path.glob(pattern))
            
            for file_path in matching_files:
                docs = self.load_document(str(file_path))
                all_documents.extend(docs)
        
        logger.info(f"Loaded {len(all_documents)} total documents from {directory_path}")
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        if not documents:
            return []
        
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")
        return chunks
    
    def process_directory(self, directory_path: str) -> List[Document]:
        """Load and process all documents from a directory"""
        documents = self.load_directory(directory_path)
        chunks = self.split_documents(documents)
        return chunks
    
    def create_documents_from_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None
    ) -> List[Document]:
        """Create documents from raw texts"""
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        documents = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]
        
        return self.split_documents(documents)

# ============================================================================
# VECTOR STORE MANAGER CLASS
# ============================================================================

class VectorStoreManager:
    """Manages the vector store for RAG"""
    
    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        persist_directory: str = VECTOR_STORE_PATH,
        use_gpu: bool = False
    ):
        self.persist_directory = persist_directory
        self.vector_store = None
        
        # Initialize embeddings
        try:
            model_kwargs = {'device': 'cuda' if use_gpu else 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            logger.info(f"Embeddings initialized: {embedding_model}")
            
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            self.embeddings = None
    
    def create_vector_store(
        self,
        documents: List[Document],
        store_type: str = "chroma"
    ) -> bool:
        """Create vector store from documents"""
        if not VECTORSTORE_AVAILABLE:
            logger.warning("Vector store libraries not available")
            return False
        
        if not self.embeddings:
            logger.error("Embeddings not initialized")
            return False
        
        if not documents:
            logger.error("No documents provided")
            return False
        
        try:
            if store_type.lower() == "chroma" and Chroma is not None:
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                    collection_name="heart_disease_knowledge"
                )
            elif store_type.lower() == "faiss" and FAISS is not None:
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
                # Save FAISS index
                self.vector_store.save_local(self.persist_directory)
            else:
                logger.error(f"Unknown store type or library not available: {store_type}")
                return False
            
            logger.info(f"Vector store created with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return False
    
    def load_vector_store(self, store_type: str = "chroma") -> bool:
        """Load existing vector store"""
        if not VECTORSTORE_AVAILABLE:
            logger.warning("Vector store libraries not available")
            return False
        
        if not self.embeddings:
            logger.error("Embeddings not initialized")
            return False
        
        try:
            if store_type.lower() == "chroma" and Chroma is not None:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name="heart_disease_knowledge"
                )
            elif store_type.lower() == "faiss" and FAISS is not None:
                self.vector_store = FAISS.load_local(
                    self.persist_directory,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                logger.error(f"Unknown store type or library not available: {store_type}")
                return False
            
            logger.info("Vector store loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to existing vector store"""
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return False
        
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def similarity_search(
        self,
        query: str,
        k: int = TOP_K_RESULTS,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """Search for similar documents"""
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return []
        
        try:
            if score_threshold:
                # Search with score filtering
                results = self.vector_store.similarity_search_with_score(
                    query, k=k
                )
                # Filter by threshold
                filtered_results = [
                    doc for doc, score in results
                    if score >= score_threshold
                ]
                return filtered_results
            else:
                return self.vector_store.similarity_search(query, k=k)
                
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def similarity_search_with_scores(
        self,
        query: str,
        k: int = TOP_K_RESULTS
    ) -> List[tuple]:
        """Search with relevance scores"""
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return []
        
        try:
            return self.vector_store.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error(f"Error in similarity search with scores: {e}")
            return []
    
    def delete_collection(self) -> bool:
        """Delete the vector store collection"""
        try:
            if hasattr(self.vector_store, 'delete_collection'):
                self.vector_store.delete_collection()
            
            # Remove persisted files
            import shutil
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
            
            self.vector_store = None
            logger.info("Vector store deleted")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting vector store: {e}")
            return False

# ============================================================================
# RAG-ENABLED CHAT ASSISTANT CLASS
# ============================================================================

class HeartDiseaseAssistant:
    """RAG-enabled chat assistant for heart disease queries"""
    
    def __init__(
        self,
        api_token: Optional[str] = None,
        model_id: str = DEFAULT_MODEL,
        knowledge_base_path: str = KNOWLEDGE_BASE_PATH,
        vector_store_path: str = VECTOR_STORE_PATH,
        use_rag: bool = True
    ):
        """
        Initialize the RAG-enabled chat assistant
        
        Args:
            api_token: Hugging Face API token
            model_id: Model repository ID
            knowledge_base_path: Path to knowledge base documents
            vector_store_path: Path to persist vector store
            use_rag: Whether to enable RAG functionality
        """
        self.api_token = api_token or self._load_api_token()
        self.model_id = model_id
        self.knowledge_base_path = knowledge_base_path
        self.vector_store_path = vector_store_path
        self.use_rag = use_rag
        
        self.chat_model = None
        self.conversation_history: List[Dict[str, str]] = []
        
        # RAG components
        self.document_processor = None
        self.vector_store_manager = None
        self.rag_initialized = False
        
        # Initialize LLM
        if self.api_token:
            self._initialize_model()
        else:
            logger.error("No API token provided")
        
        # Initialize RAG if enabled
        if self.use_rag:
            self._initialize_rag()
    
    def _load_api_token(self) -> Optional[str]:
        """Load API token from environment"""
        load_dotenv()
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        
        if not token:
            logger.warning("HUGGINGFACEHUB_API_TOKEN not found in environment")
        
        return token
    
    def _initialize_model(self) -> bool:
        """Initialize the Hugging Face chat model"""
        try:
            logger.info(f"Initializing model: {self.model_id}")
            
            llm = HuggingFaceEndpoint(
                repo_id=self.model_id,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                huggingfacehub_api_token=self.api_token,
                streaming=False,
            )
            
            self.chat_model = ChatHuggingFace(llm=llm)
            logger.info("Chat model initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing chat model: {e}")
            self.chat_model = None
            return False
    
    def _initialize_rag(self) -> bool:
        """Initialize RAG components"""
        if not VECTORSTORE_AVAILABLE or not LOADERS_AVAILABLE:
            logger.warning("RAG libraries not available. RAG features will be disabled.")
            return False
        
        try:
            logger.info("Initializing RAG components...")
            
            # Initialize document processor
            self.document_processor = DocumentProcessor(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            
            # Initialize vector store manager
            self.vector_store_manager = VectorStoreManager(
                embedding_model=EMBEDDING_MODEL,
                persist_directory=self.vector_store_path
            )
            
            # Try to load existing vector store
            if os.path.exists(self.vector_store_path):
                if self.vector_store_manager.load_vector_store():
                    self.rag_initialized = True
                    logger.info("Loaded existing vector store")
                    return True
            
            # Create new vector store from knowledge base
            if os.path.exists(self.knowledge_base_path):
                logger.info(f"Creating vector store from knowledge base: {self.knowledge_base_path}")
                documents = self.document_processor.process_directory(self.knowledge_base_path)
                
                if documents:
                    if self.vector_store_manager.create_vector_store(documents):
                        self.rag_initialized = True
                        logger.info(f"RAG initialized with {len(documents)} document chunks")
                        return True
            
            logger.warning("RAG initialization skipped - no knowledge base found")
            return False
            
        except Exception as e:
            logger.error(f"Error initializing RAG: {e}")
            return False
    
    def _retrieve_context(self, query: str) -> str:
        """Retrieve relevant context from knowledge base"""
        if not self.rag_initialized or not self.vector_store_manager:
            return ""
        
        try:
            # Search for relevant documents
            docs = self.vector_store_manager.similarity_search(
                query=query,
                k=TOP_K_RESULTS,
                score_threshold=SIMILARITY_THRESHOLD
            )
            
            if not docs:
                return ""
            
            # Combine document contents
            context_parts = []
            for doc in docs:
                content = doc.page_content.strip()
                if content:
                    context_parts.append(content)
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return ""
    
    def _build_rag_prompt(self, user_message: str, context: str) -> str:
        """Build RAG prompt with retrieved context"""
        if context:
            return RAG_PROMPT_TEMPLATE.format(context=context, question=user_message)
        return user_message
    
    def is_ready(self) -> bool:
        """Check if the chat model is ready"""
        return self.chat_model is not None
    
    def get_response(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        use_rag: Optional[bool] = None
    ) -> str:
        """
        Get response from AI model
        
        Args:
            user_message: User's input message
            context: Optional context from prediction results
            use_rag: Override for RAG usage (defaults to instance setting)
            
        Returns:
            AI response string
        """
        if not self.is_ready():
            return self._get_fallback_response()
        
        # Validate input
        if not user_message or not user_message.strip():
            return "Please provide a question or message."
        
        # Sanitize input
        user_message = self._sanitize_input(user_message)
        
        # Determine RAG usage
        use_rag_enabled = self.use_rag if use_rag is None else use_rag
        
        for attempt in range(MAX_RETRIES):
            try:
                # Retrieve RAG context if enabled
                rag_context = ""
                if use_rag_enabled:
                    rag_context = self._retrieve_context(user_message)
                
                # Build messages
                messages = self._build_messages(user_message, context, rag_context)
                
                # Get response
                response = self.chat_model.invoke(messages)
                
                # Store in conversation history
                self._update_history(user_message, response.content)
                
                return self._format_response(response.content)
                
            except Exception as e:
                logger.error(f"Error getting AI response (attempt {attempt + 1}): {e}")
                
                if attempt == MAX_RETRIES - 1:
                    return self._get_error_response(str(e))
        
        return self._get_error_response("Maximum retries exceeded")
    
    def _build_messages(
        self,
        user_message: str,
        prediction_context: Optional[Dict[str, Any]],
        rag_context: str
    ) -> List:
        """Build message list for the model"""
        messages = []
        
        # Add system prompt
        if rag_context:
            # Use RAG-enhanced system prompt
            full_prompt = f"{SYSTEM_PROMPT}\n\nYou have access to a medical knowledge base. Use the following context to provide accurate answers:\n\n{rag_context}"
            messages.append(SystemMessage(content=full_prompt))
        else:
            messages.append(SystemMessage(content=SYSTEM_PROMPT))
        
        # Add prediction context if available
        if prediction_context:
            context_message = self._build_prediction_context(prediction_context)
            messages.append(SystemMessage(content=context_message))
        
        # Add conversation history (keep last 5 exchanges)
        history_start = max(0, len(self.conversation_history) - 5)
        for entry in self.conversation_history[history_start:]:
            messages.append(HumanMessage(content=entry["user"]))
            messages.append(AIMessage(content=entry["assistant"]))
        
        # Add current user message
        messages.append(HumanMessage(content=user_message))
        
        return messages
    
    def _build_prediction_context(self, context: Dict[str, Any]) -> str:
        """Build comprehensive context message from ML prediction results"""
        if not context:
            return ""
        
        prediction = context.get('prediction', 0)
        prediction_label = context.get('prediction_label', 'Unknown')
        risk_level = context.get('risk_level', 'Unknown')
        probability_disease = context.get('probability_disease', 0)
        probability_no_disease = context.get('probability_no_disease', 0)
        recommendation = context.get('recommendation', 'None')
        patient_data = context.get('patient_data', {})
        
        # Interpret patient data values
        chest_pain_types = {
            0: "Typical Angina",
            1: "Atypical Angina", 
            2: "Non-anginal Pain",
            3: "Asymptomatic"
        }
        
        gender_str = 'Male' if patient_data.get('gender', 0) == 1 else 'Female'
        chest_pain_str = chest_pain_types.get(patient_data.get('chestpain', 0), 'Unknown')
        
        # Build detailed ML output summary
        ml_output = f"""MACHINE LEARNING MODEL OUTPUT:
================================================================================

PREDICTION RESULTS:
- Classification: {prediction_label}
- Risk Level: {risk_level}
- Disease Probability: {probability_disease:.2%}
- No Disease Probability: {probability_no_disease:.2%}
- Confidence Score: {max(probability_disease, probability_no_disease):.2%}

RISK ASSESSMENT:
"""
        
        # Add risk-based interpretation
        if probability_disease >= 0.8:
            ml_output += """- Risk Category: VERY HIGH RISK
- Interpretation: The ML model indicates a very high probability of heart disease.
- Urgency: Immediate medical consultation strongly recommended.
- Key Factors: Multiple elevated risk indicators present.
"""
        elif probability_disease >= 0.6:
            ml_output += """- Risk Category: HIGH RISK
- Interpretation: The ML model indicates elevated risk for heart disease.
- Urgency: Schedule medical consultation soon.
- Key Factors: Several input parameters suggest increased cardiovascular risk.
"""
        elif probability_disease >= 0.4:
            ml_output += """- Risk Category: MODERATE RISK
- Interpretation: The ML model indicates moderate risk for heart disease.
- Urgency: Consider preventive check-up.
- Key Factors: Some risk factors present that warrant attention.
"""
        elif probability_disease >= 0.2:
            ml_output += """- Risk Category: LOW RISK
- Interpretation: The ML model indicates lower risk for heart disease.
- Urgency: Maintain healthy habits and regular monitoring.
- Key Factors: Most parameters within healthy ranges.
"""
        else:
            ml_output += """- Risk Category: VERY LOW RISK
- Interpretation: The ML model indicates very low probability of heart disease.
- Urgency: Continue healthy lifestyle.
- Key Factors: Cardiovascular markers are favorable.
"""
        
        ml_output += f"""

PATIENT INPUT DATA:
- Age: {patient_data.get('age', 'N/A')} years
- Gender: {gender_str}
- Chest Pain Type: {chest_pain_str}
- Resting Blood Pressure: {patient_data.get('restingBP', 'N/A')} mm Hg
- Serum Cholesterol: {patient_data.get('serumcholestrol', 'N/A')} mg/dl
- Fasting Blood Sugar > 120 mg/dl: {'Yes' if patient_data.get('fastingbloodsugar', 0) == 1 else 'No'}
- Maximum Heart Rate: {patient_data.get('maxheartrate', 'N/A')}
- Exercise Induced Angina: {'Yes' if patient_data.get('exerciseangia', 0) == 1 else 'No'}
- ST Depression (Oldpeak): {patient_data.get('oldpeak', 'N/A')}

ML MODEL RECOMMENDATION:
{recommendation}

================================================================================

Use this ML model output to provide personalized, relevant health suggestions and advice based on the prediction results."""
        
        return ml_output
    
    def _sanitize_input(self, text: str) -> str:
        """Sanitize user input"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Limit length
        max_length = 1000
        if len(text) > max_length:
            text = text[:max_length] + "..."
            logger.warning(f"Input truncated to {max_length} characters")
        
        return text
    
    def _format_response(self, response: str) -> str:
        """Format the AI response"""
        # Remove any potential system artifacts
        response = response.strip()
        
        # Ensure medical disclaimer if not present
        disclaimer_keywords = ['doctor', 'physician', 'medical professional', 'healthcare provider']
        has_disclaimer = any(keyword in response.lower() for keyword in disclaimer_keywords)
        
        if not has_disclaimer and len(response) > 100:
            response += "\n\n⚕️ Remember: This information is educational. Always consult with a healthcare professional for medical advice."
        
        return response
    
    def _update_history(self, user_message: str, ai_response: str):
        """Update conversation history"""
        # Keep only last 10 messages to avoid context length issues
        self.conversation_history.append({
            "user": user_message,
            "assistant": ai_response
        })
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)
    
    def _get_fallback_response(self) -> str:
        """Return fallback response when model is unavailable"""
        return """I apologize, but the chat assistant is currently unavailable. 
        
Please ensure:
1. Your Hugging Face API token is configured correctly
2. You have an active internet connection
3. The Hugging Face service is accessible

For immediate assistance with heart disease questions, please consult a healthcare professional."""
    
    def _get_error_response(self, error_msg: str) -> str:
        """Return user-friendly error response"""
        logger.error(f"Returning error response: {error_msg}")
        return f"""I apologize, but I encountered an error processing your request.

If this persists, please:
1. Try rephrasing your question
2. Check your internet connection
3. Contact support if the issue continues

For urgent medical concerns, please contact a healthcare provider immediately."""
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
    
    def add_knowledge(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> bool:
        """Add new texts to the knowledge base"""
        if not self.vector_store_manager or not self.vector_store_manager.vector_store:
            logger.error("Vector store not initialized")
            return False
        
        try:
            # Create documents
            documents = self.document_processor.create_documents_from_texts(texts, metadatas)
            
            # Add to vector store
            if self.vector_store_manager.add_documents(documents):
                logger.info(f"Added {len(documents)} documents to knowledge base")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return False
    
    def rebuild_knowledge_base(self, knowledge_base_path: Optional[str] = None) -> bool:
        """Rebuild the entire knowledge base"""
        if knowledge_base_path:
            self.knowledge_base_path = knowledge_base_path
        
        # Delete existing vector store
        if self.vector_store_manager:
            self.vector_store_manager.delete_collection()
        
        # Reinitialize
        self.rag_initialized = False
        return self._initialize_rag()
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        if not self.rag_initialized or not self.vector_store_manager.vector_store:
            return {
                "initialized": False,
                "document_count": 0,
                "vector_store_exists": os.path.exists(self.vector_store_path)
            }
        
        try:
            # Get collection stats for Chroma
            vector_store = self.vector_store_manager.vector_store
            if hasattr(vector_store, '_collection'):
                count = len(vector_store._collection.get()['documents'])
            elif hasattr(vector_store, 'index'):
                count = vector_store.index.ntotal
            else:
                count = 0
            
            return {
                "initialized": True,
                "document_count": count,
                "vector_store_path": self.vector_store_path
            }
            
        except Exception as e:
            logger.error(f"Error getting knowledge stats: {e}")
            return {"error": str(e)}

# ============================================================================
# HELPER FUNCTIONS FOR BACKWARD COMPATIBILITY
# ============================================================================

_global_assistant: Optional[HeartDiseaseAssistant] = None

def initialize_chat_model():
    """
    Initialize the chat model (legacy function for backward compatibility)
    
    Returns:
        Chat model or None
    """
    global _global_assistant
    _global_assistant = HeartDiseaseAssistant()
    
    if _global_assistant.is_ready():
        return _global_assistant.chat_model
    return None

def get_ai_response(
    chat_model,
    user_message: str,
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Get AI response (legacy function for backward compatibility)
    
    Args:
        chat_model: Chat model instance (ignored if global assistant exists)
        user_message: User's message
        context: Optional prediction context
        
    Returns:
        AI response string
    """
    global _global_assistant
    
    if _global_assistant and _global_assistant.is_ready():
        return _global_assistant.get_response(user_message, context)
    
    # Fallback to direct model call if no global assistant
    if not chat_model:
        return "Chat model not available. Please check your API configuration."
    
    try:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ]
        
        response = chat_model.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Error in legacy get_ai_response: {e}")
        return f"Sorry, I couldn't process your request: {str(e)}"

# ============================================================================
# MAIN (FOR TESTING)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Heart Disease Chat Assistant - Test Suite")
    print("=" * 70)
    
    # Test 1: Initialize assistant
    print("\n[TEST 1] Initializing chat assistant...")
    assistant = HeartDiseaseAssistant()
    
    if assistant.is_ready():
        print("✓ Chat assistant initialized successfully!")
        
        # Test 2: Check RAG status
        print("\n[TEST 2] Checking RAG status...")
        stats = assistant.get_knowledge_stats()
        print(f"RAG Initialized: {stats.get('initialized', False)}")
        print(f"Document Count: {stats.get('document_count', 0)}")
        
        # Test 3: Simple query
        print("\n[TEST 3] Testing simple query...")
        response = assistant.get_response(
            "What are the risk factors for heart disease?"
        )
        print(f"\nAI Response:\n{response}")
        
        # Test 4: Query with context
        print("\n[TEST 4] Testing query with prediction context...")
        context = {
            'prediction_label': 'Heart Disease Detected',
            'risk_level': 'High',
            'probability_disease': 0.78,
            'recommendation': 'Consult a cardiologist immediately'
        }
        response = assistant.get_response(
            "What should I do about these results?",
            context=context
        )
        print(f"\nAI Response:\n{response}")
        
        print("\n" + "=" * 70)
        print("All tests completed!")
        print("=" * 70)
        
    else:
        print("✗ Failed to initialize chat assistant")
        print("\nTroubleshooting:")
        print("1. Check that HUGGINGFACEHUB_API_TOKEN is set in your .env file")
        print("2. Verify your Hugging Face API token is valid")
        print("3. Ensure you have internet connectivity")
