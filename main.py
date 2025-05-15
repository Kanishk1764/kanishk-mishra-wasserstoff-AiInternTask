import streamlit as st
import os
import PyPDF2
from PIL import Image
import pytesseract
from groq import Groq
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import hashlib
from datetime import datetime

# Initialize Groq client
client = Groq(api_key="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")

# Document processing and analysis class
class DocumentAnalyzer:
    def __init__(self):
        self.documents = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.cluster_model = None
        self.themes = {}
        
    def process_uploaded_file(self, uploaded_file):
        """Process uploaded files (PDF or image) and extract text"""
        file_id = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
        try:
            if file_ext == '.pdf':
                text = self._extract_text_from_pdf(uploaded_file)
            elif file_ext in ['.jpg', '.jpeg', '.png']:
                text = self._extract_text_from_image(uploaded_file)
            else:
                return None, "Unsupported file format"
                
            if not text.strip():
                return None, "No text could be extracted from the file"
                
            # Store document metadata and text
            self.documents[file_id] = {
                'id': file_id,
                'name': uploaded_file.name,
                'text': text,
                'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'pages': text.count('\f') + 1 if file_ext == '.pdf' else 1
            }
            
            return file_id, None
            
        except Exception as e:
            return None, f"Error processing file: {str(e)}"
    
    def _extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF using PyPDF2"""
        text = ""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\f"  # Form feed character for page break
        return text.strip()
    
    def _extract_text_from_image(self, image_file):
        """Extract text from image using Tesseract OCR"""
        image = Image.open(image_file)
        return pytesseract.image_to_string(image)
    
    def analyze_documents(self):
        """Analyze all documents to identify themes"""
        if not self.documents:
            return False, "No documents available for analysis"
            
        try:
            # Prepare document texts for analysis
            doc_texts = [doc['text'] for doc in self.documents.values()]
            doc_ids = list(self.documents.keys())
            
            # Vectorize the text data
            tfidf_matrix = self.vectorizer.fit_transform(doc_texts)
            
            # Cluster documents to identify themes (using 3-5 clusters)
            num_clusters = min(5, max(3, len(doc_texts)//15))
            self.cluster_model = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = self.cluster_model.fit_predict(tfidf_matrix)
            
            # Get top terms for each cluster to name the themes
            order_centroids = self.cluster_model.cluster_centers_.argsort()[:, ::-1]
            terms = self.vectorizer.get_feature_names_out()
            
            self.themes = {}
            for i in range(num_clusters):
                # Get top 5 terms for this cluster
                top_terms = [terms[ind] for ind in order_centroids[i, :5]]
                theme_name = "Theme {}: {}".format(i+1, ", ".join(top_terms))
                
                # Get documents belonging to this cluster
                theme_docs = [doc_ids[j] for j in range(len(doc_ids)) if clusters[j] == i]
                
                self.themes[theme_name] = theme_docs
                
            return True, "Analysis completed successfully"
            
        except Exception as e:
            return False, f"Analysis failed: {str(e)}"
    
    def query_documents(self, query):
        """Query the documents using LLM for detailed responses"""
        if not self.documents:
            return None, "No documents available for querying"
            
        try:
            # First get document-level responses
            doc_responses = []
            for doc_id, doc in self.documents.items():
                response = self._query_single_document(doc, query)
                if response:
                    doc_responses.append({
                        'Document ID': doc_id,
                        'Document Name': doc['name'],
                        'Extracted Answer': response['answer'],
                        'Citation': f"Page {response['page']}" if response['page'] else "Document"
                    })
            
            # Then get theme-level synthesis
            theme_response = ""
            if self.themes:
                theme_prompt = f"""
                Analyze the following themes identified across documents and provide a synthesized answer to the query: '{query}'
                
                Themes:
                {self.themes}
                
                Provide a comprehensive response that presents all relevant themes with supporting evidence.
                """
                
                theme_response = self._get_llm_response(theme_prompt)
            
            return doc_responses, theme_response, None
            
        except Exception as e:
            return None, None, f"Query failed: {str(e)}"
    
    def _query_single_document(self, doc, query):
        """Query a single document using LLM"""
        prompt = f"""
        Document Content:
        {doc['text'][:10000]}  # Limiting to first 10k chars for demo
        
        Question: {query}
        
        Extract the most relevant information from this document that answers the question.
        Provide:
        1. A concise answer
        2. The page number where this information was found (if available)
        
        Return your response in JSON format with 'answer' and 'page' keys.
        """
        
        response = self._get_llm_response(prompt)
        try:
            # Simple parsing of JSON-like response
            answer_start = response.find('"answer":') + 9
            answer_end = response.find('",', answer_start)
            answer = response[answer_start:answer_end].strip('"')
            
            page_start = response.find('"page":') + 7
            page_end = response.find('}', page_start)
            page = response[page_start:page_end].strip('" ')
            
            return {'answer': answer, 'page': page}
        except:
            return {'answer': response, 'page': ''}
    
    def _get_llm_response(self, prompt):
        """Get response from Groq LLM"""
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

# Streamlit UI
def main():
    st.set_page_config(page_title="Document Analysis Chatbot", page_icon="📄", layout="wide")
    
    st.title("Document Analysis & Theme Identification Chatbot")
    st.markdown("Upload multiple documents, analyze for themes, and query the collection")
    
    # Initialize document analyzer in session state
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = DocumentAnalyzer()
    
    # Sidebar for document upload and management
    with st.sidebar:
        st.header("Document Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload documents (PDF or images)",
            type=["pdf", "jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_id, error = st.session_state.analyzer.process_uploaded_file(uploaded_file)
                if error:
                    st.error(f"Error processing {uploaded_file.name}: {error}")
                else:
                    st.success(f"Processed {uploaded_file.name} (ID: {file_id[:8]}...)")
        
        # Show uploaded documents
        if st.session_state.analyzer.documents:
            st.subheader("Uploaded Documents")
            doc_df = pd.DataFrame([
                {
                    'ID': doc['id'][:8] + '...',
                    'Name': doc['name'],
                    'Pages': doc['pages'],
                    'Uploaded': doc['upload_time']
                }
                for doc in st.session_state.analyzer.documents.values()
            ])
            st.dataframe(doc_df, hide_index=True)
            
            # Button to analyze documents
            if st.button("Analyze Documents for Themes"):
                with st.spinner("Analyzing documents..."):
                    success, message = st.session_state.analyzer.analyze_documents()
                    if success:
                        st.success(message)
                        # Display themes
                        st.subheader("Identified Themes")
                        for theme, doc_ids in st.session_state.analyzer.themes.items():
                            st.markdown(f"**{theme}**")
                            st.write(f"Documents: {len(doc_ids)}")
                    else:
                        st.error(message)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Query Documents")
        
        query = st.text_area("Enter your question about the documents:", height=100)
        
        if st.button("Submit Query"):
            if not st.session_state.analyzer.documents:
                st.warning("Please upload documents first")
            elif not query.strip():
                st.warning("Please enter a question")
            else:
                with st.spinner("Processing your query..."):
                    doc_responses, theme_response, error = st.session_state.analyzer.query_documents(query)
                    
                    if error:
                        st.error(error)
                    else:
                        if doc_responses:
                            st.subheader("Document-Level Responses")
                            st.dataframe(pd.DataFrame(doc_responses), hide_index=True)
                        
                        if theme_response:
                            st.subheader("Theme Synthesis")
                            st.markdown(theme_response)
    
    with col2:
        st.header("Instructions")
        st.markdown("""
        1. **Upload Documents**: Use the sidebar to upload multiple PDF or image documents
        2. **Analyze**: Click "Analyze Documents" to identify common themes
        3. **Query**: Ask questions about the document collection in natural language
        
        The system will provide:
        - Individual document responses with citations
        - A synthesized analysis across all documents
        """)
        
        if st.session_state.analyzer.themes:
            st.subheader("Current Themes")
            for theme, doc_ids in st.session_state.analyzer.themes.items():
                st.markdown(f"🔹 **{theme}**")
                st.caption(f"{len(doc_ids)} documents")

if __name__ == "__main__":
    main()
