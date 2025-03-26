import os
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from collections import Counter
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict
from src.document_processor import DocumentProcessor


class GeminiAgent:
    def __init__(self):
        """Initialize Gemini API client and Qdrant Cloud database"""
        self.model = None
        self.qdrant_client = None
        self.collection_name = "document_chunks"

        # Load environment variables
        if not load_dotenv():
            st.warning("No .env file found. Using default settings.")
        st.sidebar.info(f".env loaded: {os.getenv('GEMINI_API_KEY') is not None}")

        # Initialize Gemini
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in .env file")
            genai.configure(api_key=api_key)
            self.model_name = 'gemini-1.5-flash'
            self.model = genai.GenerativeModel(self.model_name)
            st.sidebar.success(f"Model Initialized: {self.model_name}")
        except ValueError as ve:
            st.error(f"Gemini Configuration Error: {ve}")
            self.model = None
            return
        except Exception as e:
            st.error(f"Gemini Initialization Error: {e}")
            self.model = None
            return

        # Initialize Qdrant Cloud or local
        qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
        qdrant_port = int(os.getenv('QDRANT_PORT', 6333))
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        st.sidebar.info(f"Attempting connection to Qdrant at {qdrant_host}:{qdrant_port}")

        try:
            # Determine if it's a cloud setup (host contains domain)
            if '.' in qdrant_host and not qdrant_host.startswith(('http://', 'https://')):
                # Construct full URL for Qdrant Cloud
                url = f"https://{qdrant_host}:{qdrant_port}"
                self.qdrant_client = QdrantClient(
                    url=url,
                    api_key=qdrant_api_key,
                    timeout=10
                )
            else:
                # Local setup
                self.qdrant_client = QdrantClient(
                    host=qdrant_host,
                    port=qdrant_port,
                    api_key=qdrant_api_key,
                    timeout=10
                )

            self.qdrant_client.get_collections()  # Test connection
            self._setup_qdrant_collection()
            st.sidebar.success("Qdrant connection established")
        except Exception as qe:
            st.sidebar.warning(f"Qdrant connection failed: {qe}. Using local processing instead.")
            self.qdrant_client = None

    def _setup_qdrant_collection(self):
        """Set up Qdrant collection"""
        if not self.qdrant_client:
            return
        try:
            collections = self.qdrant_client.get_collections().collections
            if self.collection_name not in [c.name for c in collections]:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
                )
                st.sidebar.info(f"Created Qdrant collection: {self.collection_name}")
            # Optionally recreate to ensure fresh data
            # else:
            #     self.qdrant_client.recreate_collection(...)
        except Exception as e:
            st.sidebar.error(f"Qdrant Setup Error: {e}")
            self.qdrant_client = None

    def store_chunks(self, chunks: List[str], embeddings: List[List[float]], metadata: List[Dict]):
        """Store chunks in Qdrant Cloud"""
        if not self.qdrant_client:
            st.write("Qdrant unavailable. Skipping vector storage.")
            return
        points = [
            models.PointStruct(
                id=i,
                vector=embedding,
                payload={"text": chunk, **meta}
            )
            for i, (chunk, embedding, meta) in enumerate(zip(chunks, embeddings, metadata))
        ]
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        st.write(f"Stored {len(points)} chunks in Qdrant")

    def search_relevant_chunks(self, query: str, processor: DocumentProcessor, limit: int = 5) -> List[str]:
        """Search Qdrant for relevant chunks"""
        if not self.qdrant_client:
            st.write("Qdrant unavailable. Using full document context.")
            return []
        query_embedding = processor.embedder.encode([query])[0].tolist()
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        # st.write("Retrieved chunks from Qdrant")
        return [hit.payload["text"] for hit in search_result]

    def list_qdrant_points(self, limit: int = 10):
        """List points stored in Qdrant Cloud"""
        if not self.qdrant_client:
            st.write("Qdrant unavailable. No data to display.")
            return None
        try:
            result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False  # Omit vectors for brevity
            )
            points = result[0]
            st.write(f"Found {len(points)} points in Qdrant (showing up to {limit}):")
            for point in points:
                st.write(f"ID: {point.id}, Payload: {point.payload}")
            return points
        except Exception as e:
            st.error(f"Error listing Qdrant points: {e}")
            return None
        
        
        
    def delete_all_embeddings(self):
        try:
            self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter={})
        )
            print(f"✅ All embeddings deleted from collection: {self.collection_name}")
        except Exception as e:
            print(f"❌ Error deleting all embeddings: {e}")

        

    def analyze_document(self, document_data, query, processor: DocumentProcessor):
        """Analyze document using Qdrant Cloud and Gemini"""
        if not hasattr(self, 'model') or self.model is None:
            st.error("Model not initialized. Please check API configuration.")
            return "Error: Model not initialized"

        if not document_data or not query:
            return "Error: Document data or query is empty"

        # Print query to console
        print(f"Query received: {query}")

        # Prepare data
        is_csv = isinstance(document_data, dict) and "df" in document_data
        if is_csv:
            df = document_data["df"]
            full_text = document_data["text"]
            chunks = processor.chunk_csv(df)
            metadata = [{"type": "csv", "row_range": f"{i}-{i+10}"} for i in range(0, len(df), 10)]
        else:
            full_text = document_data
            chunks = processor.chunk_text(document_data)
            metadata = [{"type": "text"}] * len(chunks)

        # Store and search with Qdrant Cloud
        if self.qdrant_client:
            embeddings = processor.embed_chunks(chunks)
            self.store_chunks(chunks, embeddings, metadata)
            relevant_chunks = self.search_relevant_chunks(query, processor)
            context = "\n\n".join(relevant_chunks) if relevant_chunks else full_text
            # st.write(f"Debug - Using Qdrant - Relevant Chunks Count: {len(relevant_chunks)}")
        else:
            context = full_text
            st.write("Debug - Using full document context (Qdrant unavailable)")

        # st.text_area("Debug - Context Used", context, height=200)

        # Programmatic fallback for counting
        if "count" in query.lower():
            if is_csv:
                column = df.columns[0]
                counts = df[column].value_counts()
                result = "Term Counts:\n" + "\n".join(f"- {k}: {v}" for k, v in counts.items())
                return result
            else:
                terms = [t.strip() for t in full_text.split(",")]
                counts = Counter(terms)
                result = "Term Counts:\n" + "\n".join(f"- {k}: {v}" for k, v in counts.items())
                return result

        # AI-based analysis
        try:
            full_prompt = f"""
            Document Context:
            {context}

            User Query: {query}

            Instructions:
            1. Analyze the provided document context.
            2. Provide an EXACT, LITERAL response based solely on the data.
            3. Do NOT repeat the response or generate redundant text; output ONCE.
            4. For counts, list each unique term and its frequency in a concise list.
            5. Use only the data without adding extraneous information.
            6. Format as a list or table and deliver it exactly once.
            7. Write the SQl query to extract the data from the database for the query provided the table as csv_data.
            8. Do not include any other information in the response.
            """

            generation_config = {
                'temperature': 0.0,
                'max_output_tokens': 16384,
                'top_p': 1.0
            }

            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            # st.write("Debug - Raw Model Response:", response.text)
            return response.text.strip()

        except Exception as e:
            st.error(f"Document Analysis Error: {e}")
            return f"Analysis failed: {str(e)}"