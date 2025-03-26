import pandas as pd
import PyPDF2
import logging
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from io import StringIO, BytesIO
import io
import nest_asyncio
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentProcessor:
    def __init__(self):
        """Initialize embedding model"""
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model
        self.logger = logging.getLogger(__name__)
        self.logger.info("DocumentProcessor initialized with SentenceTransformer.")

    @staticmethod
    def read_pdf(file) -> str:
        """Read and extract text from a PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                text += extracted if extracted else ""
            logging.debug(f"Extracted text length from PDF: {len(text)}")
            return text.strip()
        except Exception as e:
            logging.error(f"PDF Processing Error: {e}")
            return None

    @staticmethod
    def read_csv(file) -> Dict[str, Any]:
        """
        Reads a CSV file by decoding its bytes using 'utf-8-sig' (handles BOM),
        then uses StringIO to create a file-like object for pandas.
        Returns a dictionary containing the DataFrame, a text version, and JSON representation.
        """
        try:
            # Ensure file pointer is at the start
            file.seek(0)
            content = file.read()
            if not content:
                logging.error("The file content is empty.")
                return None

            # Debug: Print a snippet of the content
            snippet = content[:200]
            logging.debug(f"File content snippet: {snippet}")

            # Decode using 'utf-8-sig' to handle BOM if present
            decoded = content.decode('utf-8-sig')
            csv_io = StringIO(decoded)

            # Attempt to read using comma delimiter first
            logging.debug("Attempting to read CSV with 'utf-8-sig' and comma delimiter...")
            try:
                df = pd.read_csv(csv_io, delimiter=',')
            except Exception as e:
                logging.warning(f"Failed to read CSV with comma delimiter: {e}")
                df = None

            # If reading with comma fails, try semicolon delimiter
            if df is None or df.empty or df.columns.size == 0:
                logging.debug("Attempting to read CSV with semicolon delimiter...")
                csv_io.seek(0)
                try:
                    df = pd.read_csv(csv_io, delimiter=';')
                except Exception as e:
                    logging.warning(f"Failed to read CSV with semicolon delimiter: {e}")
                    df = None

            if df is None or df.empty or df.columns.size == 0:
                logging.error("Failed to read the CSV file: No columns to parse from file.")
                return None

            logging.debug(f"Successfully read CSV file with {len(df)} rows and {len(df.columns)} columns.")

            return {
                "df": df,
                "text": df.to_string(index=False),
                "json": df.to_json(orient="records")
            }

        except Exception as e:
            logging.error(f"CSV Processing Error: {e}")
            return None

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks."""
        self.logger.info(f"Chunking text into chunks of size {chunk_size}")
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word) + 1
            else:
                current_chunk.append(word)
                current_length += len(word) + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        self.logger.info(f"Text chunked into {len(chunks)} chunks")
        return chunks

    def chunk_csv(self, df: pd.DataFrame, chunk_size: int = 10) -> List[str]:
        """Chunk CSV rows into groups."""
        self.logger.info(f"Chunking CSV rows into groups of size {chunk_size}")
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i + chunk_size]
            chunks.append(chunk_df.to_string(index=False))
        self.logger.info(f"CSV rows chunked into {len(chunks)} groups")
        return chunks

    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for chunks."""
        self.logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = self.embedder.encode(chunks, convert_to_tensor=False).tolist()
        self.logger.info(f"Embeddings generated for {len(chunks)} chunks")
        return embeddings
