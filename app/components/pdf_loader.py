import os
import boto3

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document

from app.common.logger import MedicalRAGLogger
from app.common.custom_exception import CustomException
from app.config.config import Config

from typing import List, Dict
import re
from datetime import datetime


class MedicalPDFLoader:
    """Enhanced PDF loader with semantic chunking and metadata handling"""

    def __init__(self):
        self.logger = MedicalRAGLogger(__name__)
        self.data_path = Config.DATAPATH
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP

        # Semantic chunker using AWS Bedrock Cohere embeddings (preferred)
        # Falls back to RecursiveCharacterTextSplitter when Bedrock is unavailable
        self.semantic_chunker = None
        try:
            boto3_session = boto3.Session(
                aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
                region_name=Config.AWS_REGION,
            )
            bedrock_client = boto3_session.client("bedrock-runtime")
            bedrock_embeddings = BedrockEmbeddings(
                model_id=Config.AWS_BEDROCK_EMBEDDING_MODEL,
                client=bedrock_client,
            )
            # breakpoint_threshold_type options:
            #   "percentile"         – split where cosine distance > Xth percentile (default)
            #   "standard_deviation" – split where distance > mean + X * std
            #   "interquartile"      – split based on IQR of distances
            # 95th-percentile works well for dense, varied medical text.
            self.semantic_chunker = SemanticChunker(
                embeddings=bedrock_embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95,
            )
            self.logger.logger.info("Semantic chunker initialised with AWS Bedrock Cohere embeddings")
        except Exception as e:
            self.logger.logger.warning(
                f"Could not initialise SemanticChunker ({e}). "
                "Falling back to RecursiveCharacterTextSplitter."
            )

        # Fallback: fixed-size recursive splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "; ", ", ", " "],
        )

    def extract_medical_metadata(self, text: str, page_num: int, filename: str) -> Dict:
        """
        Extract medical metadata from the text

        Args:
            text: Page content text
            page_num: Page number
            filename: Source file name
        """

        meta_data = {
            # Basic Meta Data
            "source": filename,
            "page": page_num,
            "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            # document type
            "doc_type": "medical_encyclopedia",

            # Medical categories (enhance based on book structure)
            "category": self.extract_category(text),
            "sub_category": self._extract_sub_category(text),
            
            # Content characteristics
            "has_definitions": self._contains_definitions(text),
            "has_symptoms": self._contains_symptoms(text),
            "has_treatments": self._contains_treatments(text),
            "has_diagnosis": self._contains_diagnosis(text),

            # Medical terminology density
            "medical_term_density": self._calculate_medical_density(text),

            # Section identification
            "section_type": self._identify_section_type(text),

            # Character count for filtering
            "char_count": len(text),
            "word_count": len(text.split())
        }

        # Extract specific medical entities
        entities = self._extract_medical_entities(text)
        meta_data.update(entities)

        return meta_data
    
    def extract_category(self, text: str) -> str:
        """ Extract medical category from the text """
        # Common medical categories in encyclopedias
        categories = {
            "cardiovascular": ["heart", "cardiac", "cardiovascular", "artery", "blood pressure"],
            "respiratory": ["lung", "breathing", "respiratory", "asthma", "pneumonia"],
            "neurological": ["brain", "nerve", "neurological", "stroke", "seizure"],
            "endocrine": ["diabetes", "thyroid", "hormone", "endocrine", "insulin"],
            "gastrointestinal": ["stomach", "intestine", "digestive", "liver", "gastro"],
            "musculoskeletal": ["bone", "muscle", "joint", "arthritis", "skeletal"],
            "dermatological": ["skin", "dermatology", "rash", "dermatitis"],
            "infectious": ["infection", "bacteria", "virus", "infectious", "pathogen"],
            "oncology": ["cancer", "tumor", "oncology", "malignant", "carcinoma"],
            "immunology": ["immune", "antibody", "immunology", "autoimmune"],
            "nephrology": ["kidney", "renal", "nephrology", "dialysis"],
            "hematology": ["blood", "hematology", "anemia", "leukemia"]
        }
        
        text_lower = text.lower()

        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category

        return "general"

    def _extract_sub_category(self, text: str) -> str:
        """Extract specific disease/condition from text"""
        # Try to find disease names in first 500 characters (usually title/heading)
        first_section = text[:500].lower()

        # Common patterns for disease names
        patterns = [
            r"(?:^|\n)([A-Z][a-z]+(?: [A-Z][a-z]+)*(?:'s)?(?: disease| syndrome| disorder)?)",
            r"Definition of ([A-Za-z ]+)",
            r"^([A-Z][A-Z ]+)(?:\n|:)"  # ALL CAPS headings
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text[:300])
            if match:
                return match.group(1).strip()
        
        return "unspecified"

    def _contains_definitions(self, text: str) -> bool:
        """Check if text contains medical definitions"""
        definition_indicators = [
            "is defined as", "refers to", "is a condition", "is characterized by",
            "definition:", "is the"
        ]

        text_lower = text.lower()
        return any(indicator in text_lower for indicator in definition_indicators)

    def _contains_symptoms(self, text: str) -> bool:
        """Check if text contains symptom information"""
        symptom_indicators = [
            "symptom", "signs", "manifestation", "complaint",
            "presents with", "clinical features", "onset"
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in symptom_indicators)

    def _contains_treatments(self, text: str) -> bool:
        """Check if text contains treatment information"""
        treatment_indicators = [
            "treatment", "therapy", "medication", "prescription",
            "management", "intervention", "proceedure", "surgery"
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in treatment_indicators)

    def _contains_diagnosis(self, text:str) -> bool:
        diagnosis_indicators = [
            "diagnosis", "diagnostic", "test", "examination",
            "screening", "biopsy", "imaging", "laboratory"
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in diagnosis_indicators)

    def _identify_section_type(self, text: str) -> str:
        """Identify the type of section"""
        text_lower = text.lower()[:200]  # Check first 200 chars
        
        if any(word in text_lower for word in ["definition", "overview", "introduction"]):
            return "definition"
        elif any(word in text_lower for word in ["symptoms", "signs", "clinical"]):
            return "symptoms"
        elif any(word in text_lower for word in ["treatment", "therapy", "management"]):
            return "treatment"
        elif any(word in text_lower for word in ["causes", "etiology", "risk factors"]):
            return "causes"
        elif any(word in text_lower for word in ["diagnosis", "diagnostic", "tests"]):
            return "diagnosis"
        elif any(word in text_lower for word in ["prognosis", "outlook", "recovery"]):
            return "prognosis"
        elif any(word in text_lower for word in ["prevention", "prophylaxis"]):
            return "prevention"
        else:
            return "general_information"

    def _calculate_medical_density(self, text: str) -> float:
        """Calculate density of medical terminology"""
        # Common medical suffixes/prefixes
        medical_indicators = [
            "ology", "itis", "osis", "emia", "pathy", "plasty",
            "ectomy", "otomy", "scopy", "graph", "megaly"
        ]
        
        words = text.lower().split()
        medical_word_count = sum(
            1 for word in words 
            if any(indicator in word for indicator in medical_indicators)
        )
        
        return round(medical_word_count / len(words) if words else 0, 3)
    
    def _extract_medical_entities(self, text: str) -> Dict:
        """Extract medical entities from text"""
        text_lower = text.lower()
        
        # Extract medications (basic pattern)
        medications = re.findall(r'\b[A-Z][a-z]+(?:cillin|mycin|pril|sartan|statin)\b', text)
        
        # Extract ICD codes if present
        icd_codes = re.findall(r'\b[A-Z]\d{2}(?:\.\d+)?\b', text)
        
        # Extract measurements
        measurements = re.findall(r'\d+(?:\.\d+)?\s*(?:mg|ml|mcg|units|mmol)', text_lower)
        
        return {
            "medications": list(set(medications))[:5] if medications else [],
            "icd_codes": list(set(icd_codes))[:3] if icd_codes else [],
            "has_measurements": len(measurements) > 0,
            "measurement_count": len(measurements)
        }

    def clean_medical_text(self, text: str) -> str:
        """Clean and normalize medical text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove page numbers
        text = re.sub(r'\n?\d+\n?', '', text)

        # Remove headers/footers (common patterns)
        text = re.sub(r'GALE ENCYCLOPEDIA.*?\n', '', text, flags=re.IGNORECASE)

        # Normalize bullet points
        text = re.sub(r'[•●○■□▪▫]', '- ', text)

        # Fix common OCR errors in medical texts
        text = text.replace('Ⅰ', 'I').replace('Ⅱ', 'II').replace('Ⅲ', 'III')

        return text.strip()

    def load_medical_pdf(self, filename: str) -> List[Document]:
        """
        Load PDF with enhanced metadata extraction
        """
        try:
            file_path = os.path.join(self.data_path, filename)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            loader = PyPDFLoader(file_path)
            raw_documents = loader.load()

            self.logger.log_pdf_processing(filename, len(raw_documents))

            # Enhance documents with medical metadata
            enhanced_documents = []

            for doc in raw_documents:
                # Clean text
                clean_text = self.clean_medical_text(doc.page_content)

                # Skip empty pages
                if len(clean_text.strip()) < 100:
                    continue

                # Extract medical metadata
                medical_metadata = self.extract_medical_metadata(clean_text, doc.metadata.get('page', 0), filename)

                # Merge with existing metadata
                doc.metadata.update(medical_metadata)
                doc.page_content = clean_text

                enhanced_documents.append(doc)

            self.logger.logger.info(f"Enhanced {len(enhanced_documents)} pages with medical metadata")

            return enhanced_documents

        except Exception as e:
            self.logger.log_error(e, context = f"Loading medical PDF: {filename}")
            raise

    def split_with_metadata_preservation(self, documents: List[Document]) -> List[Document]:
        """
        Split documents while preserving metadata
        Uses semantic chunking if available, otherwise falls back to recursive chunking.
        
        Args:
            documents: List of document objects
        
        Returns:
            List of chunked Documents with preserved metadata
        """
        try:
            # Check if semantic chunker is available and use it
            if self.semantic_chunker is not None:
                self.logger.logger.info("Performing semantic chunking with Bedrock embeddings...")
                chunks = self._semantic_split(documents)
                chunking_method = "semantic"
            
            else:
                self.logger.logger.info("Performing recursive character text splitting...")
                chunks = self.text_splitter.split_documents(documents)

            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = i
                chunk.metadata["chunk_size"] = len(chunk.page_content)
                chunk.metadata["total_chunks"] = len(chunks)
                chunk.metadata["chunking_method"] = chunking_method
                chunk.metadata["chunk_selection_type"] = self._identify_section_type(chunk.page_content)

            self.logger.log_embedding(len(chunks), "text-split")
            self.logger.logger.info(f"Created {len(chunks)} chunks using {chunking_method} chunking method")

            return chunks

        except Exception as e:
            self.logger.log_error(e, context = "Splitting documents with metadata preservation")
            raise

    def _semantic_split(self, documents: List[Document]) -> List[Document]:
        """
        Perform semantic chunking with metadata preservation and fallback handling

        Args:
            documents: List of documents to chunk semantically

        Returns:
            List of semantically chunked documents with preserved metadata
        """
        all_chunks = []
        failed_pages = 0

        # Cohere embed-english-v3 via Bedrock has a 2048-char per-sentence limit.
        # Pre-split any text that exceeds this before handing it to the semantic chunker.
        COHERE_MAX_CHARS = 2048

        for doc in documents:
            try:
                text = doc.page_content
                # If the whole page is within the limit, chunk normally.
                # Otherwise break it into ≤2048-char segments on sentence boundaries first.
                if len(text) > COHERE_MAX_CHARS:
                    import re
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    segments, current = [], ""
                    for sent in sentences:
                        # A single sentence longer than the limit must be hard-truncated.
                        if len(sent) > COHERE_MAX_CHARS:
                            sent = sent[:COHERE_MAX_CHARS]
                        if len(current) + len(sent) + 1 > COHERE_MAX_CHARS:
                            if current:
                                segments.append(current)
                            current = sent
                        else:
                            current = (current + " " + sent).strip() if current else sent
                    if current:
                        segments.append(current)
                    texts_to_chunk = segments
                else:
                    texts_to_chunk = [text]

                # Create semantic chunks for this document
                text_chunks = self.semantic_chunker.create_documents(
                    texts=texts_to_chunk,
                    metadatas=[doc.metadata] * len(texts_to_chunk)
                )

                # Preserve and merge original document metadata in each chunk
                for chunk in text_chunks:
                    # Merge original metadata (source, page, category, etc.)
                    chunk.metadata.update(doc.metadata)

                all_chunks.extend(text_chunks)

            except Exception as e:
                # Log warning and fall back to recursive splitting for this page
                page_num = doc.metadata.get('page', 'unknown')
                self.logger.logger.warning(
                    f"Semantic chunking failed for page {page_num}: {str(e)}. "
                    f"Falling back to recursive chunking for this page."
                )
                failed_pages += 1

                # Fallback to recursive splitting for this specific document
                try:
                    fallback_chunks = self.text_splitter.split_documents([doc])
                    all_chunks.extend(fallback_chunks)
                except Exception as fallback_error:
                    self.logger.logger.error(
                        f"Both semantic and recursive chunking failed for page {page_num}: {fallback_error}"
                    )
                    # Skip this page entirely if both methods fails
                    continue

        if failed_pages > 0:
            self.logger.logger.warning(
                f"Semantic chunking failed for {failed_pages} pages. "
                f"Used recursive fallback for those pages"
            )

        return all_chunks

    def process_medical_pdf(self, filename: str) -> List[Document]:
        """
        Complete pipeline: Load + Enhance + Split

        Args:
            filename: PDF filename
        
        Returns:
            List of processed document chunks
        """
        self.logger.logger.info(f"Processing medical PDF: {filename}")

        # Load with medical metadata
        documents = self.load_medical_pdf(filename)

        # Split while preserving metadata
        chunks = self.split_with_metadata_preservation(documents)

        self.logger.logger.info(f"Successfully processed {len(chunks)} chunks from {filename}")

        return chunks

    def filter_chunks_by_metadata(self, chunks: List[Document], category: str = None, section_type: str = None, min_medical_density: float = 0.0) -> List[Document]:
        """
        Filter chunks based on metadata criteria

        Args:
            chunks: List of document chunks
            category: Filter by medical category
            section_type: Filter by section type
            min_medical_density: Minimum medical density thresold

        Returns:
            Filtered list of document chunks
        """
        filtered = list(chunks)

        if category:
            filtered = [chunk for chunk in filtered if chunk.metadata.get('category') == category]

        if section_type:
            filtered = [chunk for chunk in filtered if chunk.metadata.get('section_type') == section_type]

        if min_medical_density > 0.5:
            filtered = [chunk for chunk in filtered if chunk.metadata.get('medical_term_density', 0) >= min_medical_density]

        self.logger.logger.info(f"Filtered {len(chunks)} chunks to {len(filtered)} chunks")
        return filtered

    def get_metadata_summary(self, chunks: List[Document]) -> Dict:
        """
        Get summary of metadata across all chunks

        Args:
            chunks: List of Document chunks

        Returns:
            Summary statistics dictionary
        """
        from collections import Counter

        categories = Counter(chunk.metadata.get('category', 'general') for chunk in chunks)
        section_types = Counter(chunk.metadata.get('section_type', 'general_information') for chunk in chunks)

        summary = {
            "total_chunks": len(chunks),
            "categories": dict(categories),
            "section_types": dict(section_types),
            "chunks_with_symptoms": sum(
                1
                for chunk in chunks
                if chunk.metadata.get("section_type", "general_information") == "symptoms"
            ),
            "chunks_with_treatments": sum(
                1
                for chunk in chunks
                if chunk.metadata.get("section_type", "general_information") == "treatment"
            ),
            "chunks_with_diagnosis": sum(
                1
                for chunk in chunks
                if chunk.metadata.get("section_type", "general_information") == "diagnosis"
            ),
            "avg_medical_density": round(
                sum(chunk.metadata.get("medical_term_density", 0) for chunk in chunks)
                / len(chunks),
                3,
            )
            if chunks
            else 0,
        }

        return summary

# Usage example
if __name__ == "__main__":
    print("Starting Medical RAG Pipeline...")
    loader = MedicalPDFLoader()

    try:
        # Process medical Encyclopedia
        chunks = loader.process_medical_pdf("The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf")

        # Get metadata summary
        summary = loader.get_metadata_summary(chunks)

        print(f"\n{'=' * 70}")
        print("METADATA SUMMARY:")
        print(f"\n{'=' * 70}")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Show sample chunks with metadata
        print(f"\n{'=' * 70}")
        print("SAMPLE CHUNKS WITH METADATA")
        print("\n" + "=" * 70)

        for i, chunk in enumerate(chunks[:3]):
            print(f"--- Chunk {i+1} ---")
            print(f"Content: {chunk.page_content[:200]}...")
            print(f"\nMetadata: ")
            for key, value in chunk.metadata.items():
                print(f"{key}: {value}")
            print()

        # Filter Examples
        print(f"\n{'='*60}")
        print("FILTERING EXAMPLES")
        print(f"{'='*60}\n")
        
        # Get only cardiovascular chunks
        cardio_chunks = loader.filter_chunks_by_metadata(
            chunks,
            category="cardiovascular"
        )
        print(f"Cardiovascular chunks: {len(cardio_chunks)}")
        
        # Get only treatment sections
        treatment_chunks = loader.filter_chunks_by_metadata(
            chunks,
            section_type="treatment"
        )
        print(f"Treatment section chunks: {len(treatment_chunks)}")
        
    except Exception as e:
        print(f"Error: {e}")