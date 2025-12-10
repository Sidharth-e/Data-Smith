"""
Document Splitter for Data Smith.
Implements multiple chunking strategies for LLM-compatible processing.
"""

import re
from typing import List, Literal, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

# Token counting (optional dependency)
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class SplitStrategy(str, Enum):
    """Available splitting strategies."""
    CHARACTER = "character"
    TOKEN = "token"
    SEMANTIC = "semantic"
    SLIDING_WINDOW = "sliding_window"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"


@dataclass
class DocumentChunk:
    """A chunk of document content."""
    chunk_id: int
    content: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def char_count(self) -> int:
        return len(self.content)
    
    @property
    def word_count(self) -> int:
        return len(self.content.split())


@dataclass
class SplitterConfig:
    """Configuration for document splitting."""
    strategy: SplitStrategy = SplitStrategy.SEMANTIC
    chunk_size: int = 1000
    chunk_overlap: int = 100
    min_chunk_size: int = 50
    max_chunk_size: int = 2000
    preserve_sentences: bool = True
    encoding_name: str = "cl100k_base"  # For tiktoken


class DocumentSplitter:
    """
    Split documents into chunks using various strategies.
    
    Strategies:
    - CHARACTER: Fixed character count chunks
    - TOKEN: Token-based splitting for LLM compatibility
    - SEMANTIC: Sentence/paragraph boundary-aware splitting
    - SLIDING_WINDOW: Overlapping chunks with context preservation
    - PARAGRAPH: Split by paragraphs
    - SENTENCE: Split by sentences
    """
    
    def __init__(self, config: Optional[SplitterConfig] = None):
        """
        Initialize the document splitter.
        
        Args:
            config: Splitter configuration (optional)
        """
        self.config = config or SplitterConfig()
        self._tokenizer = None
        
        if TIKTOKEN_AVAILABLE and self.config.strategy == SplitStrategy.TOKEN:
            try:
                self._tokenizer = tiktoken.get_encoding(self.config.encoding_name)
            except Exception:
                pass
    
    def split(
        self,
        text: str,
        strategy: Optional[SplitStrategy] = None,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[DocumentChunk]:
        """
        Split text into chunks using the specified strategy.
        
        Args:
            text: Text content to split
            strategy: Splitting strategy (overrides config)
            chunk_size: Target chunk size (overrides config)
            overlap: Chunk overlap (overrides config)
            
        Returns:
            List of DocumentChunk objects
        """
        if not text or not text.strip():
            return []
        
        # Use overrides or config defaults
        strategy = strategy or self.config.strategy
        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.chunk_overlap
        
        # Clean text
        text = self._normalize_whitespace(text)
        
        # Split based on strategy
        if strategy == SplitStrategy.CHARACTER:
            return self._split_by_character(text, chunk_size, overlap)
        elif strategy == SplitStrategy.TOKEN:
            return self._split_by_token(text, chunk_size, overlap)
        elif strategy == SplitStrategy.SEMANTIC:
            return self._split_semantic(text, chunk_size, overlap)
        elif strategy == SplitStrategy.SLIDING_WINDOW:
            return self._split_sliding_window(text, chunk_size, overlap)
        elif strategy == SplitStrategy.PARAGRAPH:
            return self._split_by_paragraph(text, chunk_size)
        elif strategy == SplitStrategy.SENTENCE:
            return self._split_by_sentence(text, chunk_size)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving paragraph breaks."""
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse multiple spaces
        text = re.sub(r"[ \t]+", " ", text)
        # Normalize multiple newlines (preserve double for paragraphs)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    
    def _split_by_character(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int
    ) -> List[DocumentChunk]:
        """Split text by character count with overlap."""
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Adjust end to not break words
            if end < len(text):
                # Find last space before end
                space_pos = text.rfind(" ", start, end)
                if space_pos > start:
                    end = space_pos
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    start_char=start,
                    end_char=end,
                    metadata={"strategy": "character"}
                ))
                chunk_id += 1
            
            # Move start with overlap
            start = end - overlap if end < len(text) else end
            if start >= len(text):
                break
        
        return chunks
    
    def _split_by_token(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int
    ) -> List[DocumentChunk]:
        """Split text by token count (LLM-compatible)."""
        if not TIKTOKEN_AVAILABLE or not self._tokenizer:
            # Fall back to approximate token count (4 chars per token)
            char_chunk_size = chunk_size * 4
            char_overlap = overlap * 4
            return self._split_by_character(text, char_chunk_size, char_overlap)
        
        chunks = []
        tokens = self._tokenizer.encode(text)
        chunk_id = 0
        start_token = 0
        
        while start_token < len(tokens):
            end_token = start_token + chunk_size
            
            if end_token > len(tokens):
                end_token = len(tokens)
            
            chunk_tokens = tokens[start_token:end_token]
            chunk_text = self._tokenizer.decode(chunk_tokens).strip()
            
            if chunk_text:
                # Calculate character positions
                start_char = len(self._tokenizer.decode(tokens[:start_token]))
                end_char = len(self._tokenizer.decode(tokens[:end_token]))
                
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={"strategy": "token", "token_count": len(chunk_tokens)}
                ))
                chunk_id += 1
            
            start_token = end_token - overlap
            if start_token >= len(tokens):
                break
        
        return chunks
    
    def _split_semantic(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int
    ) -> List[DocumentChunk]:
        """Split text respecting sentence and paragraph boundaries."""
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0
        start_char = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If single sentence exceeds chunk_size, split it
            if sentence_len > chunk_size:
                # Flush current chunk
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        content=chunk_text,
                        start_char=start_char,
                        end_char=start_char + len(chunk_text),
                        metadata={"strategy": "semantic"}
                    ))
                    chunk_id += 1
                    start_char += len(chunk_text) + 1
                    current_chunk = []
                    current_size = 0
                
                # Split long sentence by character
                long_chunks = self._split_by_character(sentence, chunk_size, overlap)
                for lc in long_chunks:
                    lc.chunk_id = chunk_id
                    lc.start_char = start_char + lc.start_char
                    lc.end_char = start_char + lc.end_char
                    chunks.append(lc)
                    chunk_id += 1
                start_char += sentence_len + 1
                continue
            
            # Check if adding sentence exceeds chunk size
            if current_size + sentence_len + 1 > chunk_size and current_chunk:
                # Flush current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    metadata={"strategy": "semantic"}
                ))
                chunk_id += 1
                
                # Keep overlap (last few sentences)
                if overlap > 0:
                    overlap_sentences = []
                    overlap_size = 0
                    for s in reversed(current_chunk):
                        if overlap_size + len(s) > overlap:
                            break
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s) + 1
                    current_chunk = overlap_sentences
                    current_size = overlap_size
                    start_char = start_char + len(chunk_text) - overlap_size
                else:
                    current_chunk = []
                    current_size = 0
                    start_char += len(chunk_text) + 1
            
            current_chunk.append(sentence)
            current_size += sentence_len + 1
        
        # Flush remaining
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                metadata={"strategy": "semantic"}
            ))
        
        return chunks
    
    def _split_sliding_window(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int
    ) -> List[DocumentChunk]:
        """Split with sliding window for maximum context preservation."""
        # Use larger overlap for sliding window
        effective_overlap = max(overlap, chunk_size // 3)
        return self._split_by_character(text, chunk_size, effective_overlap)
    
    def _split_by_paragraph(
        self, 
        text: str, 
        max_chunk_size: int
    ) -> List[DocumentChunk]:
        """Split text by paragraphs."""
        paragraphs = re.split(r"\n\n+", text)
        chunks = []
        chunk_id = 0
        current_pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(para) > max_chunk_size:
                # Split long paragraphs
                sub_chunks = self._split_by_character(para, max_chunk_size, 0)
                for sc in sub_chunks:
                    sc.chunk_id = chunk_id
                    sc.start_char = current_pos + sc.start_char
                    sc.end_char = current_pos + sc.end_char
                    sc.metadata["strategy"] = "paragraph"
                    chunks.append(sc)
                    chunk_id += 1
            else:
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    content=para,
                    start_char=current_pos,
                    end_char=current_pos + len(para),
                    metadata={"strategy": "paragraph"}
                ))
                chunk_id += 1
            
            current_pos += len(para) + 2  # +2 for paragraph break
        
        return chunks
    
    def _split_by_sentence(
        self, 
        text: str, 
        max_chunk_size: int
    ) -> List[DocumentChunk]:
        """Split text by sentences."""
        sentences = self._split_into_sentences(text)
        chunks = []
        chunk_id = 0
        current_pos = 0
        
        for sentence in sentences:
            if len(sentence) > max_chunk_size:
                # Split long sentences
                sub_chunks = self._split_by_character(sentence, max_chunk_size, 0)
                for sc in sub_chunks:
                    sc.chunk_id = chunk_id
                    sc.start_char = current_pos + sc.start_char
                    sc.end_char = current_pos + sc.end_char
                    sc.metadata["strategy"] = "sentence"
                    chunks.append(sc)
                    chunk_id += 1
            else:
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    content=sentence,
                    start_char=current_pos,
                    end_char=current_pos + len(sentence),
                    metadata={"strategy": "sentence"}
                ))
                chunk_id += 1
            
            current_pos += len(sentence) + 1
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Pattern matches sentence-ending punctuation followed by space or end
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def estimate_chunks(
        self, 
        text: str, 
        strategy: Optional[SplitStrategy] = None,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> int:
        """Estimate number of chunks without actually splitting."""
        strategy = strategy or self.config.strategy
        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.chunk_overlap
        
        text_len = len(text)
        effective_chunk_size = chunk_size - overlap
        
        if effective_chunk_size <= 0:
            effective_chunk_size = chunk_size // 2
        
        return max(1, (text_len + effective_chunk_size - 1) // effective_chunk_size)


# Convenience function
def split_document(
    text: str,
    strategy: str = "semantic",
    chunk_size: int = 1000,
    overlap: int = 100
) -> List[DocumentChunk]:
    """
    Quick function to split a document.
    
    Args:
        text: Text content to split
        strategy: Splitting strategy name
        chunk_size: Target chunk size
        overlap: Chunk overlap
        
    Returns:
        List of DocumentChunk objects
    """
    config = SplitterConfig(
        strategy=SplitStrategy(strategy),
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    splitter = DocumentSplitter(config)
    return splitter.split(text)
