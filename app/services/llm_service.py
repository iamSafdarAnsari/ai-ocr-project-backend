"""
Advanced LLM Service for AI-Powered Document Processing
Provides intelligent text analysis, enhancement, and generation capabilities
"""

import os
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import re

# Guard OpenAI import to allow running without the SDK available
try:
    import openai
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None
    _OPENAI_AVAILABLE = False
import json
import re


class ModelType(Enum):
    """Available AI models for different tasks"""
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo-preview"
    GPT35_TURBO = "gpt-3.5-turbo"
    GPT4_VISION = "gpt-4-vision-preview"


class TaskType(Enum):
    """Types of AI tasks supported"""
    TEXT_ENHANCEMENT = "text_enhancement"
    CONTEXT_ANALYSIS = "context_analysis"
    ENTITY_EXTRACTION = "entity_extraction"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    DOCUMENT_SUMMARIZATION = "document_summarization"
    TRANSLATION_QUALITY = "translation_quality"
    OCR_CORRECTION = "ocr_correction"
    CONTENT_GENERATION = "content_generation"


@dataclass
class AIResponse:
    """Structured response from AI models"""
    content: str
    confidence: float
    metadata: Dict[str, Any]
    model_used: str
    task_type: TaskType


@dataclass
class DocumentContext:
    """Context information for document processing"""
    document_type: str
    language: str
    domain: Optional[str] = None
    confidence_score: Optional[float] = None
    entities: Optional[List[Dict]] = None
    sentiment: Optional[Dict] = None


class EnhancedLLMService:
    """Advanced LLM service with multiple AI capabilities"""

    def __init__(self):
        self.client = None
        # Initialize client only if OpenAI SDK is available and API key set
        self._initialize_client()
        self.default_model = ModelType.GPT4_TURBO
        self.max_tokens = 4000
        self.temperature = 0.3

    def _initialize_client(self):
        """Initialize OpenAI client"""
        api_key = os.getenv("OPENAI_API_KEY")
        if _OPENAI_AVAILABLE and api_key:
            try:
                self.client = OpenAI(api_key=api_key)
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            if not _OPENAI_AVAILABLE:
                print("OpenAI SDK not available. LLM features will be limited.")
            else:
                print("Warning: OPENAI_API_KEY not found. LLM features will be limited.")

    async def analyze_document_context(self, text: str, document_type: str = "general") -> DocumentContext:
        """Analyze document context and extract metadata"""
        if not self.client:
            return DocumentContext(document_type=document_type, language="unknown")

        try:
            prompt = f"""
            Analyze the following text and provide a JSON response with:
            1. document_type: Type of document (letter, form, invoice, etc.)
            2. language: Primary language (ISO code)
            3. domain: Subject domain (legal, medical, business, etc.)
            4. confidence_score: Confidence in analysis (0-1)
            5. entities: List of named entities found
            6. sentiment: Overall sentiment analysis

            Text: {text[:2000]}

            Respond only with valid JSON.
            """

            response = await self._make_ai_request(
                prompt=prompt,
                model=ModelType.GPT4_TURBO,
                task_type=TaskType.CONTEXT_ANALYSIS
            )

            try:
                analysis = json.loads(response.content)
                return DocumentContext(
                    document_type=analysis.get("document_type", document_type),
                    language=analysis.get("language", "unknown"),
                    domain=analysis.get("domain"),
                    confidence_score=analysis.get("confidence_score", 0.5),
                    entities=analysis.get("entities", []),
                    sentiment=analysis.get("sentiment", {})
                )
            except json.JSONDecodeError:
                return DocumentContext(document_type=document_type, language="unknown")

        except Exception as e:
            print(f"Context analysis failed: {e}")
            return DocumentContext(document_type=document_type, language="unknown")

    async def enhance_extracted_text(self, text: str, context: DocumentContext) -> AIResponse:
        """Enhance OCR-extracted text using AI"""
        if not self.client:
            return AIResponse(
                content=text,
                confidence=0.5,
                metadata={"method": "fallback"},
                model_used="none",
                task_type=TaskType.TEXT_ENHANCEMENT
            )

        prompt = f"""
        You are an expert text enhancement AI. Improve the following OCR-extracted text by:
        1. Correcting obvious OCR errors and typos
        2. Fixing formatting and spacing issues
        3. Maintaining the original meaning and structure
        4. Preserving technical terms and proper nouns

        Document Context:
        - Type: {context.document_type}
        - Language: {context.language}
        - Domain: {context.domain or 'general'}

        Original Text:
        {text}

        Provide only the enhanced text without explanations.
        """

        return await self._make_ai_request(
            prompt=prompt,
            model=ModelType.GPT4_TURBO,
            task_type=TaskType.TEXT_ENHANCEMENT
        )

    async def correct_ocr_errors(self, text: str, confidence_scores: Optional[List[float]] = None) -> AIResponse:
        """Correct OCR errors using AI with confidence-based processing"""
        if not self.client:
            return AIResponse(
                content=text,
                confidence=0.5,
                metadata={"method": "fallback"},
                model_used="none",
                task_type=TaskType.OCR_CORRECTION
            )

        prompt = f"""
        You are an OCR error correction specialist. Fix the following text that was extracted using OCR:

        Rules:
        1. Correct obvious character recognition errors (e.g., 'rn' -> 'm', '0' -> 'O')
        2. Fix spacing and punctuation issues
        3. Maintain original formatting structure
        4. Don't change technical terms or proper nouns unless clearly wrong
        5. Preserve numbers and dates accurately

        Text to correct:
        {text}

        Return only the corrected text.
        """

        return await self._make_ai_request(
            prompt=prompt,
            model=ModelType.GPT4_TURBO,
            task_type=TaskType.OCR_CORRECTION
        )

    async def extract_entities(self, text: str) -> AIResponse:
        """Extract named entities from text"""
        if not self.client:
            return AIResponse(
                content="[]",
                confidence=0.5,
                metadata={"method": "fallback"},
                model_used="none",
                task_type=TaskType.ENTITY_EXTRACTION
            )

        prompt = f"""
        Extract named entities from the following text and return as JSON array.
        Include: persons, organizations, locations, dates, amounts, phone numbers, emails.

        Format: [{{"type": "PERSON", "text": "John Doe", "start": 0, "end": 8}}]

        Text: {text}

        Return only valid JSON array.
        """

        return await self._make_ai_request(
            prompt=prompt,
            model=ModelType.GPT4_TURBO,
            task_type=TaskType.ENTITY_EXTRACTION
        )

    async def analyze_sentiment(self, text: str) -> AIResponse:
        """Analyze sentiment of the text"""
        if not self.client:
            return AIResponse(
                content='{"sentiment": "neutral", "confidence": 0.5}',
                confidence=0.5,
                metadata={"method": "fallback"},
                model_used="none",
                task_type=TaskType.SENTIMENT_ANALYSIS
            )

        prompt = f"""
        Analyze the sentiment of the following text and return JSON:
        {{
            "sentiment": "positive|negative|neutral",
            "confidence": 0.0-1.0,
            "emotions": ["joy", "anger", "fear", etc.],
            "tone": "formal|informal|professional|casual"
        }}

        Text: {text}

        Return only valid JSON.
        """

        return await self._make_ai_request(
            prompt=prompt,
            model=ModelType.GPT35_TURBO,
            task_type=TaskType.SENTIMENT_ANALYSIS
        )

    async def summarize_document(self, text: str, max_length: int = 200) -> AIResponse:
        """Generate document summary"""
        if not self.client:
            return AIResponse(
                content=text[:max_length] + "...",
                confidence=0.5,
                metadata={"method": "fallback"},
                model_used="none",
                task_type=TaskType.DOCUMENT_SUMMARIZATION
            )

        prompt = f"""
        Create a concise summary of the following document in approximately {max_length} characters.
        Focus on key information, main points, and important details.

        Document: {text}

        Summary:
        """

        return await self._make_ai_request(
            prompt=prompt,
            model=ModelType.GPT4_TURBO,
            task_type=TaskType.DOCUMENT_SUMMARIZATION
        )

    async def assess_translation_quality(self, original: str, translated: str, target_language: str) -> AIResponse:
        """Assess and improve translation quality"""
        if not self.client:
            return AIResponse(
                content=translated,
                confidence=0.5,
                metadata={"method": "fallback"},
                model_used="none",
                task_type=TaskType.TRANSLATION_QUALITY
            )

        prompt = f"""
        Assess and improve this translation quality:

        Original: {original}
        Translation: {translated}
        Target Language: {target_language}

        Provide:
        1. Quality score (0-10)
        2. Improved translation if needed
        3. Issues found

        Return JSON: {{"score": 8, "improved_translation": "...", "issues": ["..."]}}
        """

        return await self._make_ai_request(
            prompt=prompt,
            model=ModelType.GPT4_TURBO,
            task_type=TaskType.TRANSLATION_QUALITY
        )

    async def _make_ai_request(self, prompt: str, model: ModelType, task_type: TaskType,
                              temperature: Optional[float] = None) -> AIResponse:
        """Make request to AI model with error handling"""
        if not self.client:
            return AIResponse(
                content="AI service unavailable",
                confidence=0.0,
                metadata={"error": "No API key"},
                model_used="none",
                task_type=task_type
            )

        try:
            response = self.client.chat.completions.create(
                model=model.value,
                messages=[
                    {"role": "system", "content": "You are an expert AI assistant specialized in document processing and text analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=temperature or self.temperature
            )

            content = response.choices[0].message.content
            confidence = self._calculate_confidence(response, task_type)

            return AIResponse(
                content=content,
                confidence=confidence,
                metadata={
                    "tokens_used": response.usage.total_tokens,
                    "model": model.value,
                    "finish_reason": response.choices[0].finish_reason
                },
                model_used=model.value,
                task_type=task_type
            )

        except Exception as e:
            print(f"AI request failed: {e}")
            return AIResponse(
                content=f"Error: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e)},
                model_used=model.value,
                task_type=task_type
            )

    def _calculate_confidence(self, response, task_type: TaskType) -> float:
        """Calculate confidence score based on response characteristics"""
        base_confidence = 0.8

        # Adjust based on finish reason
        if hasattr(response.choices[0], 'finish_reason'):
            if response.choices[0].finish_reason == "stop":
                base_confidence += 0.1
            elif response.choices[0].finish_reason == "length":
                base_confidence -= 0.2

        # Adjust based on task type
        task_confidence_modifiers = {
            TaskType.TEXT_ENHANCEMENT: 0.9,
            TaskType.OCR_CORRECTION: 0.85,
            TaskType.CONTEXT_ANALYSIS: 0.8,
            TaskType.ENTITY_EXTRACTION: 0.75,
            TaskType.SENTIMENT_ANALYSIS: 0.7,
            TaskType.TRANSLATION_QUALITY: 0.8,
            TaskType.DOCUMENT_SUMMARIZATION: 0.75
        }

        modifier = task_confidence_modifiers.get(task_type, 0.7)
        return min(1.0, base_confidence * modifier)

    async def process_document_batch(self, texts: List[str], task_type: TaskType) -> List[AIResponse]:
        """Process multiple texts in batch"""
        tasks = [self._make_ai_request(text, self.default_model, task_type) for text in texts]
        return await asyncio.gather(*tasks)

    def get_supported_models(self) -> List[str]:
        """Get list of supported AI models"""
        return [model.value for model in ModelType]

    def get_supported_tasks(self) -> List[str]:
        """Get list of supported AI tasks"""
        return [task.value for task in TaskType]


# Global instance
llm_service = EnhancedLLMService()