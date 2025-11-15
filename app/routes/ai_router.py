"""
AI-Powered Features Router
Provides endpoints for advanced AI capabilities including LLM processing, NLP analysis, and intelligent document enhancement
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import asyncio

from app.services.llm_service import llm_service, TaskType, DocumentContext
from app.services.nlp_service import nlp_service
from app.services.translation_service import translation_service

router = APIRouter()

# Request/Response Models
class TextAnalysisRequest(BaseModel):
    text: str
    include_statistics: bool = True
    include_quality: bool = True
    include_keywords: bool = True
    include_structure: bool = True

class TextAnalysisResponse(BaseModel):
    statistics: Optional[Dict[str, Any]] = None
    quality: Optional[Dict[str, Any]] = None
    keywords: Optional[List[Dict[str, Any]]] = None
    structure: Optional[Dict[str, Any]] = None

class TextEnhancementRequest(BaseModel):
    text: str
    document_type: str = "general"
    enhancement_type: str = "general"  # general, ocr_correction, grammar_fix
    context: Optional[str] = None

class TextEnhancementResponse(BaseModel):
    original_text: str
    enhanced_text: str
    confidence: float
    improvements: List[str]
    metadata: Dict[str, Any]

class ContextAnalysisRequest(BaseModel):
    text: str
    document_type: str = "general"

class ContextAnalysisResponse(BaseModel):
    document_type: str
    language: str
    domain: Optional[str]
    confidence_score: float
    entities: List[Dict[str, Any]]
    sentiment: Dict[str, Any]

class BatchTranslationRequest(BaseModel):
    texts: List[str]
    source_language: str = "auto"
    target_language: str = "en"
    domain: str = "general"
    context: Optional[str] = None

class TranslationAlternativesRequest(BaseModel):
    text: str
    source_language: str
    target_language: str
    count: int = 3

@router.post("/analyze-text", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """Comprehensive text analysis using NLP"""
    try:
        result = TextAnalysisResponse()
        
        if request.include_statistics:
            stats = nlp_service.analyze_text_statistics(request.text)
            result.statistics = {
                "word_count": stats.word_count,
                "sentence_count": stats.sentence_count,
                "paragraph_count": stats.paragraph_count,
                "character_count": stats.character_count,
                "average_word_length": stats.average_word_length,
                "readability_score": stats.readability_score,
                "language_confidence": stats.language_confidence
            }
        
        if request.include_quality:
            quality = nlp_service.assess_text_quality(request.text)
            result.quality = {
                "overall_score": quality.overall_score,
                "spelling_errors": quality.spelling_errors,
                "grammar_issues": quality.grammar_issues,
                "formatting_issues": quality.formatting_issues,
                "suggestions": quality.suggestions
            }
        
        if request.include_keywords:
            keywords = nlp_service.extract_keywords(request.text)
            result.keywords = [{"word": word, "score": score} for word, score in keywords]
        
        if request.include_structure:
            result.structure = nlp_service.detect_document_structure(request.text)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")

@router.post("/enhance-text", response_model=TextEnhancementResponse)
async def enhance_text(request: TextEnhancementRequest):
    """Enhance text using AI-powered processing"""
    try:
        # Analyze document context first
        context = await llm_service.analyze_document_context(request.text, request.document_type)
        
        # Choose enhancement method based on type
        if request.enhancement_type == "ocr_correction":
            result = await llm_service.correct_ocr_errors(request.text)
        elif request.enhancement_type == "grammar_fix":
            # Use NLP service for basic corrections, then AI enhancement
            corrected = nlp_service.correct_ocr_errors(request.text)
            result = await llm_service.enhance_extracted_text(corrected, context)
        else:
            result = await llm_service.enhance_extracted_text(request.text, context)
        
        # Identify improvements made
        improvements = []
        if len(result.content) != len(request.text):
            improvements.append("Text length adjusted")
        if result.content != request.text:
            improvements.append("Content enhanced")
        
        return TextEnhancementResponse(
            original_text=request.text,
            enhanced_text=result.content,
            confidence=result.confidence,
            improvements=improvements,
            metadata=result.metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text enhancement failed: {str(e)}")

@router.post("/analyze-context", response_model=ContextAnalysisResponse)
async def analyze_context(request: ContextAnalysisRequest):
    """Analyze document context and extract metadata"""
    try:
        context = await llm_service.analyze_document_context(request.text, request.document_type)
        
        return ContextAnalysisResponse(
            document_type=context.document_type,
            language=context.language,
            domain=context.domain,
            confidence_score=context.confidence_score or 0.0,
            entities=context.entities or [],
            sentiment=context.sentiment or {}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context analysis failed: {str(e)}")

@router.post("/extract-entities")
async def extract_entities(text: str = Form(...)):
    """Extract named entities from text"""
    try:
        result = await llm_service.extract_entities(text)
        
        # Parse JSON response
        import json
        try:
            entities = json.loads(result.content)
            return {
                "entities": entities,
                "confidence": result.confidence,
                "metadata": result.metadata
            }
        except json.JSONDecodeError:
            return {
                "entities": [],
                "confidence": 0.0,
                "error": "Failed to parse entity extraction results"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {str(e)}")

@router.post("/analyze-sentiment")
async def analyze_sentiment(text: str = Form(...)):
    """Analyze text sentiment"""
    try:
        result = await llm_service.analyze_sentiment(text)
        
        # Parse JSON response
        import json
        try:
            sentiment = json.loads(result.content)
            return {
                "sentiment": sentiment,
                "confidence": result.confidence,
                "metadata": result.metadata
            }
        except json.JSONDecodeError:
            return {
                "sentiment": {"sentiment": "neutral", "confidence": 0.5},
                "confidence": 0.5,
                "error": "Failed to parse sentiment analysis results"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

@router.post("/summarize")
async def summarize_document(text: str = Form(...), max_length: int = Form(200)):
    """Generate document summary"""
    try:
        result = await llm_service.summarize_document(text, max_length)
        
        return {
            "summary": result.content,
            "confidence": result.confidence,
            "metadata": result.metadata,
            "original_length": len(text),
            "summary_length": len(result.content)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document summarization failed: {str(e)}")

@router.post("/translate-batch")
async def translate_batch(request: BatchTranslationRequest):
    """Translate multiple texts in batch"""
    try:
        results = await translation_service.translate_batch(
            texts=request.texts,
            source_lang=request.source_language,
            target_lang=request.target_language,
            domain=request.domain
        )
        
        return {
            "translations": [
                {
                    "original": request.texts[i],
                    "translated": result.translated_text,
                    "confidence": result.confidence_score,
                    "quality": result.quality_assessment.value,
                    "issues": result.issues or []
                }
                for i, result in enumerate(results)
            ],
            "total_processed": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch translation failed: {str(e)}")

@router.post("/translation-alternatives")
async def get_translation_alternatives(request: TranslationAlternativesRequest):
    """Get multiple translation alternatives"""
    try:
        alternatives = await translation_service.get_translation_alternatives(
            text=request.text,
            source_lang=request.source_language,
            target_lang=request.target_language,
            count=request.count
        )
        
        return {
            "original": request.text,
            "alternatives": alternatives,
            "count": len(alternatives)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation alternatives failed: {str(e)}")

@router.get("/supported-models")
async def get_supported_models():
    """Get supported AI models and tasks"""
    return {
        "models": llm_service.get_supported_models(),
        "tasks": llm_service.get_supported_tasks(),
        "languages": translation_service.get_supported_languages(),
        "domains": translation_service.get_domain_contexts()
    }

@router.get("/health")
async def health_check():
    """Health check for AI services"""
    return {
        "status": "healthy",
        "services": {
            "llm_service": "available" if llm_service.client else "limited",
            "nlp_service": "available",
            "translation_service": "available" if translation_service.client else "limited"
        }
    }
