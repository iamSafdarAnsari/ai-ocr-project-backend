from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import os
import uuid
from typing import List, Optional
from pydantic import BaseModel

from app.services.ocr_service import OCRService
from app.services.pdf_service import PDFService
from app.services.llm_service import llm_service
from app.services.nlp_service import nlp_service
from app.services.translation_service import translation_service

class ProcessingOptions(BaseModel):
    enhance_text: bool = True
    analyze_context: bool = True
    extract_entities: bool = False
    translate: bool = False
    target_language: str = "en"
    domain: str = "general"

router = APIRouter()
ocr_service = OCRService()
pdf_service = PDFService()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document for processing"""
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(upload_dir, unique_filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return JSONResponse({
            "message": "Document uploaded successfully",
            "filename": unique_filename,
            "original_name": file.filename
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/process")
async def process_document(
    filename: str = Form(...),
    enhance_text: bool = Form(True),
    analyze_context: bool = Form(True),
    extract_entities: bool = Form(False),
    translate: bool = Form(False),
    target_language: str = Form("en"),
    domain: str = Form("general")
):
    """Enhanced document processing with AI capabilities"""
    try:
        file_path = os.path.join("uploads", filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Extract text based on file type
        if filename.lower().endswith('.pdf'):
            extracted_text = pdf_service.extract_text(file_path)
        else:
            extracted_text = ocr_service.extract_text(file_path)

        result = {
            "message": "Document processed successfully",
            "filename": filename,
            "extracted_text": extracted_text,
            "processing_steps": []
        }

        # AI Enhancement Pipeline
        current_text = extracted_text

        # Step 1: Analyze document context
        if analyze_context:
            context = await llm_service.analyze_document_context(current_text)
            result["context_analysis"] = {
                "document_type": context.document_type,
                "language": context.language,
                "domain": context.domain,
                "confidence": context.confidence_score,
                "entities": context.entities,
                "sentiment": context.sentiment
            }
            result["processing_steps"].append("Context analysis completed")

        # Step 2: Enhance text with AI
        if enhance_text and current_text.strip():
            # Use context from previous step if available
            doc_context = context if analyze_context else None
            enhancement_result = await llm_service.enhance_extracted_text(current_text, doc_context)

            result["enhanced_text"] = enhancement_result.content
            result["enhancement_confidence"] = enhancement_result.confidence
            result["enhancement_metadata"] = enhancement_result.metadata
            current_text = enhancement_result.content
            result["processing_steps"].append("Text enhancement completed")

        # Step 3: Extract entities
        if extract_entities and current_text.strip():
            entities_result = await llm_service.extract_entities(current_text)
            try:
                import json
                entities = json.loads(entities_result.content)
                result["entities"] = entities
                result["processing_steps"].append("Entity extraction completed")
            except json.JSONDecodeError:
                result["entities"] = []
                result["processing_steps"].append("Entity extraction failed")

        # Step 4: Translation
        if translate and current_text.strip():
            translation_result = await translation_service.translate(
                text=current_text,
                target_lang=target_language,
                domain=domain
            )
            result["translation"] = {
                "translated_text": translation_result.translated_text,
                "source_language": translation_result.source_language,
                "target_language": translation_result.target_language,
                "confidence": translation_result.confidence_score,
                "quality": translation_result.quality_assessment.value
            }
            result["processing_steps"].append("Translation completed")

        # Add text statistics
        stats = nlp_service.analyze_text_statistics(current_text)
        result["text_statistics"] = {
            "word_count": stats.word_count,
            "sentence_count": stats.sentence_count,
            "readability_score": stats.readability_score,
            "language_confidence": stats.language_confidence
        }

        # Expose the requested target language at the top level so callers (frontend)
        # can reliably read which target language was used for translation.
        result["target_language"] = target_language

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.get("/files")
async def list_uploaded_files():
    """List all uploaded files"""
    try:
        upload_dir = "uploads"
        if not os.path.exists(upload_dir):
            return JSONResponse({"files": []})
        
        files = []
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            if os.path.isfile(file_path):
                files.append({
                    "filename": filename,
                    "size": os.path.getsize(file_path)
                })
        
        return JSONResponse({"files": files})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")

@router.post("/analyze")
async def analyze_document(filename: str = Form(...)):
    """Comprehensive document analysis with AI insights"""
    try:
        file_path = os.path.join("uploads", filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Extract text
        if filename.lower().endswith('.pdf'):
            extracted_text = pdf_service.extract_text(file_path)
        else:
            extracted_text = ocr_service.extract_text(file_path)

        if not extracted_text.strip():
            return JSONResponse({
                "message": "No text found in document",
                "filename": filename
            })

        # Comprehensive analysis
        analysis_results = {}

        # 1. Context Analysis
        context = await llm_service.analyze_document_context(extracted_text)
        analysis_results["context"] = {
            "document_type": context.document_type,
            "language": context.language,
            "domain": context.domain,
            "confidence": context.confidence_score
        }

        # 2. Text Statistics
        stats = nlp_service.analyze_text_statistics(extracted_text)
        analysis_results["statistics"] = {
            "word_count": stats.word_count,
            "sentence_count": stats.sentence_count,
            "paragraph_count": stats.paragraph_count,
            "readability_score": stats.readability_score,
            "average_word_length": stats.average_word_length
        }

        # 3. Quality Assessment
        quality = nlp_service.assess_text_quality(extracted_text)
        analysis_results["quality"] = {
            "overall_score": quality.overall_score,
            "issues_found": len(quality.spelling_errors) + len(quality.grammar_issues),
            "suggestions": quality.suggestions[:3]  # Top 3 suggestions
        }

        # 4. Document Structure
        structure = nlp_service.detect_document_structure(extracted_text)
        analysis_results["structure"] = structure

        # 5. Keywords
        keywords = nlp_service.extract_keywords(extracted_text, max_keywords=10)
        analysis_results["keywords"] = [{"word": word, "importance": score} for word, score in keywords]

        # 6. Sentiment Analysis
        sentiment_result = await llm_service.analyze_sentiment(extracted_text)
        try:
            import json
            sentiment = json.loads(sentiment_result.content)
            analysis_results["sentiment"] = sentiment
        except json.JSONDecodeError:
            analysis_results["sentiment"] = {"sentiment": "neutral", "confidence": 0.5}

        # 7. Summary
        summary_result = await llm_service.summarize_document(extracted_text, max_length=150)
        analysis_results["summary"] = summary_result.content

        return JSONResponse({
            "message": "Document analysis completed",
            "filename": filename,
            "extracted_text": extracted_text,
            "analysis": analysis_results
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/enhance")
async def enhance_document_text(filename: str = Form(...), enhancement_type: str = Form("general")):
    """Enhance document text using AI"""
    try:
        file_path = os.path.join("uploads", filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Extract text
        if filename.lower().endswith('.pdf'):
            extracted_text = pdf_service.extract_text(file_path)
        else:
            extracted_text = ocr_service.extract_text(file_path)

        if not extracted_text.strip():
            return JSONResponse({
                "message": "No text found in document",
                "filename": filename
            })

        # Analyze context first
        context = await llm_service.analyze_document_context(extracted_text)

        # Apply appropriate enhancement
        if enhancement_type == "ocr_correction":
            # First apply NLP corrections, then AI enhancement
            nlp_corrected = nlp_service.correct_ocr_errors(extracted_text)
            ai_result = await llm_service.correct_ocr_errors(nlp_corrected)
        elif enhancement_type == "grammar_fix":
            ai_result = await llm_service.enhance_extracted_text(extracted_text, context)
        else:
            ai_result = await llm_service.enhance_extracted_text(extracted_text, context)

        # Calculate improvement metrics
        original_quality = nlp_service.assess_text_quality(extracted_text)
        enhanced_quality = nlp_service.assess_text_quality(ai_result.content)

        improvement_score = enhanced_quality.overall_score - original_quality.overall_score

        return JSONResponse({
            "message": "Text enhancement completed",
            "filename": filename,
            "original_text": extracted_text,
            "enhanced_text": ai_result.content,
            "enhancement_type": enhancement_type,
            "confidence": ai_result.confidence,
            "improvement_score": improvement_score,
            "metadata": ai_result.metadata
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")