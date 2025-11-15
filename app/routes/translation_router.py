from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from app.services.translation_service import translation_service
from app.services.voice_service import VoiceService

router = APIRouter()
voice_service = VoiceService()

class EnhancedTranslationRequest(BaseModel):
    text: str
    source_language: str = "auto"
    target_language: str = "en"
    domain: str = "general"
    context: Optional[str] = None
    include_alternatives: bool = False
    assess_quality: bool = True

class VoiceRequest(BaseModel):
    text: str
    language: str = "en"

@router.post("/translate")
async def translate_text(request: EnhancedTranslationRequest):
    """Enhanced translation with AI-powered quality assessment and context awareness"""
    try:
        # Perform enhanced translation
        result = await translation_service.translate(
            text=request.text,
            source_lang=request.source_language,
            target_lang=request.target_language,
            domain=request.domain,
            context=request.context
        )

        response_data = {
            "original_text": request.text,
            "translated_text": result.translated_text,
            "source_language": result.source_language,
            "target_language": result.target_language,
            "domain": result.domain,
            "confidence_score": result.confidence_score,
            "quality_assessment": result.quality_assessment.value
        }

        # Add issues if any
        if result.issues:
            response_data["issues"] = result.issues

        # Get alternatives if requested
        if request.include_alternatives:
            alternatives = await translation_service.get_translation_alternatives(
                text=request.text,
                source_lang=result.source_language,
                target_lang=result.target_language,
                count=3
            )
            response_data["alternatives"] = alternatives

        return JSONResponse(response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

# Keep the old endpoint for backward compatibility
@router.post("/translate-simple")
async def translate_text_simple(
    text: str = Form(...),
    source_language: str = Form("auto"),
    target_language: str = Form("en")
):
    """Simple translation endpoint for backward compatibility"""
    try:
        result = await translation_service.translate(
            text=text,
            source_lang=source_language,
            target_lang=target_language
        )

        return JSONResponse({
            "original_text": text,
            "translated_text": result.translated_text,
            "source_language": result.source_language,
            "target_language": result.target_language
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@router.post("/generate-voice")
async def generate_voice(request: VoiceRequest):
    """Generate voice from text"""
    try:
        audio_file = voice_service.text_to_speech(request.text, request.language)
        
        return JSONResponse({
            "message": "Voice generated successfully",
            "audio_file": audio_file,
            "text": request.text,
            "language": request.language
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice generation failed: {str(e)}")

@router.get("/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    try:
        languages = translation_service.get_supported_languages()
        return JSONResponse({"languages": languages})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get languages: {str(e)}")