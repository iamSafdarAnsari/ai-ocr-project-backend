# """
# Enhanced Translation Service with AI-Powered Context-Aware Translation
# Provides intelligent translation with quality assessment and domain-specific optimization
# """

# import os
# import asyncio
# from typing import List, Dict, Optional, Tuple
# from dataclasses import dataclass
# from enum import Enum
# import json
# import re

# # Guard OpenAI import
# try:
#     from openai import OpenAI
#     _OPENAI_AVAILABLE = True
# except Exception:
#     OpenAI = None
#     _OPENAI_AVAILABLE = False

# class TranslationQuality(Enum):
#     """Translation quality levels"""
#     EXCELLENT = "excellent"
#     GOOD = "good"
#     FAIR = "fair"
#     POOR = "poor"

# @dataclass
# class TranslationResult:
#     """Enhanced translation result with metadata"""
#     translated_text: str
#     source_language: str
#     target_language: str
#     confidence_score: float
#     quality_assessment: TranslationQuality
#     domain: Optional[str] = None
#     alternatives: Optional[List[str]] = None
#     issues: Optional[List[str]] = None

# @dataclass
# class LanguageInfo:
#     """Language information with metadata"""
#     code: str
#     name: str
#     native_name: str
#     direction: str  # ltr or rtl
#     family: str

# class EnhancedTranslationService:
#     def __init__(self):
#         # Initialize OpenAI client
#         api_key = os.getenv("OPENAI_API_KEY")
#         self.client = None
#         # self.client=OpenAI(api_key=api_key)
#         if _OPENAI_AVAILABLE and api_key:
#             try:
#                 self.client = OpenAI(api_key=api_key)
#             except Exception as e:
#                 print(f"Failed to initialize OpenAI client for translation: {e}")
#                 self.client = None
#         else:
#             if not _OPENAI_AVAILABLE:
#                 print("OpenAI SDK not available. Translation will use fallback method.")
#             else:
#                 print("Warning: OPENAI_API_KEY not found. Translation will use fallback method.")

#         # Enhanced language support with metadata
#         self.supported_languages = {
#             "en": LanguageInfo("en", "English", "English", "ltr", "Germanic"),
#             "es": LanguageInfo("es", "Spanish", "Español", "ltr", "Romance"),
#             "fr": LanguageInfo("fr", "French", "Français", "ltr", "Romance"),
#             "de": LanguageInfo("de", "German", "Deutsch", "ltr", "Germanic"),
#             "it": LanguageInfo("it", "Italian", "Italiano", "ltr", "Romance"),
#             "pt": LanguageInfo("pt", "Portuguese", "Português", "ltr", "Romance"),
#             "ru": LanguageInfo("ru", "Russian", "Русский", "ltr", "Slavic"),
#             "ja": LanguageInfo("ja", "Japanese", "日本語", "ltr", "Japonic"),
#             "ko": LanguageInfo("ko", "Korean", "한국어", "ltr", "Koreanic"),
#             "zh": LanguageInfo("zh", "Chinese", "中文", "ltr", "Sino-Tibetan"),
#             "ar": LanguageInfo("ar", "Arabic", "العربية", "rtl", "Semitic"),
#             "hi": LanguageInfo("hi", "Hindi", "हिन्दी", "ltr", "Indo-European"),
#             "nl": LanguageInfo("nl", "Dutch", "Nederlands", "ltr", "Germanic"),
#             "sv": LanguageInfo("sv", "Swedish", "Svenska", "ltr", "Germanic"),
#             "no": LanguageInfo("no", "Norwegian", "Norsk", "ltr", "Germanic"),
#             "da": LanguageInfo("da", "Danish", "Dansk", "ltr", "Germanic"),
#             "fi": LanguageInfo("fi", "Finnish", "Suomi", "ltr", "Finno-Ugric"),
#             "pl": LanguageInfo("pl", "Polish", "Polski", "ltr", "Slavic"),
#             "tr": LanguageInfo("tr", "Turkish", "Türkçe", "ltr", "Turkic"),
#             "he": LanguageInfo("he", "Hebrew", "עברית", "rtl", "Semitic")
#         }

#         # Domain-specific translation contexts
#         self.domain_contexts = {
#             "legal": "Legal document translation requiring precise terminology",
#             "medical": "Medical document translation with accurate medical terms",
#             "business": "Business document translation with professional tone",
#             "technical": "Technical document translation with specialized terminology",
#             "academic": "Academic document translation with scholarly language",
#             "general": "General purpose translation maintaining natural flow"
#         }
    
#     async def translate(self, text: str, source_lang: str = "auto", target_lang: str = "en",
#                        domain: str = "general", context: Optional[str] = None) -> TranslationResult:
#         """Enhanced translation with context awareness and quality assessment"""
#         try:
#             if not text.strip():
#                 return TranslationResult(
#                     translated_text="",
#                     source_language=source_lang,
#                     target_language=target_lang,
#                     confidence_score=1.0,
#                     quality_assessment=TranslationQuality.EXCELLENT
#                 )

#             # Detect source language if auto
#             if source_lang == "auto":
#                 source_lang = await self.detect_language(text)

#             # Perform translation with AI
#             if self.client:
#                 return await self._translate_with_ai(text, source_lang, target_lang, domain, context)
#             else:
#                 # Fallback translation
#                 return self._translate_fallback(text, source_lang, target_lang)

#         except Exception as e:
#             return TranslationResult(
#                 translated_text=f"Translation failed: {str(e)}",
#                 source_language=source_lang,
#                 target_language=target_lang,
#                 confidence_score=0.0,
#                 quality_assessment=TranslationQuality.POOR,
#                 issues=[str(e)]
#             )
    
#     async def _translate_with_ai(self, text: str, source_lang: str, target_lang: str,
#                                 domain: str, context: Optional[str]) -> TranslationResult:
#         """Advanced AI translation with context and quality assessment"""
#         try:
#             source_info = self.supported_languages.get(source_lang)
#             target_info = self.supported_languages.get(target_lang)

#             if not source_info or not target_info:
#                 raise ValueError(f"Unsupported language pair: {source_lang} -> {target_lang}")

#             # Build context-aware prompt
#             domain_context = self.domain_contexts.get(domain, self.domain_contexts["general"])

#             prompt = f"""
#             You are an expert translator specializing in {domain_context}.

#             Task: Translate from {source_info.name} ({source_info.native_name}) to {target_info.name} ({target_info.native_name})
#             Domain: {domain}
#             {f"Additional context: {context}" if context else ""}

#             Requirements:
#             1. Maintain the original meaning and tone
#             2. Use appropriate terminology for the {domain} domain
#             3. Ensure natural flow in the target language
#             4. Preserve formatting and structure
#             5. Handle cultural nuances appropriately

#             Text to translate:
#             {text}

#             Provide only the translation without explanations.
#             """

#             response = self.client.chat.completions.create(
#                 model="gpt-4-turbo-preview",
#                 messages=[
#                     {"role": "system", "content": "You are a professional translator with expertise in multiple domains and cultural contexts."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 max_tokens=2000,
#                 temperature=0.2
#             )

#             translated_text = response.choices[0].message.content.strip()

#             # Assess translation quality
#             quality_result = await self._assess_translation_quality(text, translated_text, source_lang, target_lang, domain)

#             return TranslationResult(
#                 translated_text=translated_text,
#                 source_language=source_lang,
#                 target_language=target_lang,
#                 confidence_score=quality_result["confidence"],
#                 quality_assessment=quality_result["quality"],
#                 domain=domain,
#                 alternatives=quality_result.get("alternatives", []),
#                 issues=quality_result.get("issues", [])
#             )

#         except Exception as e:
#             print(f"AI translation failed: {e}")
#             return self._translate_fallback(text, source_lang, target_lang)
    
#     async def _assess_translation_quality(self, original: str, translated: str,
#                                           source_lang: str, target_lang: str, domain: str) -> Dict:
#         """Assess translation quality using AI"""
#         try:
#             prompt = f"""
#             Assess the quality of this translation and provide a JSON response:

#             Original ({source_lang}): {original}
#             Translation ({target_lang}): {translated}
#             Domain: {domain}

#             Evaluate:
#             1. Accuracy of meaning preservation
#             2. Fluency and naturalness
#             3. Domain-appropriate terminology
#             4. Cultural appropriateness
#             5. Grammar and syntax

#             Provide JSON with:
#             {{
#                 "confidence": 0.0-1.0,
#                 "quality": "excellent|good|fair|poor",
#                 "issues": ["list of issues found"],
#                 "alternatives": ["alternative translations if needed"],
#                 "scores": {{
#                     "accuracy": 0.0-1.0,
#                     "fluency": 0.0-1.0,
#                     "terminology": 0.0-1.0,
#                     "cultural": 0.0-1.0,
#                     "grammar": 0.0-1.0
#                 }}
#             }}
#             """

#             response = self.client.chat.completions.create(
#                 model="gpt-4-turbo-preview",
#                 messages=[
#                     {"role": "system", "content": "You are a translation quality assessment expert."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 max_tokens=1000,
#                 temperature=0.1
#             )

#             return json.loads(response.choices[0].message.content)

#         except Exception as e:
#             print(f"Quality assessment failed: {e}")
#             return {
#                 "confidence": 0.7,
#                 "quality": "good",
#                 "issues": [],
#                 "alternatives": []
#             }

#     def _translate_fallback(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
#         """Simple fallback translation"""
#         return TranslationResult(
#             translated_text=f"[Translated to {target_lang}]: {text}",
#             source_language=source_lang,
#             target_language=target_lang,
#             confidence_score=0.5,
#             quality_assessment=TranslationQuality.FAIR,
#             issues=["Using fallback translation method"]
#         )
    
#     def get_supported_languages(self) -> List[Dict[str, str]]:
#         """Get list of supported languages with enhanced metadata"""
#         return [
#             {
#                 "code": code,
#                 "name": lang_info.name,
#                 "native_name": lang_info.native_name,
#                 "direction": lang_info.direction,
#                 "family": lang_info.family
#             }
#             for code, lang_info in self.supported_languages.items()
#         ]

#     async def detect_language(self, text: str) -> str:
#         """Enhanced language detection with confidence scoring"""
#         try:
#             if self.client and len(text.strip()) > 10:
#                 prompt = f"""
#                 Detect the language of the following text and respond with only the ISO 639-1 language code.
#                 If uncertain, respond with the most likely code.

#                 Text: {text[:500]}

#                 Respond with only the 2-letter language code (e.g., 'en', 'es', 'fr').
#                 """

#                 response = self.client.chat.completions.create(
#                     model="gpt-3.5-turbo",
#                     messages=[
#                         {"role": "system", "content": "You are a language detection expert. Respond with only the ISO language code."},
#                         {"role": "user", "content": prompt}
#                     ],
#                     max_tokens=10,
#                     temperature=0
#                 )

#                 detected_lang = response.choices[0].message.content.strip().lower()

#                 # Validate detected language
#                 if detected_lang in self.supported_languages:
#                     return detected_lang
#                 else:
#                     return "en"  # Default to English
#             else:
#                 return self._detect_language_heuristic(text)

#         except Exception as e:
#             print(f"Language detection failed: {e}")
#             return self._detect_language_heuristic(text)

#     def _detect_language_heuristic(self, text: str) -> str:
#         """Simple heuristic language detection"""
#         text_lower = text.lower()

#         # Simple character-based detection
#         if re.search(r'[а-я]', text_lower):
#             return "ru"
#         elif re.search(r'[α-ω]', text_lower):
#             return "el"
#         elif re.search(r'[أ-ي]', text_lower):
#             return "ar"
#         elif re.search(r'[一-龯]', text_lower):
#             return "zh"
#         elif re.search(r'[ひらがなカタカナ]', text_lower):
#             return "ja"
#         elif re.search(r'[가-힣]', text_lower):
#             return "ko"
#         elif re.search(r'[ñáéíóúü]', text_lower):
#             return "es"
#         elif re.search(r'[àâäéèêëïîôöùûüÿç]', text_lower):
#             return "fr"
#         elif re.search(r'[äöüß]', text_lower):
#             return "de"
#         else:
#             return "en"  # Default to English

#     async def translate_batch(self, texts: List[str], source_lang: str = "auto",
#                              target_lang: str = "en", domain: str = "general") -> List[TranslationResult]:
#         """Translate multiple texts in batch"""
#         tasks = [
#             self.translate(text, source_lang, target_lang, domain)
#             for text in texts
#         ]
#         return await asyncio.gather(*tasks)

#     async def get_translation_alternatives(self, text: str, source_lang: str,
#                                           target_lang: str, count: int = 3) -> List[str]:
#         """Get multiple translation alternatives"""
#         try:
#             if not self.client:
#                 return [text]

#             source_info = self.supported_languages.get(source_lang)
#             target_info = self.supported_languages.get(target_lang)

#             prompt = f"""
#             Provide {count} different translation alternatives for the following text.
#             From {source_info.name} to {target_info.name}.

#             Each alternative should have a slightly different style or approach while maintaining accuracy.

#             Text: {text}

#             Provide only the translations, one per line, numbered 1-{count}.
#             """

#             response = self.client.chat.completions.create(
#                 model="gpt-4-turbo-preview",
#                 messages=[
#                     {"role": "system", "content": "You are a professional translator providing alternative translations."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 max_tokens=1000,
#                 temperature=0.7
#             )

#             alternatives = []
#             for line in response.choices[0].message.content.strip().split('\n'):
#                 if line.strip() and not line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
#                     alternatives.append(line.strip())
#                 elif '. ' in line:
#                     alternatives.append(line.split('. ', 1)[1].strip())

#             return alternatives[:count]

#         except Exception as e:
#             print(f"Alternative generation failed: {e}")
#             return [text]

#     def get_domain_contexts(self) -> List[str]:
#         """Get available domain contexts"""
#         return list(self.domain_contexts.keys())


# # Global instance
# translation_service = EnhancedTranslationService()











"""
Enhanced Translation Service with AI-Powered Context-Aware Translation
Supports multi-language translation with domain optimization
"""

import os
import json
import re
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Guard OpenAI import so this module can be used without the SDK
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None
    _OPENAI_AVAILABLE = False

# Load environment variables
load_dotenv()

# Initialize OpenAI client only if available and API key present
api_key = os.getenv("OPENAI_API_KEY")
if _OPENAI_AVAILABLE and api_key:
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Warning: failed to initialize OpenAI client: {e}")
        client = None
else:
    if not _OPENAI_AVAILABLE:
        print("OpenAI SDK not installed; translation will use fallback mode.")
    else:
        print("OPENAI_API_KEY not found; translation will use fallback mode.")
    client = None

# ---------------- ENUMS & DATACLASSES ----------------

class TranslationQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class TranslationResult:
    translated_text: str
    source_language: str
    target_language: str
    confidence_score: float
    quality_assessment: TranslationQuality
    domain: Optional[str] = None
    alternatives: Optional[List[str]] = None
    issues: Optional[List[str]] = None

@dataclass
class LanguageInfo:
    code: str
    name: str
    native_name: str
    direction: str
    family: str

# ---------------- MAIN CLASS ----------------

class EnhancedTranslationService:
    def __init__(self):
        self.client = client
         # Supported languages
        self.supported_languages = {
            # 🌍 Global Languages  
            "en": LanguageInfo("en", "English", "English", "ltr", "Germanic"),
            "es": LanguageInfo("es", "Spanish", "Español", "ltr", "Romance"),
            "fr": LanguageInfo("fr", "French", "Français", "ltr", "Romance"),
            "de": LanguageInfo("de", "German", "Deutsch", "ltr", "Germanic"),
            "it": LanguageInfo("it", "Italian", "Italiano", "ltr", "Romance"),
            "pt": LanguageInfo("pt", "Portuguese", "Português", "ltr", "Romance"),
            "ru": LanguageInfo("ru", "Russian", "Русский", "ltr", "Slavic"),
            "ar": LanguageInfo("ar", "Arabic", "العربية", "rtl", "Semitic"),
            "nl": LanguageInfo("nl", "Dutch", "Nederlands", "ltr", "Germanic"),
            "sv": LanguageInfo("sv", "Swedish", "Svenska", "ltr", "Germanic"),
            "no": LanguageInfo("no", "Norwegian", "Norsk", "ltr", "Germanic"),
            "da": LanguageInfo("da", "Danish", "Dansk", "ltr", "Germanic"),
            "fi": LanguageInfo("fi", "Finnish", "Suomi", "ltr", "Finno-Ugric"),
            "pl": LanguageInfo("pl", "Polish", "Polski", "ltr", "Slavic"),
            "tr": LanguageInfo("tr", "Turkish", "Türkçe", "ltr", "Turkic"),
            "he": LanguageInfo("he", "Hebrew", "עברית", "rtl", "Semitic"),
            "zh": LanguageInfo("zh", "Chinese", "中文", "ltr", "Sino-Tibetan"),
            "ja": LanguageInfo("ja", "Japanese", "日本語", "ltr", "Japonic"),
            "ko": LanguageInfo("ko", "Korean", "한국어", "ltr", "Koreanic"),

            # 🇮🇳 Indian Languages
            "hi": LanguageInfo("hi", "Hindi", "हिन्दी", "ltr", "Indo-European"),
            "ta": LanguageInfo("ta", "Tamil", "தமிழ்", "ltr", "Dravidian"),
            "te": LanguageInfo("te", "Telugu", "తెలుగు", "ltr", "Dravidian"),
            "ur": LanguageInfo("ur", "Urdu", "اردو", "rtl", "Indo-European"),
            "ml": LanguageInfo("ml", "Malayalam", "മലയാളം", "ltr", "Dravidian"),
            "bn": LanguageInfo("bn", "Bengali", "বাংলা", "ltr", "Indo-Aryan"),
            "pa": LanguageInfo("pa", "Punjabi", "ਪੰਜਾਬੀ", "ltr", "Indo-Aryan"),
            "gu": LanguageInfo("gu", "Gujarati", "ગુજરાતી", "ltr", "Indo-Aryan"),
        }

        # 🌐 Domain Contexts
        self.domain_contexts = {
            "general": "General conversation translation maintaining natural tone",
            "technical": "Technical document translation with precise terminology",
            "medical": "Medical text translation with accurate terms",
            "business": "Business or professional correspondence translation",
            "legal": "Legal document translation with formal accuracy",
            "academic": "Academic paper translation with formal tone",
        }

    # ---------------- MAIN TRANSLATION ----------------

    async def translate(self, text: str, source_lang: str = "auto",
                        target_lang: str = "en", domain: str = "general",
                        context: Optional[str] = None) -> TranslationResult:
        """Perform enhanced translation with AI"""

        if not text.strip():
            return TranslationResult("", source_lang, target_lang, 1.0, TranslationQuality.EXCELLENT)

        # Detect language if auto
        if source_lang == "auto":
            source_lang = await self.detect_language(text)

        try:
            translated_text = await self._translate_with_ai(
                text, source_lang, target_lang, domain, context
            )

            quality = await self._assess_translation_quality(
                text, translated_text, source_lang, target_lang
            )

            return TranslationResult(
                translated_text=translated_text,
                source_language=source_lang,
                target_language=target_lang,
                confidence_score=quality["confidence"],
                quality_assessment=TranslationQuality(quality["quality"]),
                domain=domain,
                issues=quality.get("issues", [])
            )

        except Exception as e:
            # Log and include the original error message in the fallback so the
            # caller (frontend) can show a meaningful message to the user.
            err_msg = str(e)
            print(f"⚠️ Translation failed, fallback used: {err_msg}")
            return self._translate_fallback(text, source_lang, target_lang, error_msg=err_msg)

    # ---------------- OPENAI TRANSLATION ----------------

    async def _translate_with_ai(self, text: str, source_lang: str, target_lang: str,
                                 domain: str, context: Optional[str]) -> str:
        source_info = self.supported_languages.get(source_lang)
        target_info = self.supported_languages.get(target_lang)

        if not source_info or not target_info:
            raise ValueError(f"Unsupported language pair: {source_lang} -> {target_lang}")

        domain_context = self.domain_contexts.get(domain, self.domain_contexts["general"])

        prompt = f"""
        You are an expert translator.
        Translate the following text from {source_info.name} ({source_info.native_name})
        to {target_info.name} ({target_info.native_name}).

        Domain: {domain_context}
        Additional context: {context or "N/A"}

        Rules:
        - Maintain meaning, tone, and style
        - Use natural language for {target_info.name}
        - Output only the translated text

        Text:
        {text}
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional multilingual translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )

        translated_text = response.choices[0].message.content.strip()
        return translated_text

    # ---------------- QUALITY ASSESSMENT ----------------

    async def _assess_translation_quality(self, original: str, translated: str,
                                          source_lang: str, target_lang: str) -> Dict:
        try:
            prompt = f"""
            Evaluate the translation quality between:
            Original ({source_lang}): {original}
            Translation ({target_lang}): {translated}

            Give a JSON response with:
            {{
                "confidence": number between 0.0-1.0,
                "quality": "excellent" | "good" | "fair" | "poor",
                "issues": ["optional list of issues"]
            }}
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a translation evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            return json.loads(response.choices[0].message.content)

        except Exception:
            return {"confidence": 0.8, "quality": "good", "issues": []}

    # ---------------- FALLBACK TRANSLATION ----------------

    def _translate_fallback(self, text: str, source_lang: str, target_lang: str, error_msg: Optional[str] = None) -> TranslationResult:
        """Fallback if API fails. Includes optional error message in issues."""
        issues = ["Fallback translation used"]
        if error_msg:
            issues.append(f"Error: {error_msg}")

        # Keep translated_text readable and indicate the target language
        translated_text = f"[{source_lang}->{target_lang}] {text}"

        return TranslationResult(
            translated_text=translated_text,
            source_language=source_lang,
            target_language=target_lang,
            confidence_score=0.5,
            quality_assessment=TranslationQuality.FAIR,
            issues=issues
        )

    # ---------------- LANGUAGE DETECTION ----------------

    async def detect_language(self, text: str) -> str:
        try:
            prompt = f"Detect the language of this text and return only its ISO code (like en, hi, fr):\n{text[:300]}"
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a language detection expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5,
                temperature=0
            )
            code = response.choices[0].message.content.strip().lower()
            return code if code in self.supported_languages else "en"
        except Exception:
            return "en"

    # ---------------- EXTRA UTILITIES ----------------

    def get_supported_languages(self) -> List[Dict[str, str]]:
        return [
            {
                "code": code,
                "name": info.name,
                "native_name": info.native_name,
                "direction": info.direction,
                "family": info.family
            }
            for code, info in self.supported_languages.items()
        ]

# Global instance
translation_service = EnhancedTranslationService()
