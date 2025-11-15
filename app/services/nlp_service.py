"""
Advanced NLP Service for Text Processing and Analysis
Provides comprehensive natural language processing capabilities
"""

import re
import string
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import Counter
import asyncio


@dataclass
class TextStatistics:
    """Text analysis statistics"""
    word_count: int
    sentence_count: int
    paragraph_count: int
    character_count: int
    average_word_length: float
    readability_score: float
    language_confidence: float


@dataclass
class TextQuality:
    """Text quality assessment"""
    overall_score: float
    spelling_errors: List[str]
    grammar_issues: List[str]
    formatting_issues: List[str]
    suggestions: List[str]


class AdvancedNLPService:
    """Advanced NLP service for comprehensive text analysis"""
    
    def __init__(self):
        self.stop_words = self._load_stop_words()
        self.common_ocr_errors = self._load_ocr_error_patterns()
    
    def _load_stop_words(self) -> set:
        """Load common stop words for multiple languages"""
        # Basic English stop words - in production, load from comprehensive lists
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
    
    def _load_ocr_error_patterns(self) -> Dict[str, str]:
        """Load common OCR error patterns and corrections"""
        return {
            r'\brn\b': 'm',  # rn -> m
            r'\bvv\b': 'w',  # vv -> w
            r'\bcl\b': 'd',  # cl -> d
            r'\bii\b': 'll', # ii -> ll
            r'\b0\b': 'O',   # 0 -> O (in words)
            r'\b1\b': 'l',   # 1 -> l (in words)
            r'\b5\b': 'S',   # 5 -> S (in words)
            r'\b8\b': 'B',   # 8 -> B (in words)
            r'\s+': ' ',     # Multiple spaces -> single space
            r'([.!?])\s*([a-z])': r'\1 \2',  # Fix spacing after punctuation
        }
    
    def analyze_text_statistics(self, text: str) -> TextStatistics:
        """Analyze comprehensive text statistics"""
        if not text.strip():
            return TextStatistics(0, 0, 0, 0, 0.0, 0.0, 0.0)
        
        # Basic counts
        words = self._extract_words(text)
        sentences = self._split_sentences(text)
        paragraphs = text.split('\n\n')
        
        word_count = len(words)
        sentence_count = len(sentences)
        paragraph_count = len([p for p in paragraphs if p.strip()])
        character_count = len(text)
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / max(1, word_count)
        
        # Simple readability score (Flesch-like)
        readability = self._calculate_readability(words, sentences)
        
        # Language confidence (basic heuristic)
        language_confidence = self._estimate_language_confidence(text)
        
        return TextStatistics(
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            character_count=character_count,
            average_word_length=avg_word_length,
            readability_score=readability,
            language_confidence=language_confidence
        )
    
    def assess_text_quality(self, text: str) -> TextQuality:
        """Assess text quality and identify issues"""
        spelling_errors = self._find_spelling_errors(text)
        grammar_issues = self._find_grammar_issues(text)
        formatting_issues = self._find_formatting_issues(text)
        suggestions = self._generate_suggestions(text, spelling_errors, grammar_issues, formatting_issues)
        
        # Calculate overall score
        error_count = len(spelling_errors) + len(grammar_issues) + len(formatting_issues)
        word_count = len(self._extract_words(text))
        error_ratio = error_count / max(1, word_count)
        overall_score = max(0.0, 1.0 - error_ratio)
        
        return TextQuality(
            overall_score=overall_score,
            spelling_errors=spelling_errors,
            grammar_issues=grammar_issues,
            formatting_issues=formatting_issues,
            suggestions=suggestions
        )
    
    def correct_ocr_errors(self, text: str) -> str:
        """Correct common OCR errors using pattern matching"""
        corrected_text = text
        
        for pattern, replacement in self.common_ocr_errors.items():
            corrected_text = re.sub(pattern, replacement, corrected_text)
        
        # Additional OCR corrections
        corrected_text = self._fix_character_spacing(corrected_text)
        corrected_text = self._fix_word_boundaries(corrected_text)
        
        return corrected_text
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[Tuple[str, float]]:
        """Extract keywords with importance scores"""
        words = self._extract_words(text.lower())
        
        # Filter out stop words and short words
        filtered_words = [
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        # Calculate word frequencies
        word_freq = Counter(filtered_words)
        
        # Simple TF scoring (in production, use TF-IDF)
        total_words = len(filtered_words)
        keywords = [
            (word, count / total_words) 
            for word, count in word_freq.most_common(max_keywords)
        ]
        
        return keywords
    
    def detect_document_structure(self, text: str) -> Dict[str, Any]:
        """Detect document structure and components"""
        structure = {
            "has_title": self._has_title(text),
            "has_headers": self._has_headers(text),
            "has_lists": self._has_lists(text),
            "has_tables": self._has_tables(text),
            "has_dates": self._has_dates(text),
            "has_addresses": self._has_addresses(text),
            "has_phone_numbers": self._has_phone_numbers(text),
            "has_emails": self._has_emails(text),
            "paragraph_count": len(text.split('\n\n')),
            "line_count": len(text.split('\n'))
        }
        
        return structure
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text"""
        # Remove punctuation and split
        translator = str.maketrans('', '', string.punctuation)
        clean_text = text.translate(translator)
        return [word for word in clean_text.split() if word.strip()]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_readability(self, words: List[str], sentences: List[str]) -> float:
        """Calculate readability score (simplified Flesch formula)"""
        if not words or not sentences:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables = sum(self._count_syllables(word) for word in words) / len(words)
        
        # Simplified Flesch Reading Ease
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        return max(0.0, min(100.0, score))
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _estimate_language_confidence(self, text: str) -> float:
        """Estimate confidence in language detection"""
        # Simple heuristic based on character patterns
        english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        total_chars = sum(1 for c in text if c.isalpha())
        
        if total_chars == 0:
            return 0.5
        
        return english_chars / total_chars
    
    def _find_spelling_errors(self, text: str) -> List[str]:
        """Find potential spelling errors (basic implementation)"""
        words = self._extract_words(text.lower())
        errors = []
        
        for word in words:
            # Very basic checks - in production, use proper spell checker
            if len(word) > 20:  # Unusually long words
                errors.append(f"Unusually long word: {word}")
            elif re.search(r'(.)\1{3,}', word):  # Repeated characters
                errors.append(f"Repeated characters: {word}")
        
        return errors
    
    def _find_grammar_issues(self, text: str) -> List[str]:
        """Find potential grammar issues"""
        issues = []
        
        # Check for double spaces
        if '  ' in text:
            issues.append("Multiple consecutive spaces found")
        
        # Check for missing spaces after punctuation
        if re.search(r'[.!?][a-zA-Z]', text):
            issues.append("Missing space after punctuation")
        
        return issues
    
    def _find_formatting_issues(self, text: str) -> List[str]:
        """Find formatting issues"""
        issues = []
        
        # Check for inconsistent line endings
        if '\r\n' in text and '\n' in text.replace('\r\n', ''):
            issues.append("Inconsistent line endings")
        
        # Check for excessive whitespace
        if re.search(r'\n{3,}', text):
            issues.append("Excessive blank lines")
        
        return issues
    
    def _generate_suggestions(self, text: str, spelling_errors: List[str], 
                            grammar_issues: List[str], formatting_issues: List[str]) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        if spelling_errors:
            suggestions.append("Review and correct spelling errors")
        
        if grammar_issues:
            suggestions.append("Fix grammar and punctuation issues")
        
        if formatting_issues:
            suggestions.append("Improve text formatting and structure")
        
        # Additional suggestions based on text analysis
        stats = self.analyze_text_statistics(text)
        
        if stats.readability_score < 30:
            suggestions.append("Consider simplifying sentence structure for better readability")
        
        if stats.average_word_length > 7:
            suggestions.append("Consider using shorter, more common words")
        
        return suggestions


# Utility functions for document structure detection
    def _has_title(self, text: str) -> bool:
        lines = text.split('\n')
        if lines and len(lines[0]) < 100 and lines[0].isupper():
            return True
        return False
    
    def _has_headers(self, text: str) -> bool:
        return bool(re.search(r'^[A-Z][A-Z\s]+$', text, re.MULTILINE))
    
    def _has_lists(self, text: str) -> bool:
        return bool(re.search(r'^\s*[-*•]\s+', text, re.MULTILINE))
    
    def _has_tables(self, text: str) -> bool:
        return bool(re.search(r'\|.*\|', text))
    
    def _has_dates(self, text: str) -> bool:
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
        return bool(re.search(date_pattern, text))
    
    def _has_addresses(self, text: str) -> bool:
        address_pattern = r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)'
        return bool(re.search(address_pattern, text, re.IGNORECASE))
    
    def _has_phone_numbers(self, text: str) -> bool:
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\(\d{3}\)\s*\d{3}[-.]?\d{4}'
        return bool(re.search(phone_pattern, text))
    
    def _has_emails(self, text: str) -> bool:
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return bool(re.search(email_pattern, text))
    
    def _fix_character_spacing(self, text: str) -> str:
        """Fix character spacing issues"""
        # Fix spaces around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s*([a-zA-Z])', r'\1 \2', text)
        return text
    
    def _fix_word_boundaries(self, text: str) -> str:
        """Fix word boundary issues"""
        # Fix missing spaces between words and punctuation
        text = re.sub(r'([a-zA-Z])([A-Z][a-z])', r'\1 \2', text)
        return text


# Global instance
nlp_service = AdvancedNLPService()
