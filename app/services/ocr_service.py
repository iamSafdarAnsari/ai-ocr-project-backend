import os

# Try to import heavy OCR/image libraries; if unavailable, provide graceful fallbacks
try:
    import cv2
    import numpy as np
    import easyocr
    from PIL import Image
    _OCR_LIBS_AVAILABLE = True
except Exception:
    _OCR_LIBS_AVAILABLE = False


class OCRService:
    def __init__(self):
        # Initialize EasyOCR reader if available; otherwise keep a fallback
        if _OCR_LIBS_AVAILABLE:
            try:
                # Maintain a cache of readers for different language combinations
                self._readers = {}
                # Default languages: try Hindi first (hi) then English (en)
                # EasyOCR uses language codes like 'hi' for Hindi; if not available,
                # the reader initialization will fall back.
                default_langs = ('hi', 'en')
                try:
                    # create a reader for the default set
                    self._readers[default_langs] = easyocr.Reader(list(default_langs))
                except Exception:
                    # If creating reader with hi fails, fall back to English-only
                    try:
                        self._readers[('en',)] = easyocr.Reader(['en'])
                    except Exception:
                        self._readers = {}
            except Exception:
                self._readers = {}
        else:
            self._readers = {}
        
    def extract_text(self, image_path: str, languages: tuple = None) -> str:
        """Extract text from image using OCR"""
        try:
            if not _OCR_LIBS_AVAILABLE or not getattr(self, '_readers', None):
                # Lightweight fallback: return empty so callers can handle absence
                return ""

            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image")

            # Preprocess image
            processed_image = self._preprocess_image(image)
            # Determine which language reader(s) to try
            lang_sets = []
            if languages:
                # allow caller to pass a tuple like ('hi',) or ('hi','en')
                lang_sets.append(tuple(languages))
            # always try default hi+en then en-only
            lang_sets.append(('hi', 'en'))
            lang_sets.append(('en',))

            # Try each reader until we get non-empty text
            for lang_set in lang_sets:
                reader = self._readers.get(lang_set)
                if reader is None:
                    # try to create and cache it
                    try:
                        reader = easyocr.Reader(list(lang_set))
                        self._readers[lang_set] = reader
                    except Exception:
                        # can't create this reader, skip
                        continue

                # use paragraph=True to improve multi-line handwriting extraction where supported
                try:
                    results = reader.readtext(processed_image, detail=0, paragraph=True)
                except TypeError:
                    # older easyocr versions may not accept paragraph kwarg
                    results = reader.readtext(processed_image, detail=0)

                # results may be a list of strings
                text = ' '.join([r for r in results if isinstance(r, str)]) if results else ''
                if text and text.strip():
                    return text

            # If nothing found
            return "No text detected in the image."

        except Exception as e:
            return f"OCR extraction failed: {str(e)}"

    def _preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        # If OpenCV/numpy not available, return original image path to let callers handle fallback
        if not _OCR_LIBS_AVAILABLE:
            return image

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize image to improve OCR accuracy for small handwritten text
        height, width = gray.shape[:2]
        scale = 1.0
        if min(height, width) < 800:
            scale = 2.0
        elif min(height, width) < 1200:
            scale = 1.5

        if scale != 1.0:
            gray = cv2.resize(gray, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LINEAR)

        # Apply bilateral filter to preserve edges while reducing noise (better for handwriting)
        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # Adaptive thresholding often works better for uneven illumination
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # Morphological operations to connect broken strokes
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return cleaned