import os

# Guard heavy PDF libraries
try:
    import fitz  # PyMuPDF
    import PyPDF2
    _PDF_LIBS_AVAILABLE = True
except Exception:
    fitz = None
    PyPDF2 = None
    _PDF_LIBS_AVAILABLE = False
class PDFService:
    def __init__(self):
        pass
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF document"""
        try:
            # Try PyMuPDF first (better for complex PDFs)
            if _PDF_LIBS_AVAILABLE:
                text = self._extract_with_pymupdf(pdf_path)

                # If PyMuPDF fails or returns empty, try PyPDF2
                if not text.strip():
                    text = self._extract_with_pypdf2(pdf_path)
                return text
            else:
                # No PDF libs available — return empty string so caller can handle it
                return ""
        except Exception as e:
            return f"PDF extraction failed: {str(e)}"
    
    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            
            doc.close()
            return text
        except Exception as e:
            print(f"PyMuPDF extraction failed: {e}")
            return ""
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
                return text
        except Exception as e:
            print(f"PyPDF2 extraction failed: {e}")
            return ""
    
    def get_pdf_info(self, pdf_path: str) -> dict:
        """Get PDF metadata and information"""
        try:
            doc = fitz.open(pdf_path)
            info = {
                "page_count": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", "")
            }
            doc.close()
            return info
        except Exception as e:
            return {"error": f"Failed to get PDF info: {str(e)}"}