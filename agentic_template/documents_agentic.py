"""
Simple Document Content Gathering Agent using Google ADK with Gemini Flash 2.0
"""

import os
from google import genai
from google.genai import types


class DocumentAgent:
    """Agent for gathering and processing document content using Gemini Flash 2.0"""
    
    def __init__(self, api_key:  str = None):
        """
        Initialize the document agent
        
        Args:
            api_key:  Google AI API key.  If None, will use GOOGLE_API_KEY env variable
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided or set in GOOGLE_API_KEY environment variable")
        
        # Initialize the client
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.0-flash-exp"
        
    def gather_document_content(self, document_path: str, query: str = None) -> str:
        """
        Gather content from a document file
        
        Args:
            document_path: Path to the document file
            query: Optional query to ask about the document
            
        Returns: 
            str: Extracted or analyzed content
        """
        # Upload the file
        with open(document_path, 'rb') as f:
            uploaded_file = self.client.files. upload(file=f)
        
        # Prepare the prompt
        if query:
            prompt = f"Analyze this document and answer:  {query}"
        else:
            prompt = "Extract and summarize the main content from this document."
        
        # Generate response
        response = self.client. models.generate_content(
            model=self.model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part. from_uri(
                            file_uri=uploaded_file.uri,
                            mime_type=uploaded_file.mime_type
                        ),
                        types.Part.from_text(prompt)
                    ]
                )
            ]
        )
        
        return response.text
    
    def gather_from_text(self, text: str, task: str = "summarize") -> str:
        """
        Process text content directly
        
        Args:
            text: Text content to process
            task: Task to perform (summarize, extract, analyze, etc.)
            
        Returns:
            str: Processed content
        """
        prompt = f"{task. capitalize()} the following content:\n\n{text}"
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        
        return response.text
    
    def gather_from_url(self, url: str, query: str = None) -> str:
        """
        Gather content from a URL (if the document is publicly accessible)
        
        Args: 
            url: URL to the document
            query: Optional query about the document
            
        Returns: 
            str:  Extracted or analyzed content
        """
        if query:
            prompt = f"Access this URL:  {url}\n\nThen answer: {query}"
        else:
            prompt = f"Access this URL and extract the main content:  {url}"
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        
        return response.text


# Example usage
if __name__ == "__main__":
    # Initialize agent
    agent = DocumentAgent()
    
    # Example 1: Gather from a local document
    # result = agent.gather_document_content(
    #     document_path="path/to/your/document.pdf",
    #     query="What are the main topics discussed?"
    # )
    # print("Document Analysis:", result)
    
    # Example 2: Process text directly
    sample_text = """
    This is a sample document about artificial intelligence. 
    AI is transforming various industries including healthcare,
    finance, and transportation. 
    """
    result = agent.gather_from_text(sample_text, task="summarize")
    print("Text Summary:", result)
    
    # Example 3: Gather from URL
    # result = agent.gather_from_url(
    #     url="https://example.com/document",
    #     query="What is this document about?"
    # )
    # print("URL Content:", result)
