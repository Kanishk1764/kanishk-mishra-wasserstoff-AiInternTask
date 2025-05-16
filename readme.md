# DocTheme: Document Analysis & Theme Identification System

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

DocTheme is an intelligent document analysis system that enables users to upload, process, and query multiple documents while automatically identifying common themes. The application uses natural language processing and machine learning to extract insights from PDFs and images, allowing users to interact with their document collection through simple queries.

## 🌟 Features

- **Multi-format document processing**: Upload and extract text from PDFs and images
- **Automated theme identification**: Discover common themes across your document collection
- **Intelligent document querying**: Ask questions about your documents in natural language
- **Theme-based synthesis**: Get responses that integrate information across multiple documents
- **Interactive web interface**: User-friendly Streamlit application

## 📋 Table of Contents

- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [API Keys](#api-keys)
- [Contributing](#contributing)
- [License](#license)

## 🎮 Demo

![DocTheme Demo](https://via.placeholder.com/800x450.png?text=DocTheme+Demo)

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR ([Installation guide](https://github.com/tesseract-ocr/tesseract))
- Groq API key

### Setup

1. Clone the repository:

```bash
git clone https://github.com/Kanishk1764/kanishk-mishra-wasserstoff-AiInternTask.git
cd doctheme
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Requirements

```
streamlit>=1.22.0
PyPDF2>=3.0.0
pillow>=9.0.0
pytesseract>=0.3.10
groq>=0.4.0
pandas>=1.5.0
scikit-learn>=1.2.0
numpy>=1.22.0
```

## 🚀 Usage

1. Set your Groq API key as an environment variable:

```bash
export GROQ_API_KEY="your_groq_api_key"  # On Windows: set GROQ_API_KEY=your_groq_api_key
```

2. Run the application:

```bash
streamlit run app.py
```

3. Navigate to the URL shown in your terminal (typically http://localhost:8501)

4. Using the application:
   - Upload documents (PDF or images) using the sidebar
   - Click "Analyze Documents for Themes" to identify common themes
   - Enter questions in the query box to search across your documents
   - View document-specific answers and thematic synthesis of results

## ⚙️ How It Works

### Document Processing Pipeline

1. **Upload**: Users upload PDF or image files through the Streamlit interface
2. **Text Extraction**: 
   - PDFs: Text is extracted using PyPDF2
   - Images: Text is extracted using Tesseract OCR
3. **Storage**: Document text and metadata are stored in memory for analysis

### Theme Identification

The system uses unsupervised machine learning to identify common themes:

1. **Text Vectorization**: Documents are converted to TF-IDF vectors
2. **Clustering**: K-Means algorithm groups similar documents
3. **Theme Naming**: Each cluster is named based on its most prominent terms

### Document Querying

The system leverages Large Language Models for intelligent document querying:

1. **Document-Level Queries**: Each document is queried independently
2. **Theme-Level Synthesis**: Information is synthesized across identified themes
3. **Response Generation**: Structured responses with citations are provided

## 🔑 API Keys

To use this application, you'll need a Groq API key:

1. Sign up at [Groq](https://console.groq.com/signup)
2. Generate an API key in your account dashboard
3. Set the API key as an environment variable or update the `client = Groq(api_key="your_api_key")` line in the code

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/) for the interactive web framework
- [Groq](https://groq.com/) for LLM API access
- [scikit-learn](https://scikit-learn.org/) for machine learning capabilities
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for image text extraction

---

Made with ❤️ by [Your Name or Organization]
