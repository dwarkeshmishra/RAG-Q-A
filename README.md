# 💬 Loan Approval RAG Chatbot

A professional, Retrieval-Augmented Generation (RAG) chatbot for intelligent loan approval guidance, combining document retrieval and generative AI. Powered by Groq's Llama-3 and Hugging Face's MiniLM, this app delivers accurate, context-rich answers for all your loan queries—no custom model training needed!

#Live Project Link

**https://rag-loan.streamlit.app/**

## 🚀 Overview

This project is a modern chatbot that assists users with loan approval questions by leveraging:

- **Document retrieval** (RAG) from your own knowledge base
- **Generative AI** for smart, conversational answers
- **Zero dataset training**—just connect your API key and go!

## ✨ Features

- 🔍 **Retrieval-Augmented Generation:** Fetches loan info from your uploaded docs for context-aware answers
- 🤖 **LLM-Powered:** Uses Groq’s Llama-3 for natural, detailed responses
- ⚡ **Fast & Lightweight:** Hugging Face’s MiniLM for speedy document search
- 🖥️ **Streamlit UI:** Clean, interactive chat interface with history
- 🧠 **Fallback Mode:** If no docs, answers from LLM’s built-in knowledge

## 🏗️ Architecture

| Component        | Technology                        | Purpose                              |
|------------------|-----------------------------------|--------------------------------------|
| Retrieval        | FAISS + MiniLM (Hugging Face)     | Fast, semantic search over knowledge |
| Generation       | Groq Llama-3 (LLM)                | Contextual, conversational answers   |
| Orchestration    | LangChain                         | RAG pipeline management              |
| UI               | Streamlit                         | User-friendly chat interface         |

## 📦 Installation

### Prerequisites

- Python 3.8+
- Groq API key (get yours from Groq)

### Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/dwarkeshmishra/RAG-Q-A.git
   cd RAG-Q-A
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install streamlit langchain langchain-groq sentence-transformers faiss-cpu python-dotenv
   ```

4. **Set Up Your API Key:**
   - Create a `.env` file in your project root:
     ```
     GROQ_API_KEY=your_actual_groq_api_key
     ```

5. **Prepare Knowledge Base (Optional):**
   - Add `.txt` files with loan guidelines or FAQs to the `knowledge_base/` folder.
   - If empty, the chatbot uses the LLM’s general financial knowledge.

6. **Run the App:**
   ```bash
   streamlit run app.py
   ```
   - The app will open in your browser at `http://localhost:8501`.

## 💡 Example Questions

- What factors affect loan approval?
- Can a person with a $50,000 income and good credit get a $200,000 loan?
- What documents are needed for a home loan?
- How can I improve my chances of getting a loan?
- What’s the impact of credit history on loan eligibility?

## 🗂️ Project Structure

```
loan-approval-rag-chatbot/
├── app.py
├── .env                # (not in repo)
├── .gitignore
├── requirements.txt
├── knowledge_base/
│   └── (your .txt files)
├── vectorstore/
│   └── (auto-generated index; not in repo)
└── README.md
```

## 🛡️ Security & Best Practices

- **Never push your `.env` file or any API keys to GitHub.**
- Use Streamlit Cloud’s "Secrets management" for cloud deployments.
- Document any required environment variables in your README.

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch
3. Commit and push your changes
4. Open a Pull Request 🚀

## 📄 License

MIT License

## 🙏 Acknowledgements

- **Groq:** for blazing-fast Llama-3 API
- **Hugging Face:** for MiniLM embeddings
- **LangChain:** for the RAG framework
- **Streamlit:** for an intuitive UI

> **Tip:** No dataset training or ML expertise needed. Just bring your questions and let the chatbot do the rest!
