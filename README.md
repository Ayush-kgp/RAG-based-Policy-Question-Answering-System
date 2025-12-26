# RAG-based-Policy-Question-Answering-System

This project implements a **Retrieval-Augmented Generation (RAG)** system to answer questions from company policy documents. The system retrieves relevant document chunks using semantic search and generates grounded answers using a Large Language Model (LLM), with a strong focus on **hallucination avoidance**, **prompt engineering**, and **evaluation**.

## Objectives

- Answer questions **only using retrieved policy context**
- Handle **missing or out-of-scope questions gracefully**
- Demonstrate **prompt iteration and improvement**
- Evaluate the system on **answerable, partially answerable, and unanswerable questions**

---

## Dataset

The knowledge base consists of **three publicly available policy documents**, adapted for demonstration purposes:

1. **TnC – Direct Intracity** (includes service rules and FAQs)  
2. **TnC – Direct National Courier** (includes service rules and FAQs)  
3. **Privacy Policy** (data collection and usage practices)

Each document is stored separately to preserve service-specific context.

---

### Prompt Comparison (v1 vs v2)

To evaluate the impact of prompt engineering, we compared two prompt versions using the same retrieval settings (k = 8).

**Prompt v1 (Initial):**
- Basic context-based QA
- No explicit grounding or fallback rules
- Resulted in unsafe extrapolation and hallucinations

**Observed issues with Prompt v1:**
- Hallucinated answers for unanswerable questions (e.g., courier partner for intracity deliveries)
- Added external suggestions not present in the documents (e.g., advising users to contact customer service)
- No clear distinction between answerable and unanswerable cases

**Prompt v2 (Improved):**
- Enforced strict grounding to retrieved context
- Explicit fallback for missing information
- Prevented hallucinations and unsafe assumptions

This comparison demonstrates how prompt constraints significantly improve reliability and safety in RAG-based systems.

## Architecture Overview

Policy Documents -> 
Chunking + Metadata (data_loader.py) ->
Embeddings + Vector Store (FAISS) ->
Semantic Retrieval (Top-k) ->
Prompt Construction ->
LLM Answer Generation

---

## Data Preparation & Chunking

- **Chunk size:** ~600 characters was chosen to ensure that each chunk contains enough semantic context (e.g., full policy clauses or rules) without becoming too large and diluting relevance during retrieval.
- **Chunk overlap:** ~100 characters was added to preserve continuity across adjacent policy sections, preventing loss of important boundary information when clauses span multiple chunks.
- **Splitter:** RecursiveCharacterTextSplitter was used to respect natural text boundaries (paragraphs and line breaks) rather than splitting arbitrarily, which improves semantic coherence.

### FAQ Handling
- FAQ sections are explicitly identified
- Each FAQ question–answer pair is treated as an **atomic chunk**
- FAQs are never split across chunks

### Metadata attached to each chunk
- `doc_name` (source document)
- `section` (`policy` or `faq`)

This improves retrieval accuracy and traceability.

---

## Retrieval Strategy

- **Vector Store:** FAISS (lightweight and cross-platform)
- **Embeddings:** OpenAI embeddings
- **Retrieval method:** Top-k semantic similarity search
- **k value:** 8

FAISS was chosen to avoid platform-specific dependency issues and to keep the system simple and robust.

---

## Evaluation

### Evaluation Set
A small handcrafted evaluation set consisting of:
- Answerable questions
- Partially answerable questions
- Unanswerable (out-of-scope) questions

### Evaluation Criteria
- **Accuracy**
- **Hallucination avoidance**
- **Answer clarity**

### Scoring Rubric
- ✅ Correct and grounded  
- ⚠️ Partially correct / conservative  
- ❌ Hallucinated or incorrect  

### Observations
- The system correctly avoided hallucinations in all unanswerable cases.
- In one answerable privacy-related question, the system returned a fallback due to conservative retrieval. This behavior was preferred over hallucination and reflects a safety-first design.

---

## Edge Case Handling

The system explicitly handles:
- No relevant documents retrieved
- Questions outside the knowledge base

In both scenarios, the model returns a clear fallback response instead of fabricating information.

---

## Optional Enhancements (Bonus)

- Implemented and compared **two prompt versions (v1 vs v2)**
- Used programmatic prompt templating via LangChain
- Added basic logging during evaluation to trace questions and model responses

Advanced optimizations (reranking, schema validation) were intentionally avoided to keep the system aligned with the assignment scope.

---

## How to Run

### 1. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Set API key
```bash
OPENAI_API_KEY=your_api_key_here
```
### 4. Run evaluation

```bash
python evaluate.py

```

## Key Trade-offs and Future Improvements

**FAISS vs. Persistent Vector Databases:**  
FAISS was chosen for its lightweight setup and reliability in a local experimentation context. With more time and production requirements, a persistent vector database such as Chroma or Pinecone would be preferred for scalability and persistence.

**Top-K Retrieval:**  
The system currently uses `k = 8`, which improves recall but increases token usage. A reranking step using a cross-encoder could improve precision by selecting only the top 2–3 most relevant chunks for generation.

**Hybrid Search:**  
Future iterations could combine keyword-based retrieval (e.g., BM25) with semantic search to better handle specific entity-based queries such as vehicle names or regulatory terms.
