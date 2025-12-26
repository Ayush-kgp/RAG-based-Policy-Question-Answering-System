from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List


def chunk_policy_document(
    file_path: str,
    doc_name: str,
    faq_header: str | None = "Frequently Asked Questions",
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> List[Document]:
    """
    Load and chunk a policy document.
    - Main policy text is chunked using RecursiveCharacterTextSplitter
    - FAQ Q&A pairs are kept as atomic chunks
    - Metadata is attached to every chunk

    Returns a list of LangChain Document objects.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    documents: List[Document] = []

    #Splitting policy & FAQ 
    if faq_header and faq_header in text:
        policy_text, faq_text = text.split(faq_header, 1)
    else:
        policy_text, faq_text = text, None

    #  Chunk main policy 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    policy_chunks = splitter.create_documents(
        [policy_text],
        metadatas=[{
            "doc_name": doc_name,
            "section": "policy"
        }]
    )

    documents.extend(policy_chunks)

    #  Handle FAQs (atomic Q&A) 
    if faq_text:
        faq_entries = faq_text.strip().split("\n\nQ")

        for faq in faq_entries:
            faq_clean = faq.strip()
            if not faq_clean:
                continue

            if not faq_clean.startswith("Q"):
                faq_clean = "Q" + faq_clean

            documents.append(
                Document(
                    page_content=faq_clean,
                    metadata={
                        "doc_name": doc_name,
                        "section": "faq"
                    }
                )
            )

    return documents


def load_all_documents() -> List[Document]:
    """
    Load and chunk all policy documents used in the project.
    """

    all_chunks: List[Document] = []

    all_chunks.extend(
        chunk_policy_document(
            file_path="Data/TnC_Direct_Intracity.txt",
            doc_name="Data/tnc_direct_intracity"
        )
    )

    all_chunks.extend(
        chunk_policy_document(
            file_path="Data/TnC_Direct_National_Courier.txt",
            doc_name="Data/tnc_direct_national"
        )
    )

    all_chunks.extend(
        chunk_policy_document(
            file_path="Data/Privacy Policy_delhivery.txt",
            doc_name="Data/privacy_policy",
            faq_header=None  # No FAQs in privacy policy
        )
    )

    return all_chunks
