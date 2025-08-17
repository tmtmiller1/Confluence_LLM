```mermaid
%%{init: {"theme":"base"}}%%
flowchart TB
  %% Entry
  Q["Query string"] --> S["Store.search"]
  S --> SEL{"STORE env selects backend"}

  %% Branches
  SEL -->|bm25| BM0["BM25 path"]
  SEL -->|faiss| FA0["FAISS path"]
  SEL -->|pgvector| PG0["pgvector path"]

  %% ===== BM25 (store.py) =====
  subgraph BM25_backend
    direction TB
    BM0 --> BM1["Tokenize query"]
    BM1 --> BM2["Compute BM25 scores"]
    BM2 --> BM3["Preference boosts"]
    BM3 --> BM4["Recency time decay"]
    BM4 --> BM5{"ACL filter"}
    BM5 --> BM6["Top K"]
  end

  %% ===== FAISS (store.py) =====
  subgraph FAISS_backend
    direction TB
    FA0 --> FA1["Load encoder SentenceTransformer"]
    FA1 --> FA2["Encode query to vector"]
    FA2 --> FA3["FAISS IndexFlatIP search oversample"]
    FA3 --> FA4["Preference boosts"]
    FA4 --> FA5["Recency time decay"]
    FA5 --> FA6{"ACL filter"}
    FA6 --> FA7["Top K"]
  end

  %% ===== pgvector (store_pgvector.py) =====
  subgraph PGVECTOR_backend
    direction TB
    PG0 --> PG1["Load encoder SentenceTransformer"]
    PG1 --> PG2["Encode query to vector"]
    PG2 --> PG3["psycopg connect"]
    PG3 --> PG4["Ensure extension vector and table rag_docs"]
    PG4 --> PG5["SELECT order by embedding <=> LIMIT oversample"]
    PG5 --> PG6["Preference boosts"]
    PG6 --> PG7["Recency time decay"]
    PG7 --> PG8{"ACL filter"}
    PG8 --> PG9["Top K"]
  end

  %% Shared inputs to boosts
  U["User profile data"] --> BM3
  U --> FA4
  U --> PG6
