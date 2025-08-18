```mermaid
%%{init: {"theme": "base"}}%%
flowchart LR
  %% ===== Classes =====
  classDef api fill:#E3F2FD,stroke:#64B5F6,color:#0D47A1,stroke-width:1px;
  classDef process fill:#F3F4F6,stroke:#9CA3AF,color:#111827;
  classDef parse fill:#E8EAF6,stroke:#7986CB,color:#1A237E;
  classDef store fill:#F3E5F5,stroke:#BA68C8,color:#4A148C;
  classDef restrict fill:#FFEBEE,stroke:#EF9A9A,color:#B71C1C;
  classDef user fill:#E8F5E9,stroke:#81C784,color:#1B5E20;
  classDef results fill:#FFF8E1,stroke:#FFD54F,color:#E65100;

  %% ===== Ingestion =====
  subgraph Ingestion
    A["Confluence API"]:::api
    I1["HTML/Storage fetch"]:::process
    I2["Download attachments"]:::process
    P["Attachment parsers"]:::parse
    T["Chunk and normalize"]:::process
    R["Restrictions API"]:::restrict
    M["Index store\nBM25 / FAISS / pgvector"]:::store

    A -->|pages| I1
    A -->|attachments| I2
    I1 --> T
    I2 --> P --> T
    A -.-> R
    R --> T
    T --> M
  end

  %% ===== Retrieval =====
  subgraph Retrieval
    Q["Query"]:::process
    S["Store.search"]:::process
    C["Candidate set"]:::results
    U["User profile\nprefs, negatives, recency, groups"]:::user
    B["Re-rank and time-decay"]:::process
    F{"ACL post-filter"}:::restrict
    K(["Top-K"]):::results

    Q --> S -->|BM25 or Vector| C --> B
    U -.-> B
    B --> F --> K
  end

  %% ===== UI =====
  subgraph UI
    ST["Streamlit"]:::user
    Profiles["YAML profiles"]:::user
    Metrics["Metrics"]:::results

    ST --> Profiles -.-> U
    ST --> Q
    ST --> Metrics
  end
