```mermaid
flowchart TB
  %% ===== Inputs =====
  subgraph Inputs
    Q["Query text"]
    ENC["Encode with SentenceTransformer"]
    QV["qvec"]
    TSQ["Build tsquery"]
    Q --> ENC --> QV
    Q --> TSQ
  end

  %% ===== Postgres prep =====
  subgraph Postgres setup
    EXT["Enable vector extension\nAdd tsvector column tsv"]
    IDX["Indexes\nivfflat on embedding\nGIN on tsv"]
    EXT --> IDX
  end

  %% ===== SQL scoring =====
  subgraph SQL compute
    SQLH["SELECT id, meta\nsim = 1 - embedding <=> qvec\nbm = ts_rank_cd(tsv, tsquery)\nblended = 0.6*sim + 0.4*bm\nFROM rag_docs\nWHERE tsv @@ tsquery\nORDER BY blended DESC\nLIMIT k"]
  end

  QV --> SQLH
  TSQ --> SQLH
  IDX --> SQLH

  SQLH --> TOP["Top K rows"]
