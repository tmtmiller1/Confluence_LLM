```mermaid
%%{init: {"theme":"base"}}%%
flowchart TB
  Q["Query"] --> HB["Hybrid.search"]
  HB --> BMN["BM25 search (k x 5)"]
  HB --> VVN["Vector search (k x 5)"]
  BMN --> N1["Normalize BM25 scores (per query)"]
  VVN --> N2["Normalize vector scores (per query)"]
  N1 --> BL["Blend score = alpha * vector + (1 - alpha) * bm25"]
  N2 --> BL
  BL --> A4{"ACL filter"}
  A4 --> K4["Top K"]
