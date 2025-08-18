```mermaid
%%{init: {"theme":"base"}}%%
flowchart TB
  Q["Query"] --> H["Hybrid.search"]
  H --> BMRUN["BM25 search (k x 5)"]
  H --> VRUN["Vector search (k x 5)"]
  BMRUN --> FUSE["Reciprocal Rank Fusion"]
  VRUN --> FUSE
  FUSE --> A3{"ACL filter"}
  A3 --> K3["Top K"]
