```mermaid
%%{init: {"theme":"base"}}%%
flowchart TB
  Q["Query"] --> E1["Load encoder"]
  E1 --> VQ["Encode to vector"]
  VQ --> S1["FAISS IndexFlatIP search (oversample)"]
  U["User profile"] --> P1["Preference boosts"]
  S1 --> P1 --> R1["Recency time decay"]
  R1 --> A1{"ACL filter"}
  A1 --> K1["Top K"]
