```mermaid
%%{init: {"theme":"base"}}%%
flowchart TB
  Q["Query"] --> TK["Tokenize"]
  TK --> BM["BM25 score (k1=1.5, b=0.75)"]
  U["User profile"] --> PB["Preference boosts"]
  BM --> PB --> RD["Recency time decay"]
  RD --> AF{"ACL filter"}
  AF --> TOP["Top K"]
