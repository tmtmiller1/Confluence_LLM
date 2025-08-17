```mermaid
%%{init: {"theme":"base"}}%%
flowchart TB
  Q["Query"] --> E2["Load encoder"]
  E2 --> V2["Encode to vector"]
  V2 --> C2["psycopg connect"]
  C2 --> SQL2["SELECT ordered by embedding <=> (oversample)"]
  U["User profile"] --> P2["Preference boosts"]
  SQL2 --> P2 --> R2["Recency time decay"]
  R2 --> A2{"ACL filter"}
  A2 --> K2["Top K"]
