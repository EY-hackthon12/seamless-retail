# Data Flow

```mermaid
flowchart LR
    subgraph INPUT["Data Sources"]
        API[REST API]
        WS[WebSocket]
        CSV[Batch Files]
        STREAM[Event Stream]
    end

    subgraph PROCESS["Processing"]
        QUEUE[Message Queue<br/>Redis Streams]
        ETL[ETL Pipeline<br/>PySpark]
        FEAT[Feature Store]
    end

    subgraph STORE["Storage"]
        DELTA[Data Lake<br/>Parquet]
        VECTOR[Vector DB<br/>ChromaDB]
        SQL[PostgreSQL]
        KV[Redis<br/>Sessions]
    end

    subgraph ML["ML Pipeline"]
        TRAIN[Training<br/>QLoRA]
        SERVE[Inference<br/>vLLM/Triton]
    end

    API --> QUEUE
    WS --> QUEUE
    STREAM --> QUEUE
    CSV --> ETL
    
    QUEUE --> ETL
    ETL --> FEAT
    FEAT --> DELTA
    FEAT --> VECTOR
    FEAT --> SQL
    
    DELTA --> TRAIN
    VECTOR --> SERVE
    TRAIN --> SERVE
    
    SQL <--> SERVE
    KV <--> SERVE
```

## Data Stores

| Store | Type | Use Case |
|-------|------|----------|
| Parquet | Data Lake | Training data, analytics |
| ChromaDB | Vector | Semantic search, RAG |
| PostgreSQL | Relational | Products, orders, users |
| Redis | Key-Value | Sessions, cache, queues |
