# Audio Vector Database Benchmarking

A comparative analysis of Qdrant and Weaviate vector databases for audio embedding storage and retrieval.

## Overview

This project conducts a comprehensive benchmark comparison between two leading vector databases (Qdrant and Weaviate) for storing and querying audio embeddings. Using audio data from the blues genre, the analysis evaluates which database provides better performance across multiple critical dimensions: search speed, filtering capabilities, accuracy of similarity matching, and resource consumption.

Two different audio embedding methods were used to ensure robust comparison:
- Transformer-based embeddings (768 dimensions)
- PANNs (Pretrained Audio Neural Networks) embeddings (2048 dimensions)

## Motivation

Vector databases are increasingly important for audio applications like music recommendation, genre classification, and audio similarity search. This project aims to answer key questions:

1. Which vector database performs better for audio data?
2. How do different embedding dimensions affect database performance?
3. What are the trade-offs between search speed, accuracy, and resource usage?
4. How do filtering capabilities compare across databases?

## Dataset

This benchmark uses the [Ludwig Music Dataset (Moods and Subgenres)](https://www.kaggle.com/datasets/dhruvildave/ludwig-music-dataset-moods-and-subgenres), which includes:

- 601 MP3 audio files from the blues genre
- Rich metadata including artist, genre, name, and subgenres
- Diverse subgenres (electric blues, country blues, etc.) allowing for nuanced similarity testing

## Benchmark Methodology

The benchmarking process consists of five main components:

1. **Data Processing**: Loading audio files and processing metadata
2. **Embedding Generation**: Creating vector embeddings using Transformer and PANNs models
3. **Database Setup**: Creating collections and uploading embeddings to both Qdrant and Weaviate
4. **Search & Filter Benchmarking**: Testing search speed and filtering capabilities
5. **Accuracy & Resource Evaluation**: Measuring genre matching precision and resource consumption

Each benchmark was executed using identical parameters across both databases to ensure fair comparison.

## Dependencies

- Python 3.8+
- PyTorch
- Transformers
- PANNs-inference
- Qdrant Client
- Weaviate Client
- Pandas
- NumPy
- Matplotlib
- Librosa

## Benchmark Results

### Embedding Accuracy

Both embedding methods show strong performance in matching songs by genre and subgenre:

| Method | Database | Genre Precision | Subgenre Precision | Combined |
|--------|----------|-----------------|-------------------|----------|
| PANNs | Qdrant | 100% | 65.17% | 82.59% |
| Transformer | Qdrant | 100% | 58.28% | 79.14% |
| PANNs | Weaviate | 100% | 78.33% | 89.17% |
| Transformer | Weaviate | 100% | 66.33% | 83.17% |

### Search Performance

Vector search performance at various result sizes (k):

| Method | Database | k=1 | k=10 | k=50 | k=100 |
|--------|----------|-----|------|------|-------|
| Transformer | Qdrant | 22.36 ms | 22.41 ms | 23.00 ms | 23.99 ms |
| PANNs | Qdrant | 22.55 ms | 22.32 ms | 22.80 ms | 23.64 ms |
| Transformer | Weaviate | 435.18 ms | 323.14 ms | 382.53 ms | 489.40 ms |
| PANNs | Weaviate | 339.23 ms | 341.16 ms | 391.96 ms | 504.92 ms |

### Resource Usage

Memory and computational resource requirements:

| Method | Database | Memory per Vector | Search Time | Filtered Search | Recommendation |
|--------|----------|-------------------|-------------|-----------------|----------------|
| Transformer | Qdrant | 3.0 KB | 27.03 ms | 26.77 ms | 23.09 ms |
| PANNs | Qdrant | 8.0 KB | 23.41 ms | 23.63 ms | 23.67 ms |
| Transformer | Weaviate | 3.0 KB | 384.00 ms | 381.64 ms | 389.01 ms |
| PANNs | Weaviate | 8.0 KB | 394.63 ms | 396.25 ms | 397.78 ms |

## Key Findings

1. **Embedding Quality**:
   - PANNs embeddings (2048 dimensions) consistently outperform Transformer embeddings (768 dimensions) in accuracy
   - Subgenre precision is significantly better with PANNs, particularly in Weaviate

2. **Database Performance**:
   - Qdrant delivers significantly faster query times than Weaviate (20-25ms vs 320-500ms)
   - Both databases scale efficiently with result size (k)
   - Filtering operations add minimal overhead to search times

3. **Memory Efficiency**:
   - PANNs requires ~2.67x more memory than Transformer embeddings
   - Memory usage scales linearly with vector dimensionality

4. **Trade-offs**:
   - PANNs offers better accuracy at the cost of higher memory usage
   - Transformer provides good accuracy with more efficient storage
   - Qdrant provides faster queries while Weaviate offers slightly better accuracy preservation

## Conclusion & Recommendations

Based on comprehensive benchmarking, here are the key findings and recommendations:

### 1. Performance

- **Search Speed**: Qdrant significantly outperforms Weaviate with ~10-15x faster query times (20-25ms vs 320-500ms)
- **Scaling**: Both databases handle increasing result sizes (k) well, with minimal performance degradation
- **First Query Latency**: Both databases show a "warm-up" effect, with the first query taking significantly longer

### 2. Accuracy

- **Genre Recognition**: Both databases maintain perfect (100%) genre precision
- **Subgenre Precision**: 
  - Weaviate shows better accuracy preservation, especially with PANNs embeddings (78.33% vs 65.17%)
  - Both databases preserve embedding similarity relationships well

### 3. Filtering Capabilities

- **Filter Performance**: Both databases handle complex filters efficiently
- **Qdrant Advantage**: Qdrant's filtering is slightly faster (~23ms vs ~350ms)
- **Complex Filters**: Both databases handle AND, OR and negation conditions effectively

### 4. Resource Usage

- **Memory Efficiency**: Both databases show similar memory usage patterns
- **Embedding Size Impact**: PANNs embeddings (2048 dimensions) require ~2.67x more memory than Transformer embeddings (768 dimensions) in both databases
- **Operation Types**: Search, filtered search, and recommendation operations show similar performance within each database

### 5. Collection Creation & Upload Time

- **Creation Speed**: Both databases create collection schemas quickly (~0.5s)
- **Upload Performance**: Qdrant is significantly faster at data ingestion than Weaviate
  - Transformer embeddings: 0.54s (Qdrant) vs 113.09s (Weaviate)
  - PANNs embeddings: 0.51s (Qdrant) vs 303.51s (Weaviate)

## Recommendations

**For Real-time Applications**:
- Use Qdrant if query speed is the primary concern (music players, live recommendations)
- Consider Transformer embeddings if memory efficiency is important

**For High-Accuracy Applications**:
- Use Weaviate with PANNs embeddings if maximum similarity precision is required
- Accept the trade-off of slower query time for better accuracy

**For Balanced Applications**:
- Qdrant with PANNs embeddings offers a good balance of speed and accuracy
- This combination provides good subgenre precision with excellent query performance

## Future Directions

For further investigation:
- Test with larger datasets (10K+ songs) to evaluate scaling properties
- Benchmark with additional vector databases (Milvus, Pinecone, Chroma, etc.)
- Explore additional embedding methods specialized for music
- Evaluate performance with multi-modal queries (text + audio)
- Test with different audio genres and more diverse metadata

## Acknowledgments

This project utilizes the [Ludwig Music Dataset](https://www.kaggle.com/datasets/dhruvildave/ludwig-music-dataset-moods-and-subgenres) for audio samples and metadata.
