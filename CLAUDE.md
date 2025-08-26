# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an image embedding extraction service that provides REST API endpoints for generating embeddings from images using various computer vision models. The service supports multiple model architectures including DINOv2 and InternViT variants, with different extraction strategies for various use cases.

## Development Commands

### Running the Application
```bash
python3.11 start.py
```

### Running the FastAPI Service
```bash
# The service will start on port 8000 by default
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Docker Build and Run
```bash
# Build Docker image
docker build -t dino-embedding-service .

# Run container
docker run -p 8000:8000 dino-embedding-service
```

### Installing Dependencies
```bash
pip install -r req.txt
```

## Architecture

### Core Components

**Embedding Extractors** (`embedding_handler.py`):
- `Dino2ExtractorV1`: Uses Facebook's DINOv2 model (dinov2_vitg14) for feature extraction
- `InternVIT600mbExtractor`: Uses OpenGVLab's InternViT-300M model for lightweight extraction
- `InternVITThreeLevelExtractor`: Advanced extractor with global, focused, and tile-based features
- `InternVITSimpleExtractor`: Simple InternViT feature extractor using mean pooling

**Image Loading** (`embedding_handler.py`):
- `URLImageLoader`: Downloads and processes images from URLs

**Service Layer** (`embedding_handler.py`):
- `EmbeddingService`: Coordinates image loading and embedding extraction

**API Layer** (`api.py`):
- FastAPI application with two main endpoints:
  - `/embedding/fast_extract`: Uses DINOv2 for extraction
  - `/embedding/test_extract`: Uses InternViT with concurrency control

**Controller Layer** (`controller.py`):
- Batch processing workflows for building and task collections
- Handles interaction with external ArtCracker API

**Request Handler** (`request_handler.py`):
- HTTP client for ArtCracker API endpoints
- Handles authentication and data transfer

### Key Features

1. **Multiple Model Support**: DINOv2 and InternViT models with different configurations
2. **Concurrency Control**: Semaphore-based request limiting for model inference
3. **Three-Level Feature Extraction**: Global, focused, and tile-based features for comprehensive analysis
4. **External API Integration**: Communicates with ArtCracker backend for data synchronization
5. **Batch Processing**: Support for processing entire collections of images

### Configuration

The application requires a `.env` file with:
- `TOKEN`: Authentication token for ArtCracker API

### Device Requirements

- CUDA-compatible GPU (models default to 'cuda' device)
- Python 3.11
- NVIDIA CUDA 11.8+ (as specified in Dockerfile)

### Model Loading

Models are loaded on service startup and cached in memory. The service automatically handles:
- Model downloading from HuggingFace Hub
- GPU memory management
- Image preprocessing and normalization

### API Endpoints

- `POST /embedding/fast_extract`: Fast embedding extraction using DINOv2
- `POST /embedding/test_extract`: InternViT extraction with queue management
- `GET /`: Health check endpoint

Request format:
```json
{
  "url": "https://example.com/image.jpg"
}
```

Response format:
```json
{
  "embedding": [0.1, 0.2, ...],
  "url": "https://example.com/image.jpg"
}
```