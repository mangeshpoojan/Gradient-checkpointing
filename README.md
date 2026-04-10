# CUDA MLP Forward Pass with Multi-Stream Execution

A CUDA implementation of a multi-layer perceptron (MLP) forward pass using shared memory tiled matrix multiplication and multiple CUDA streams for parallel batch processing.

## Overview

The program runs an MLP forward pass on a batch of inputs entirely on the GPU. The batch is split across multiple CUDA streams so that each stream processes an independent chunk in parallel.

The number of layers is **dynamic** — controlled by the `LAYERS` compile-time constant. Changing `LAYERS` automatically adjusts the depth of the network without any other code changes.

**Architecture (default, `LAYERS=4`):**
- Input layer: 32 → 32 (hidden)
- Hidden layers: 32 → 32 × (`LAYERS` − 2)
- Output layer: 32 → 32
- Activation: ReLU on all hidden layers, none on output

## Key Kernels

### `matmul_gpu_shared_mem_tiling`
Tiled matrix multiplication with bias addition using shared memory.
- Loads 32×32 tiles of A and B into shared memory to reduce global memory traffic
- Bias is loaded into a shared array once before the tiling loop
- Boundary checks handle non-multiples of tile size

### `relu_inplace`
Applies ReLU activation in-place on a flat array.

## Configuration (compile-time constants)

| Constant | Value | Description |
|---|---|---|
| `BATCH_SIZE` | 32 | Total number of input samples |
| `INPUT_VEC` | 32 | Input feature size |
| `OUTPUT_VEC` | 32 | Output size |
| `N` | 32 | Hidden layer size |
| `LAYERS` | 4 | **Number of layers (dynamic)** — change to any value ≥ 2 to adjust network depth |
| `STREAMS` | 4 | Number of CUDA streams |
| `BLOCK_SIZE` | 32 | CUDA thread block tile size |
| `LEARNING_RATE` | 0.001 | Learning rate (reserved for future use) |

- `BATCH_SIZE` must be divisible by `STREAMS`.
- `LAYERS` must be ≥ 2 (at least one input and one output layer).

## Build & Run

```bash
nvcc -O2 -o pa3 PA3.cu
./pa3
```

## Output

```
Time: <elapsed> microseconds
```

The elapsed time covers kernel launches and GPU execution (stream synchronization), but excludes the initial host-to-device memory transfers.
