# The Concept of Low-Rank Adaptation

LoRA addresses this challenge by reducing the number of trainable parameters through low-rank matrix approximations.

## Mathematical Mechanism

Consider a pre-trained weight matrix W₀ with dimensions d × k, where:

- d is the input dimension
- k is the output dimension

During fine-tuning, we typically update W₀ to a new weight matrix W. LoRA proposes modelling the weight update ΔW (which is W - W₀) using two smaller matrices, A and B:

```
ΔW = A × Bᵀ
```

Here:

- A is a d × r matrix
- B is a k × r matrix
- r is a small integer representing the rank, with r much smaller than the minimum of d and k
- Bᵀ denotes the transposition of the matrix B

By doing this, instead of updating the entire d × k weight matrix, we only need to train the much smaller matrices A and B, significantly reducing the number of trainable parameters.

## Example Calculation

Let's work through an example to see how much we reduce the parameters.

Suppose:
- Input dimension d = 1024
- Output dimension k = 1024
- Rank r = 16

### Without LoRA

The number of parameters to update is:

```
Number of parameters = d × k = 1024 × 1024 = 1,048,576 parameters
```

### With LoRA

The number of parameters to train is:

```
Number of parameters = (d × r) + (k × r) = (1024 × 16) + (1024 × 16) = 32,768 parameters
```

### Reduction Factor

```
Reduction Factor = Original Parameters / LoRA Parameters = 1,048,576 / 32,768 = 32
```

So, using LoRA reduces the number of trainable parameters by a factor of 32.