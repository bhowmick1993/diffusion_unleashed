# Step 1: Understanding Diffusion Models

## What are Diffusion Models?

Diffusion models are a class of generative models that learn to generate data by reversing a gradual noising process. Think of it like this: if you can learn to remove noise from an image step by step, you can generate new images by starting with pure noise and iteratively denoising it.

## Core Concepts

### 1. Forward Diffusion Process (Adding Noise)

The forward process gradually adds Gaussian noise to an image over T timesteps:

```
x₀ → x₁ → x₂ → ... → xₜ → ... → xₜ
```

At each timestep t, we add a small amount of noise:
- **x₀**: Original clean image
- **xₜ**: Image at timestep t (more noisy)
- **xₜ**: Pure noise (after T steps)

**Mathematical Formulation:**
```
q(xₜ|xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI)
```

Where:
- βₜ is the noise schedule (how much noise to add at step t)
- The process is designed so we can sample xₜ directly from x₀ in one step

### 2. Reverse Diffusion Process (Removing Noise)

The reverse process learns to denoise:
```
xₜ → xₜ₋₁ → ... → x₁ → x₀
```

We train a neural network to predict the noise at each step:
```
ε_θ(xₜ, t) ≈ ε (the noise that was added)
```

### 3. Training Objective

The model learns to predict the noise that was added:
```
L = E[||ε - ε_θ(xₜ, t)||²]
```

Where:
- ε is the actual noise added
- ε_θ(xₜ, t) is the model's prediction
- We minimize the mean squared error

## Key Insights

1. **Noise Schedule**: The βₜ values control how quickly noise is added. Common schedules:
   - Linear: βₜ increases linearly
   - Cosine: βₜ follows a cosine schedule (better for high-quality images)

2. **U-Net Architecture**: The model uses a U-Net (encoder-decoder with skip connections) because:
   - It preserves spatial information
   - Skip connections help with fine details
   - It's proven effective for image-to-image tasks

3. **Timestep Conditioning**: The model receives the timestep t as input (via positional embeddings) so it knows which step of denoising it's performing.

## Intuition: Why Does This Work?

1. **Easy to Learn**: Removing a small amount of noise is easier than generating an entire image from scratch
2. **Stable Training**: The gradual process leads to stable gradients
3. **High Quality**: By making many small corrections, the model can generate high-quality images

## Comparison to Other Generative Models

| Model Type | Training | Sampling | Quality |
|------------|----------|----------|---------|
| **GANs** | Unstable, mode collapse | Fast (single pass) | High |
| **VAEs** | Stable | Fast | Lower quality |
| **Diffusion** | Stable | Slower (iterative) | Very high |

## Next Steps

Now that you understand the theory, let's set up your environment in [Step 2: Environment Setup](./02-environment-setup.md).

