
import jax
import jax.numpy as jnp

@jax.jit
def naive_attention(
    query: jax.Array, 
    key: jax.Array, 
    value: jax.Array
) -> jax.Array:
    """Compute attention given query, key, and value tensors.
    Inspired by DD2424
    
    Args:
        query: A tensor of shape (seq_len, dim)
        key: A tensor of shape (seq_len, dim)
        value: A tensor of shape (seq_len, dim)"""
    scale = jnp.sqrt(query.shape[-1])
    scores = query @ key.T
    scores = scores / scale
    weights = jax.nn.softmax(scores, axis=-1)
    output = weights @ value
    return output

def create_inputs(
    seq_len: int = 512,
    dim: int = 64,
    seed: int = 0
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Create random query, key, and value tensors for testing"""
    key = jax.random.PRNGKey(seed)
    a, b, c = jax.random.split(key, 3)
    query = jax.random.normal(a, (seq_len, dim))
    key = jax.random.normal(b, (seq_len, dim))
    value = jax.random.normal(c, (seq_len, dim))
    return query, key, value

def main():
    # Create random inputs
    query, key, value = create_inputs()

    # Compute attention
    output = naive_attention(query, key, value)
    output.block_until_ready() 

    print("Input shape:", query.shape)
    print("Output shape:", output.shape)
    print("Output mean:", float(output.mean()))

if __name__ == "__main__":
    main()
    