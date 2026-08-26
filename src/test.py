
import jax
import jax.numpy as jnp

def main():
    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.square(a)

    print("JAX version:", jax.__version__)
    print("Backend:", jax.default_backend())
    print("Device:", jax.devices())
    print("Result:", b)

if __name__ == "__main__":
    main()