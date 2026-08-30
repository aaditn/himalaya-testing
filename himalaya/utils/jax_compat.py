"""Narrow compatibility helpers for Brax on newer JAX releases."""


def install_brax_compatibility() -> None:
    """Restore the removed replication helper with JAX sharding primitives."""
    import jax

    if hasattr(jax, "device_put_replicated"):
        return

    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    def device_put_replicated(value, devices):
        mesh = Mesh(np.array(devices), ("device",))
        sharding = NamedSharding(mesh, P("device"))
        return jax.tree.map(
            lambda leaf: jax.device_put(
                jnp.stack([leaf] * len(devices)), sharding
            ),
            value,
        )

    jax.device_put_replicated = device_put_replicated
