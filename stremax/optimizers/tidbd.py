from dataclasses import dataclass

import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from stremax.utils import broadcast
from stremax.utils.typing import Array, PyTree


@struct.dataclass(frozen=True)
class TIDBDConfig:
    meta_lr: float
    init_lr: float = 1e-3
    beta_max: float = 2.0


@struct.dataclass(frozen=True)
class TIDBDState:
    beta: PyTree
    h: PyTree


@dataclass
class TIDBD:
    """Temporal-difference Incremental Delta-Bar-Delta.

    Sutton's IDBD (1992) generalized to TD(λ) with eligibility traces
    (Kearney et al. 2018, "Every step you take"). Adapts a *per-parameter* step
    size ``α = exp(β)`` by meta-gradient descent on the log step size ``β``,
    using a memory trace ``h`` that tracks the accumulated effect past updates
    had on each weight. Standalone: the emitted update ``α ⊙ δ ⊙ z`` is the
    complete weight change, so this is a sibling of ``ObGD`` rather than a
    wrapper over another optimizer.

    Per weight, with TD error ``δ``, eligibility trace ``z`` and current value
    gradient ``φ``::

        β ← β + θ · δ · z · h        # meta-update, uses the *old* h
        α = exp(min(β, β_max))       # per-weight step size
        Δw = α · δ · z               # the returned update
        h ← h · [1 − α·z·φ]⁺ + Δw    # memory trace, uses the new α and φ

    With ``λ = 0`` and a linear network (``z = φ = x``) this collapses to
    Sutton's original IDBD.
    """

    cfg: TIDBDConfig
    name: str = "tidbd"

    def init(self, parameters: PyTree, num_envs: int) -> TIDBDState:
        beta = jax.tree.map(
            lambda p: jnp.full(
                (num_envs, *p.shape), jnp.log(self.cfg.init_lr), dtype=jnp.float32
            ),
            parameters,
        )
        h = jax.tree.map(
            lambda p: jnp.zeros((num_envs, *p.shape), dtype=jnp.float32),
            parameters,
        )
        return TIDBDState(beta=beta, h=h)

    def update(
        self,
        state: TIDBDState,
        gradient: PyTree,
        trace: PyTree,
        td_error: Array,
    ) -> tuple[PyTree, TIDBDState]:
        cfg = self.cfg

        # β ← β + θ δ z h using the *old* h, then clamp to guard exp() blow-up.
        new_beta = jax.tree.map(
            lambda beta, h, z: jnp.minimum(
                beta + cfg.meta_lr * broadcast(td_error, z) * z * h, cfg.beta_max
            ),
            state.beta,
            state.h,
            trace,
        )
        alpha = jax.tree.map(jnp.exp, new_beta)

        # Δw = α δ z, returned averaged over the env axis.
        delta_w = jax.tree.map(
            lambda a, z: a * broadcast(td_error, z) * z, alpha, trace
        )
        updates = jax.tree.map(lambda dw: dw.mean(axis=0), delta_w)

        # h ← h · [1 − α z φ]⁺ + Δw, using the new α and the current gradient φ.
        new_h = jax.tree.map(
            lambda h, a, z, phi, dw: h * jnp.maximum(1.0 - a * z * phi, 0.0) + dw,
            state.h,
            alpha,
            trace,
            gradient,
            delta_w,
        )

        def tree_mean(tree: PyTree) -> Array:
            leaves = jax.tree.leaves(tree)
            return sum(jnp.sum(leaf) for leaf in leaves) / sum(
                leaf.size for leaf in leaves
            )

        lox.log(
            {
                f"{self.name}/update_norm": optax.global_norm(updates),
                f"{self.name}/mean_lr": tree_mean(alpha),
                f"{self.name}/mean_h": tree_mean(new_h),
            }
        )

        return updates, TIDBDState(beta=new_beta, h=new_h)
