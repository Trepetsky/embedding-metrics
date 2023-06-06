import jax
import jax.numpy as jnp
from jax.scipy.linalg import svd
from jax import random
from tqdm import tqdm

def report_all_metrics(tensor):
    tensor = jnp.array(tensor, dtype=jnp.float32)
    u, s, _ = svd(tensor, full_matrices=False)
    fns = [
        rankme,
        coherence,
        pseudo_condition_number,
        alpha_req,
        stable_rank,
        ne_sum,
        avg_self_clustering
    ]
    return {fn.__name__: fn(tensor, u=u, s=s) for fn in tqdm(fns)}


def pseudo_condition_number(tensor, s=None, epsilon=1e-12, **_):
    if s is None:
        s = svd(tensor, compute_uv=False)
    return (s[-1] / (s[0] + epsilon)).item()


def coherence(tensor, u=None, **_):
    if u is None:
        u, _, _ = svd(tensor, compute_uv=True, full_matrices=False)
    maxu = jnp.linalg.norm(u, axis=1).max() ** 2
    return (maxu * u.shape[0] / u.shape[1]).item()


def stable_rank(tensor, s=None, epsilon=1e-12, **_):
    if s is None:
        s = svd(tensor, compute_uv=False)
    trace = jnp.square(tensor).sum()
    denominator = s[0] * s[0] + epsilon
    return (trace / denominator).item()


def rankme(tensor, s=None, epsilon=1e-12, **_):
    if s is None:
        s = svd(tensor, compute_uv=False)
    p_ks = s / jnp.sum(s + epsilon) + epsilon
    return jnp.exp(-jnp.sum(p_ks * jnp.log(p_ks))).item()


def ne_sum(tensor, epsilon=1e-12, **_):
    cov_t = jnp.cov(tensor.T)
    ei_t = jnp.linalg.eigvalsh(cov_t) + epsilon
    return (ei_t / ei_t[-1]).sum().item()


def alpha_req(tensor, s=None, epsilon=1e-12, **_):
    if s is None:
        s = svd(tensor, compute_uv=False)
    n = s.shape[0]
    s += epsilon
    features = jnp.vstack([jnp.linspace(1, 0, n), jnp.ones(n)]).T
    a, _, _, _ = jnp.linalg.lstsq(features, jnp.log(s), rcond=None)
    return a[0].item()


def self_clustering(tensor, epsilon=1e-12, **_):
    tensor = tensor + epsilon
    tensor /= jnp.linalg.norm(tensor, axis=1)[:, jnp.newaxis]
    n, d = tensor.shape
    expected = n + n * (n - 1) / d
    actual = jnp.sum(jnp.square(tensor @ tensor.T))
    return ((actual - expected) / (n * n - expected)).item()


def avg_self_clustering(tensor, num_repetitions=100, batch_size=1000, **_):
    all_scores = []
    key = random.PRNGKey(0)
    
    for _ in range(num_repetitions):
        for batch in get_batches(tensor, batch_size, key):
            score = self_clustering(batch)
            all_scores.append(score)

    avg_all_scores = jnp.mean(jnp.array(all_scores))
    
    return avg_all_scores.item()


def get_batches(tensor, batch_size, key):
    n_samples = tensor.shape[0]
    indices = jax.random.permutation(key, n_samples)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield tensor[batch_indices]