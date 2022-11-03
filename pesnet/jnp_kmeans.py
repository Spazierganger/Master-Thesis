import jax
import jax.nn as jnn
import jax.numpy as jnp
import time


def initialize_centroids(key, points, k):
    idx = jax.random.choice(key, points.shape[0], shape=(k,), replace=False)
    return points[idx]


def closest_centroid(points, centroids):
    distances = jnp.linalg.norm(points[:, None] - centroids[None], keepdims=False, axis=-1)
    return jnp.argmin(distances, axis=1)


def new_mean_centroid(arr, centroid_idx, idx):
    mask = centroid_idx == idx
    mask = jnp.array(mask, dtype=jnp.float32)
    mask /= mask.sum()
    return jnp.sum(arr * mask[:, None], axis=0, keepdims=False)


batch_new_mean_centroid = jax.vmap(new_mean_centroid, in_axes=(None, None, 0))


def update_centroid(arr, centroid_idx, k):
    new_centroids = batch_new_mean_centroid(arr, centroid_idx, jnp.arange(k))
    return new_centroids


def compute_distortion(arr, centroid_idx, centroid, idx):
    mask = centroid_idx == idx
    mask = jnp.array(mask, dtype=jnp.float32)
    dist = arr - centroid[None]
    return jnp.sum((dist ** 2) * mask[:, None])

batch_compute_distortion = jax.vmap(compute_distortion, in_axes=(None, None, 0, 0))
    

def kmeans_run(key, arr, k, n_iter=100):    
    centroids = initialize_centroids(key, arr, k)
    for _ in range(n_iter):
        centroid_idx = closest_centroid(arr, centroids)
        centroids = update_centroid(arr, centroid_idx, k)
        distortion = batch_compute_distortion(arr, centroid_idx, centroids, jnp.arange(k)).sum()
    return centroids, distortion

batch_kmeans = jax.vmap(kmeans_run, in_axes=(0, None, None, None))

def kmeans(arr, k, n_kmeans=20, n_iter=100, key=None):
    if key is None:
        key = jax.random.PRNGKey(int(time.time()))
    subkeys = jax.random.split(key, n_kmeans)
    cens, dists = batch_kmeans(subkeys, arr, k, n_iter)
    return cens[jnp.argmin(dists)]
