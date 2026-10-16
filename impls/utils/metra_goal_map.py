"""Plot METRA representation distance to a fixed observed goal."""

import os

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def save_metra_goal_map(agent, state_source, save_dir, epoch, seed=0, num_states=2048):
    """Color observed states by ||phi(state) - phi(goal)||.

    The probe is chosen once from actual observed states and reused at every
    evaluation, so consecutive images show changes in the representation.
    """
    if state_source is None or state_source.size == 0:
        return None

    probe_path = os.path.join(save_dir, 'metra_goal_map_probe.npz')
    if os.path.exists(probe_path):
        with np.load(probe_path) as probe:
            observations = probe['observations']
            goal = probe['goal']
    else:
        rng = np.random.default_rng(seed)
        if hasattr(state_source, 'valid_idxs'):
            source_indices = np.asarray(state_source.valid_idxs)
        else:
            source_indices = np.arange(state_source.size)
        indices = rng.choice(source_indices, size=min(num_states, len(source_indices)), replace=False)
        observations = np.asarray(state_source['observations'][indices], dtype=np.float32)
        goal = observations[rng.integers(len(observations))].copy()
        np.savez_compressed(probe_path, observations=observations, goal=goal)

    if observations.ndim != 2 or observations.shape[1] < 2:
        raise ValueError('METRA goal map requires observations with XY in the first two coordinates.')

    phi = agent.representation_network.select('phi')
    goal_phi = np.asarray(phi(goal[None]))[0]
    distances = np.concatenate([
        np.linalg.norm(np.asarray(phi(batch)) - goal_phi, axis=-1)
        for batch in np.array_split(observations, max(1, int(np.ceil(len(observations) / 512))))
    ])
    figure = Figure(figsize=(7, 6), constrained_layout=True)
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    dots = ax.scatter(observations[:, 0], observations[:, 1], c=distances, s=10, cmap='viridis')
    ax.scatter(goal[0], goal[1], marker='*', s=220, c='red', edgecolors='black', label='fixed observed goal')
    ax.set(xlabel='observed x', ylabel='observed y', aspect='equal',
           title=f'METRA representation distance to goal | epoch {epoch}')
    ax.legend(loc='best')
    figure.colorbar(dots, ax=ax, label='||phi(state) - phi(goal)||')
    output_path = os.path.join(save_dir, f'metra_goal_map_{epoch}.png')
    figure.savefig(output_path, dpi=150)
    return output_path
