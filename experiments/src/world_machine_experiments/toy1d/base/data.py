import numpy as np

from world_machine_experiments.shared.split_data import split_data_dict


def _state_transition_matrix() -> np.ndarray:
    t = 0.1

    F = np.eye(3)
    F[0, 1] = t
    F[0, 2] = np.power(t, 2)/2
    F[1, 0] = -0.1*t
    F[1, 2] = t

    return F


def _state_control(state: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    # = G@u or B@u

    state_range = np.sqrt(state.max(axis=0))

    if state_range[2] == 0:
        state_range[2] = 5

    n_sequence = state.shape[0]
    state_control = np.zeros((n_sequence, 3))
    state_control[:, 0] = state_range[0]*(generator.random(n_sequence)-0.5)
    state_control[:, 1] = state_range[1]*(generator.random(n_sequence)-0.5)
    state_control[:, 2] = state_range[2]*(generator.random(n_sequence)-0.5)

    return state_control


def _observation_matrix(generator: np.random.Generator) -> np.ndarray:
    H = generator.random((2, 3))  # NOSONAR
    H = 2*(H-0.5)

    return H


def toy1d_data(n_sequence: int = 10000, sequence_length: int = 1000,
               generator_numpy: np.random.Generator | None = None) -> dict[str, np.ndarray]:

    if generator_numpy is None:
        generator_numpy = np.random.default_rng(0)
    generator = generator_numpy

    H = _observation_matrix(generator)  # NOSONAR
    F = _state_transition_matrix()

    # Sequence generation
    state = np.zeros((n_sequence, 3))

    states = np.empty((sequence_length, n_sequence, 3))
    state_controls = np.empty((sequence_length, n_sequence, 3))

    for i in range(sequence_length):
        state: np.ndarray = np.dot(
            F, state.reshape(-1, 3).T).T.reshape(state.shape)

        Gu = _state_control(state, generator)  # NOSONAR
        state += Gu

        state[1] = np.clip(state[1], -1, 1)
        state[2] = np.clip(state[1], -1, 1)

        states[i] = state
        state_controls[i] = Gu

    # States
    states = np.transpose(states, (1, 0, 2))

    state_max = states[:, :, 0].max()
    state_min = states[:, :, 0].min()

    states = (states - state_min)/(state_max-state_min)
    states = 2*(states-0.5)

    # State Controls
    state_controls = np.transpose(state_controls, (1, 0, 2))

    state_controls = (state_controls - state_controls.min()) / \
        (state_controls.max()-state_controls.min())
    state_controls = 2*(state_controls-0.5)

    # Measurements
    next_states = np.roll(states, shift=-1, axis=0)
    measurements = np.dot(
        H, next_states.reshape(-1, 3).T).T.reshape((n_sequence, sequence_length, 2))

    data = {"states": states, "state_controls": state_controls,
            "next_measurements": measurements}

    return data


def toy1d_data_splitted(toy1d_data: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    return split_data_dict(toy1d_data)
