import numpy as np
from scipy import signal

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


def _periodic_state_control(state: np.ndarray, generator: np.random.Generator, index: int, control_state: dict, sequence_length: int) -> np.ndarray:
    # = G@u or B@u

    n_sequence = state.shape[0]

    if len(control_state) == 0:

        period = np.full(n_sequence, sequence_length/10)
        period += 2*(generator.random(n_sequence)-0.5)*period

        phase = generator.random(n_sequence)*period
        t = np.linspace(phase, sequence_length+phase,
                        sequence_length, endpoint=False).T
        control = signal.square(2 * np.pi * ((1/period)*t.T).T)

        control_state["values"] = control

        period = np.full(n_sequence, sequence_length/10)
        period += 2*(generator.random(n_sequence)-0.5)*period

        phase = generator.random(n_sequence)*period
        t = np.linspace(phase, sequence_length+phase,
                        sequence_length, endpoint=False).T
        control_pulse = signal.square(2 * np.pi * ((1/period)*t.T).T)
        control_pulse = (control_pulse-np.roll(control_pulse, 1, 1))/2

        control_state["pulse"] = control_pulse

    state_range = np.sqrt(state.max(axis=0))

    state_control = np.zeros((n_sequence, 3))
    state_control[:, 0] = 10*state_range[0]*control_state["pulse"][:, index]
    # state_control[:, 1] = control_state["values"][:, index]
    state_control[:, 2] = (0.75+(0.25*generator.random(
        n_sequence)))*control_state["values"][:, index]

    return state_control


def _observation_matrix(generator: np.random.Generator, measurement_size: int = 2) -> np.ndarray:
    if measurement_size <= 2:
        H = generator.random((2, 3))
        H = H[:measurement_size, :]
    else:
        H = generator.random((measurement_size, 3))

    H = 2*(H-0.5)

    return H


def toy1d_data(n_sequence: int = 10000, sequence_length: int = 1000,
               generator_numpy: np.random.Generator | None = None,
               state_control: str | None = None,
               measurement_size: int = 2) -> dict[str, np.ndarray]:

    if generator_numpy is None:
        generator_numpy = np.random.default_rng(0)
    generator = generator_numpy

    H = _observation_matrix(generator, measurement_size)  # NOSONAR
    F = _state_transition_matrix()

    # print(H)

    # Sequence generation
    state = (2*generator.random((n_sequence, 3)))-1

    states = np.empty((sequence_length, n_sequence, 3))
    state_controls = np.empty((sequence_length, n_sequence, 3))

    control_state = {}

    for i in range(sequence_length):
        state: np.ndarray = np.dot(
            F, state.reshape(-1, 3).T).T.reshape(state.shape)

        if state_control is not None:
            if state_control == "random":
                Gu = _state_control(state, generator)  # NOSONAR
            elif state_control == "periodic":
                Gu = _periodic_state_control(
                    state, generator, i, control_state, sequence_length)

            state += Gu
            state_controls[i] = Gu

        state[1] = np.clip(state[1], -1, 1)
        state[2] = np.clip(state[2], -1, 1)

        states[i] = state

    # States
    states = np.transpose(states, (1, 0, 2))

    # State Controls
    state_controls = np.transpose(state_controls, (1, 0, 2))

    # Measurements
    measurements = np.dot(
        H, states.reshape(-1, 3).T).T.reshape((n_sequence, sequence_length, measurement_size))

    data = {"state_decoded": states, "state_control": state_controls,
            "measurement": measurements}

    return data


def toy1d_data_splitted(toy1d_data: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    return split_data_dict(toy1d_data)


def resize(sequence):
    return sequence
