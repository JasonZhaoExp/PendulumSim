import os
import sys
from collections import deque

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# -------------------------------------------------------------------
# GPU Acceleration: Use CuPy if available, otherwise fallback to NumPy
# -------------------------------------------------------------------
try:
    import cupy as cp

    xp = cp
    gpu_enabled = True
    print("GPU acceleration enabled with CuPy.")
except ImportError:
    xp = np
    gpu_enabled = False
    print("CuPy not available, using NumPy for computations.")


def to_cpu(x):
    """Convert an array from GPU (CuPy) to CPU (NumPy) if needed."""
    if gpu_enabled and xp is cp:
        return cp.asnumpy(x)
    return x


# -------------------------------------------------------------------
# TOML loader helper
# -------------------------------------------------------------------
try:
    import tomllib  # Python 3.11+

    _use_tomllib = True
except ImportError:
    try:
        import toml

        _use_tomllib = False
    except ImportError:
        sys.exit(
            "Please install the 'toml' package or use Python 3.11+ for tomllib support."
        )


def load_toml(filename):
    if _use_tomllib:
        with open(filename, "rb") as f:
            return tomllib.load(f)
    else:
        with open(filename, "r", encoding="utf-8") as f:
            return toml.load(f)


# -------------------------------------------------------------------
# General pendulum (n_links >= 3) using Verlet integration with constraints
# -------------------------------------------------------------------
class PendulumString:
    def __init__(
        self,
        n_links,
        base,
        base_angle,
        L=1.0,
        chaos_mode=False,
        dt=0.01,
        history_length=50,
    ):
        self.n_links = n_links
        self.L = L
        self.dt = dt
        self.base = xp.array(base, dtype=xp.float64)

        self.positions = xp.zeros((n_links + 1, 2), dtype=xp.float64)
        self.positions[0] = self.base
        self.prev_positions = xp.zeros_like(self.positions, dtype=xp.float64)

        # Initialize joints along a straight line.
        for i in range(1, n_links + 1):
            if chaos_mode:
                theta = xp.pi + base_angle
            else:
                theta = base_angle
            disp = xp.array([xp.sin(theta), -xp.cos(theta)], dtype=xp.float64) * L
            self.positions[i] = self.positions[i - 1] + disp

        self.prev_positions = xp.copy(self.positions)
        self.history = deque(maxlen=history_length)
        self.history.append(xp.copy(self.positions[-1]))

    def step(self, g):
        acc = xp.array([0, -g], dtype=xp.float64)
        new_positions = xp.copy(self.positions)
        for i in range(1, self.n_links + 1):
            new_positions[i] = (
                self.positions[i]
                + (self.positions[i] - self.prev_positions[i])
                + acc * (self.dt**2)
            )
        self.prev_positions[1:] = xp.copy(self.positions[1:])
        self.positions[1:] = new_positions[1:]
        # Enforce constraints (iterate several times)
        for _ in range(10):
            for i in range(self.n_links):
                p1 = self.positions[i]
                p2 = self.positions[i + 1]
                delta = p2 - p1
                dist = xp.linalg.norm(delta)
                if dist == 0:
                    continue
                diff = (dist - self.L) / dist
                if i == 0:
                    self.positions[i + 1] -= delta * diff
                else:
                    self.positions[i] += 0.5 * delta * diff
                    self.positions[i + 1] -= 0.5 * delta * diff
        self.history.append(xp.copy(self.positions[-1]))

    def get_positions(self):
        return to_cpu(self.positions)

    def get_history(self):
        return to_cpu(xp.array(self.history))


# -------------------------------------------------------------------
# Optimized simulation for single pendulums (n_links == 1)
# -------------------------------------------------------------------
class SinglePendulumString:
    def __init__(
        self, base, base_angle, L=1.0, chaos_mode=False, dt=0.01, history_length=50
    ):
        self.base = xp.array(base, dtype=xp.float64)
        self.L = L
        self.dt = dt
        if chaos_mode:
            self.theta = xp.float64(xp.pi + base_angle)
        else:
            self.theta = xp.float64(base_angle)
        self.omega = xp.float64(0.0)
        self.history = deque(maxlen=history_length)
        self.history.append(xp.copy(self.get_positions()[-1]))

    def step(self, g):
        dt = self.dt

        def f(state):
            theta, omega = state
            return xp.array([omega, -g / self.L * xp.sin(theta)], dtype=xp.float64)

        state = xp.array([self.theta, self.omega], dtype=xp.float64)
        k1 = dt * f(state)
        k2 = dt * f(state + 0.5 * k1)
        k3 = dt * f(state + 0.5 * k2)
        k4 = dt * f(state + k3)
        state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        self.theta, self.omega = state
        self.history.append(xp.copy(self.get_positions()[-1]))

    def get_positions(self):
        x_end = self.base[0] + self.L * xp.sin(self.theta)
        y_end = self.base[1] - self.L * xp.cos(self.theta)
        return to_cpu(xp.array([self.base, [x_end, y_end]], dtype=xp.float64))

    def get_history(self):
        return to_cpu(xp.array(self.history))


# -------------------------------------------------------------------
# Optimized simulation for double pendulums (n_links == 2)
# -------------------------------------------------------------------
class DoublePendulumString:
    def __init__(
        self, base, base_angle, L=1.0, chaos_mode=False, dt=0.01, history_length=50
    ):
        self.base = xp.array(base, dtype=xp.float64)
        self.L = L
        self.dt = dt
        if chaos_mode:
            self.theta1 = xp.float64(xp.pi + base_angle)
            self.theta2 = xp.float64(xp.pi + base_angle)
        else:
            self.theta1 = xp.float64(base_angle)
            self.theta2 = xp.float64(base_angle)
        self.omega1 = xp.float64(0.0)
        self.omega2 = xp.float64(0.0)
        self.history = deque(maxlen=history_length)
        self.history.append(xp.copy(self.get_positions()[-1]))

    def step(self, g):
        dt = self.dt
        L = self.L

        def f(state):
            theta1, theta2, omega1, omega2 = state
            delta = theta1 - theta2
            dtheta1 = omega1
            dtheta2 = omega2
            denom = L * (2 - xp.cos(2 * delta))
            domega1 = (
                -g * (2 * xp.sin(theta1) + xp.sin(theta1 - 2 * theta2))
                - 2 * xp.sin(delta) * (omega2**2 * L + omega1**2 * L * xp.cos(delta))
            ) / denom
            domega2 = (
                2
                * xp.sin(delta)
                * (
                    2 * omega1**2 * L
                    + 2 * g * xp.cos(theta1)
                    + omega2**2 * L * xp.cos(delta)
                )
            ) / denom
            return xp.array([dtheta1, dtheta2, domega1, domega2], dtype=xp.float64)

        state = xp.array(
            [self.theta1, self.theta2, self.omega1, self.omega2], dtype=xp.float64
        )
        k1 = dt * f(state)
        k2 = dt * f(state + 0.5 * k1)
        k3 = dt * f(state + 0.5 * k2)
        k4 = dt * f(state + k3)
        state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        self.theta1, self.theta2, self.omega1, self.omega2 = state
        self.history.append(xp.copy(self.get_positions()[-1]))

    def get_positions(self):
        x1 = self.base[0] + self.L * xp.sin(self.theta1)
        y1 = self.base[1] - self.L * xp.cos(self.theta1)
        joint = xp.array([x1, y1], dtype=xp.float64)
        x2 = x1 + self.L * xp.sin(self.theta2)
        y2 = y1 - self.L * xp.cos(self.theta2)
        endpoint = xp.array([x2, y2], dtype=xp.float64)
        return to_cpu(xp.array([self.base, joint, endpoint], dtype=xp.float64))

    def get_history(self):
        return to_cpu(xp.array(self.history))


# -------------------------------------------------------------------
# Optimized simulation class that picks the integration method based on n_links
# -------------------------------------------------------------------
class OptimizedPendulumSimulation:
    def __init__(
        self, n_strings, n_links, degrees_between, chaos_mode, history_length, g, dt, L
    ):
        self.n_strings = n_strings
        self.n_links = n_links
        self.strings = []
        common_base = [0, 0]  # keep as list for consistency
        for i in range(n_strings):
            if chaos_mode:
                base_angle = xp.deg2rad((i + 1) * degrees_between)
            else:
                base_angle = xp.deg2rad(i * degrees_between)
            if n_links == 1:
                self.strings.append(
                    SinglePendulumString(
                        common_base, base_angle, L, chaos_mode, dt, history_length
                    )
                )
            elif n_links == 2:
                self.strings.append(
                    DoublePendulumString(
                        common_base, base_angle, L, chaos_mode, dt, history_length
                    )
                )
            else:
                self.strings.append(
                    PendulumString(
                        n_links,
                        common_base,
                        base_angle,
                        L,
                        chaos_mode,
                        dt,
                        history_length,
                    )
                )
        self.g = g
        self.dt = dt

    def step(self):
        for s in self.strings:
            s.step(self.g)

    def get_all_positions(self):
        return [s.get_positions() for s in self.strings]

    def get_all_histories(self):
        return [s.get_history() for s in self.strings]


# -------------------------------------------------------------------
# Plotting and animation (same for all cases)
# -------------------------------------------------------------------
def run_simulation(settings):
    sim = OptimizedPendulumSimulation(
        n_strings=settings["n_strings"],
        n_links=settings["n_links"],
        degrees_between=settings["degrees_between"],
        chaos_mode=settings["chaos_mode"],
        history_length=settings["render_history_length"],
        g=settings["g"],
        dt=settings["dt"],
        L=settings["L"],
    )

    n_strings = settings["n_strings"]
    color_mode = settings["color_mode"]
    if color_mode == "rainbow":
        cmap = plt.cm.hsv
        colors = [cmap(i / n_strings) for i in range(n_strings)]
    elif color_mode == "gradient":
        cmap = LinearSegmentedColormap.from_list(
            "custom_gradient", [settings["gradient_start"], settings["gradient_end"]]
        )
        if n_strings > 1:
            colors = [cmap(i / (n_strings - 1)) for i in range(n_strings)]
        else:
            colors = [settings["gradient_start"]]
    else:
        colors = [settings["single_color"]] * n_strings

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.grid(True, color=settings["grid_color"])
    fig.patch.set_facecolor(settings["bg_color"])
    ax.set_facecolor(settings["bg_color"])

    R = settings["n_links"] * settings["L"]
    margin_factor = 0.2
    limit = R * (1 + margin_factor)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    rod_lines = []
    joint_dots = []
    endpoint_dots = []
    history_lines = []
    for i in range(n_strings):
        if settings["n_links"] == 1:
            num_pts = 2
        elif settings["n_links"] == 2:
            num_pts = 3
        else:
            num_pts = settings["n_links"] + 1
        if settings["render_pendulums"]:
            (rod_line,) = ax.plot([], [], lw=2, color=colors[i])
        else:
            rod_line = None
        rod_lines.append(rod_line)
        if settings["render_joints"]:
            (joint_dot,) = ax.plot([], [], "o", color=colors[i], markersize=4)
        else:
            joint_dot = None
        joint_dots.append(joint_dot)
        if settings["render_endpoints"]:
            (endpoint_dot,) = ax.plot([], [], "o", color=colors[i], markersize=6)
        else:
            endpoint_dot = None
        endpoint_dots.append(endpoint_dot)
        if settings["render_history_length"] > 0:
            (history_line,) = ax.plot([], [], "-", color=colors[i], lw=1, alpha=0.5)
        else:
            history_line = None
        history_lines.append(history_line)

    if settings["render_base"]:
        (base_dot,) = ax.plot([], [], "o", color=settings["base_color"], markersize=6)
    else:
        base_dot = None

    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        color=settings["text_color"],
    )

    def animate(frame):
        sim.step()
        positions_list = sim.get_all_positions()
        histories = sim.get_all_histories()
        for i, pos in enumerate(positions_list):
            x = pos[:, 0]
            y = pos[:, 1]
            if rod_lines[i] is not None:
                rod_lines[i].set_data(x, y)
            if joint_dots[i] is not None:
                joint_dots[i].set_data(x, y)
            if endpoint_dots[i] is not None:
                endpoint_dots[i].set_data([x[-1]], [y[-1]])
            if history_lines[i] is not None and len(histories[i]) > 0:
                hx = histories[i][:, 0]
                hy = histories[i][:, 1]
                history_lines[i].set_data(hx, hy)
        if base_dot is not None:
            base_dot.set_data([0], [0])
        sim_time = frame * settings["dt"]
        time_text.set_text(f"Time: {sim_time:.5f} s")
        return (
            rod_lines
            + joint_dots
            + endpoint_dots
            + history_lines
            + ([base_dot] if base_dot is not None else [])
            + [time_text]
        )

    if settings["gif_mode"]:
        n_frames = int(settings["t_stop"] / settings["dt"])
    else:
        n_frames = 10000

    ani = animation.FuncAnimation(
        fig, animate, frames=n_frames, interval=settings["dt"] * 1000, blit=False
    )

    if settings["gif_mode"]:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(
            fps=int(1 / settings["dt"]),
            metadata=dict(artist="Simulation"),
            bitrate=1800,
        )
        ani.save("pendulum_simulation.mp4", writer=writer)
        print("Animation saved to pendulum_simulation.mp4")
    else:
        plt.show()


# -------------------------------------------------------------------
# Settings loader from TOML
# -------------------------------------------------------------------
def load_settings(filename="settings.toml"):
    if not os.path.exists(filename):
        sys.exit(f"Settings file '{filename}' not found!")
    config = load_toml(filename)
    sim_config = config.get("simulation", {})

    defaults = {
        "n_strings": 3,
        "n_links": 2,  # Try 1 or 2 for optimized cases; >2 for general case.
        "degrees_between": 5,
        "chaos_mode": True,
        "render_pendulums": True,
        "render_endpoints": True,
        "render_joints": True,
        "render_base": True,
        "render_history_length": 50,
        "g": 9.81,
        "dt": 0.01,
        "t_stop": 20,
        "gif_mode": False,
        "color_mode": "rainbow",  # Options: "rainbow", "single", "gradient"
        "single_color": "blue",
        "grid_color": "lightgray",
        "L": 1.0,
        "base_color": "black",
        "gradient_start": "blue",
        "gradient_end": "red",
        "bg_color": "white",
        "text_color": "black",
    }
    for key, val in defaults.items():
        sim_config.setdefault(key, val)
    return sim_config


if __name__ == "__main__":
    settings = load_settings("settings.toml")
    run_simulation(settings)
