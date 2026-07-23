import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#------Constants------------------------------------------------
G = 1
dt = 0.001
steps = 50000
softening = 0.01

#------Initial Conditions---------------------------------------
masses = np.array([1.0, 1.0, 1.0])
position = np.array([
    [-0.97000436,  0.24308753, 0.0],
    [ 0.97000436, -0.24308753, 0.0],
    [ 0.0,         0.0,        0.0],
])
velocity = np.array([
    [ 0.93240737/2,  0.86473146/2, 0.0],
    [ 0.93240737/2,  0.86473146/2, 0.0],
    [-0.93240737,   -0.86473146,   0.0],
])
#------physics part-----------------------------------------------
def compute_acceleration(position, masses):
    acc = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i != j:
                r_vec = position[j] - position[i]
                r_mag = np.sqrt(np.sum(r_vec**2) + softening**2)
                acc[i] += G * masses[j] * r_vec / r_mag**3
    return acc

def simulate(position, velocity,  masses, dt, steps):
    trajectory = np.zeros((steps, 3, 3))
    acc = compute_acceleration(position, masses)

    for step in range(steps):
        velocity += 0.5*acc*dt
        position += velocity*dt
        acc = compute_acceleration(position, masses)
        velocity += 0.5*acc*dt
        trajectory[step] = position
    
    return trajectory

#------------animation------------------------------------------------------------------------
def animate(trajectory):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')

    colors = ['cyan', 'magenta', 'yellow']
    
    # dots for current position
    dots = [ax.plot([], [], 'o', color=c, markersize=6)[0] for c in colors]
    # lines for trails
    trails = [ax.plot([], [], '-', color=c, alpha=0.3, linewidth=0.8)[0] for c in colors]

    trail_length = 500  # how many past positions to show

    def update(frame):
        for i in range(3):
            # current dot
            dots[i].set_data([trajectory[frame, i, 0]], [trajectory[frame, i, 1]])
            # trail
            start = max(0, frame - trail_length)
            trails[i].set_data(trajectory[start:frame, i, 0], trajectory[start:frame, i, 1])
        return dots + trails

    ani = animation.FuncAnimation(
        fig, update,
        frames=range(0, steps, 10),  # skipping every 10 frames to speed up
        interval=1,                  # time in ms between frames
        blit=True
    )

    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------------------------
trajectory = simulate(position, velocity, masses, dt, steps)
animate(trajectory)