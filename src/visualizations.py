import matplotlib.pyplot as plt


def plot_noiseness(actions):
    plt.figure(figsize=(12, 8))

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(actions[:100, i])
        plt.title(f"Action Dimension {i}")
        plt.ylabel("Value")
        plt.xlabel("Timestep")

    plt.tight_layout()
    plt.show()
