import matplotlib.pyplot as plt
import numpy as np

grid2_data = np.array([0.28369159300113095,
                0.2639044770004693,
                0.24453423599887172,
                0.2590059069998097,
                0.25694649399782066,
                0.2478399960009847,
                0.25095022000023165,
                0.24968809200043324
                ])

hallway_data = np.array([6.386896010005148,
5.699117479997222,
6.34049034500058,
5.7377180799972844,
5.9598324000020515,
6.271642159999464,
5.575040295002691
])

if __name__ == "__main__":
    plt.figure(figsize=(8, 6))

    # Plot horizontal line at y = 1
    plt.axhline(y=1, color='gray', linestyle='--', label='A*')

    grid2_speedups = grid2_data[1:] / grid2_data[0]
    hallway_speedups = hallway_data[1:] / hallway_data[0]

    # Plot speedups
    plt.plot(range(50, 50*len(grid2_data), 50), grid2_speedups, marker='o', label='Grid2 Scenario')
    plt.plot(range(50, 50*len(hallway_data), 50), hallway_speedups, marker='o', label='Hallway '
                                                                                  'Scenario')
    plt.xlabel("Number of Planned Paths in Social Graph", fontsize=16)
    plt.ylabel("Relative Path Planning Time", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=12, loc='upper right', )
    plt.title("Computation Time for Social Graphs", fontsize=16)
    plt.tight_layout()
    plt.savefig("benchmarks/benchmark_plot.png", dpi=600)
    plt.show()

