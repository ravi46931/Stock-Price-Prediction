import matplotlib.pyplot as plt

# Create Figure 1
plt.figure()  # Create a new figure
plt.plot([1, 2, 3, 4])
plt.title("Figure 1")

# Save Figure 1
plt.savefig("figure1.png")

# Create Figure 2
plt.figure()  # Create another new figure
plt.plot([4, 3, 2, 1])
plt.title("Figure 2")

# Save Figure 2
plt.savefig("figure2.png")
