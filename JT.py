import matplotlib.pyplot as plt

# Data
joining_temperature = [486.05, 396.75, 307.45]  # Joining temperature [C]
deposition_width = [24.1, 27.8441, 29.4918]  # Deposition width [mm]
deposition_thickness = [2.51, 2.8403, 2.96997]  # Deposition thickness [mm]

# Create figure and axis objects
fig, ax1 = plt.subplots()

# Plotting the Deposition width vs. Joining temperature
ax1.plot(joining_temperature, deposition_width, 'r--', label='Deposition Width [mm]', marker='s')
ax1.set_xlabel('Joining temperature [°C]')


ax1.set_ylabel('Deposition width [mm]', color='r')
ax1.set_ylim(23, 31)
ax1.set_xlim(300, 500)
ax1.tick_params(axis='y', labelcolor='r')

# Create a second y-axis for the Deposition thickness
ax2 = ax1.twinx()
ax2.plot(joining_temperature, deposition_thickness, 'b', label='Deposition Thickness [mm]', marker='o', linestyle="dotted")
ax2.set_ylabel('Deposition thickness [mm]', color='b')
ax2.set_ylim(2.3, 3.1)
ax2.tick_params(axis='y', labelcolor='b')

# Enable grid
ax1.grid(True)

# Show plot
#plt.title('Deposition Width and Thickness vs. Joining Temperature')
plt.savefig("ps_jt_width_thickness.pdf")
plt.show()
