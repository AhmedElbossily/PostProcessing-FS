import matplotlib.pyplot as plt

# Data
Simulation = ["1st", "2nd", "3rd"]
#NumberOfDepositedParticles = [4220, 4030, 3811]
#NumberOfDepositedParticles = [4109, 3955, 4114]
NumberOfDepositedParticles = [4351, 3997, 3965]

# Plotting the data
plt.bar(Simulation, NumberOfDepositedParticles, color='blue')

# Adding labels and title
plt.xlabel('Simulation')
plt.ylabel('Number of Deposited Particles')
plt.ylim((3500))

# Display the plot
plt.show()