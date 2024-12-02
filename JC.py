import csv
import glob
import matplotlib.pyplot as plt
import numpy as np

# Johnson-Cook model function
def johnson_cook_stress(A, B, C, n, m, D, strain, strain_rate, temperature):
    Tref = 293.0
    Tmelt = 864.0
    t=(temperature - Tref)/(Tmelt - Tref)
    term1 = A + B * (strain ** n)
    term2 = 1 + C * np.log(strain_rate/10.e-4)
    term3 = 1 - (t ** m)
    term4 = 1 + D * np.log(strain_rate)
    return term1 * term2 * term3 * term4 

# Material constants (example values for aluminum alloy)
A = 1.67e8  # in MPa
B = 0  # in MPa
C = 0.001
n = 0
m = 0.859
D = 0

# Strain values
strain = np.linspace(0, 0.1, 10)

# Strain rates (example values)
strain_rates = [50]  # in s^-1
# Temperatures (example values)
temperatures = [293,0.85*864,864]  

# Plot stress-strain curves for different strain rates and temperatures
for strain_rate in strain_rates:
    for temperature in temperatures:
        stress = johnson_cook_stress(A, B, C, n, m, D, strain, strain_rate, temperature)
        label =  f'JC: Strain Rate: {strain_rate} s^-1, Temperature: {temperature} C'
        plt.plot(strain, stress, label=label)

plt.xlabel('Strain')
plt.ylabel('Stress (MPa)')
plt.title('Johnson-Cook Stress-Strain Curve')
plt.legend()
plt.grid(True)
plt.show()
