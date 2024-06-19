import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Task 1
P1 = np.random.normal(loc=0, scale=1, size=4) # mean=0, standard deviation=1
sample_mean = np.mean(P1)

# Task 2
alpha = 0.05
p_values = []
sample_means = []
sample_size = 4
mean = 0
st_d = 1
n_trials = 1000
for _ in range(n_trials):
    P1 = np.random.normal(loc=mean, scale=st_d, size=sample_size) 
    sample_mean = np.mean(P1)
    sample_std = np.std(P1, ddof=1)

    t_stat, p_value = stats.ttest_1samp(P1, 0)
    
    p_values.append(p_value)
    sample_means.append(sample_mean)

# Count the number of times we reject the null hypothesis
rejections = sum(1 for p in p_values if p < alpha)

# Plot the histogram of p-values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(p_values, bins=20, edgecolor='black')
plt.title('Histogram of p-values')
plt.xlabel('p-value')
plt.ylabel('Frequency')

# Plot the histogram of sample means
plt.subplot(1, 2, 2)
plt.hist(sample_means, bins=20, edgecolor='black', density=True)
plt.title('Histogram of Sample Means')

# Overlay the theoretical normal distribution
x = np.linspace(min(sample_means), max(sample_means), 100)
y = stats.norm.pdf(x, 0, 1 / np.sqrt(sample_size))
plt.plot(x, y, label='Theoretical Normal Distribution', linewidth=2)
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Number of times null hypothesis was rejected: {rejections}")

# Task 3
sample_size = 100 # do it again

# Task 4
alpha = 0.05
p_values = []
sample_means = []
sample_size = 4
mean = 1
st_d = 1
n_trials = 1000
for _ in range(n_trials):
    P1 = np.random.normal(loc=mean, scale=st_d, size=sample_size) 
    sample_mean = np.mean(P1)
    sample_std = np.std(P1, ddof=1)

    t_stat, p_value = stats.ttest_1samp(P1, 0)
    
    p_values.append(p_value)
    sample_means.append(sample_mean)

# Count the number of times we reject the null hypothesis
rejections = sum(1 for p in p_values if p < alpha)


    
    