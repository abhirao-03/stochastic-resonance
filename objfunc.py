from scipy import stats
import numpy.random as random

def run(jump_mult):
    noise = random.normal(loc=0.0,
                          scale=dt**(1/2),
                          size=(num_trajectories, num_steps))

    jump_times = PJ(jump_mult, noise)

    x_transformed = exp_cdf(jump_times, rate)

    met = stats.cramervonmises(x_transformed, 'uniform').pvalue

    return met