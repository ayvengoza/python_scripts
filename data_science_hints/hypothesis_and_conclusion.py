import math
from probability import normal_cdf, inverse_normal_cdf
import random

def normal_approximation_to_binomial(n, p):
    mu = n * p
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

normal_probability_below = normal_cdf

def normal_probability_above(lo, mu = 0, sigma = 1):
    return (1 - normal_cdf(lo, mu, sigma))

def normal_probabilitiy_between(lo, hi, mu = 0, sigma = 1):
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

def norma_probability_outside(lo, hi, mu = 0, sigma = 1):
    return 1 - normal_probabilitiy_between(lo, hi, mu, sigma)

def normal_upper_bound(probability, mu = 0, sigma = 1):
    """find value that corresponds left part of probability (look normal_cdf)
        <start><left_probability><value><right_probability><end>"""
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability, mu = 0, sigma = 1):
    """find value that corresponds right part of probability (look normal_cdf)
        <start><left_probability><value><right_probability><end>"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability, mu = 0, sigma = 1):
    tail_probability = (1 - probability) / 2

    upper_bound =  normal_lower_bound(tail_probability, mu, sigma)
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound

def coin_hypothesis():
    print("Hypothesis that eagle heppends in 1/2 cases")
    # H0 hypothesis
    # type_1_probability = 0.05
    mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
    low_0, hi_0 = normal_two_sided_bounds(0.95, mu_0, sigma_0)
    print("low_0={}, hi_0={}".format(low_0, hi_0))

    #H1 hypothesis
    mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)
    type_2_probability = normal_probabilitiy_between(low_0, hi_0, mu_1, sigma_1)
    power = 1 - type_2_probability
    print("Power of hypothesis(+0.05 check) {}%".format(power * 100))

def coin_eagle_nomore_then_half_hypothesis():
    print("Hypothesis that eagle heppends no more than in 1/2 cases")
    # H0 hypothesis
    mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
    hi = normal_upper_bound(0.95, mu_0, sigma_0)
    print("Eagle hi bound ", hi)

    #H1 hypothesis
    mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)
    type_2_probability = normal_probability_below(hi, mu_1, sigma_1)
    power = 1 - type_2_probability
    print("Power of hypothesis(+0.05 check) {}%".format(power * 100))

# P value approach

def two_sided_p_value(x, mu=0, sigma=1):
    if x >= mu:
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        return 2 * normal_probability_below(x, mu, sigma)

# Confidence Intervals

def confience_interval_approach(n, N):
    """ N - amount of iteration, n - amount of success """
    p_hat = n / N
    mu = p_hat
    sigma = math.sqrt(p_hat * (1 - p_hat) / N)
    return normal_two_sided_bounds(0.95, mu, sigma)

# Fit P value

def run_experiment():
    """ Throw coin 1000 times 
    True = head, False = tail"""
    return [random.random() < 0.5 for _ in range(1000)]

def reject_fairness(experiment):
    """ use 5% level of value """
    num_heads = len([flip for flip in experiment if flip])
    return num_heads < 469 or num_heads > 531

def test_fit_p_value():
    random.seed(0)
    experiments = [run_experiment() for _ in range(1000)]
    num_rejections = len([experiment
                            for experiment in experiments
                            if reject_fairness(experiment)])
    print("Num rejections:", num_rejections)

# A/B Testing

def estimate_paramenters(N, n):
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma


def a_b_test_statistic(N_A, n_A, N_B, n_B):
    """ H0 -> p_A == p_B """
    p_A, sigma_A = estimate_paramenters(N_A, n_A)
    p_B, sigma_B = estimate_paramenters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)


def a_b_test_run():
    z = a_b_test_statistic(1000, 200, 1000, 150)
    print("A/B Testing, A(200/1000), B(150/1000)")
    print("Probability of A and B is same:", two_sided_p_value(z))

# Bayes static conclusion

def B(alpfa, beta):
    return math.gamma(alpfa) * math.gamma(beta) / math.gamma(alpfa + beta)
    
def beta_pdf(x, alpfa, beta):
    if x < 0 or x > 1: return 0
    return x ** (alpfa - 1) * (1 - x) ** (beta -1) / B(alpfa, beta)


if __name__ == "__main__":
    coin_hypothesis()
    coin_eagle_nomore_then_half_hypothesis()
    print("Confidence Interval of 525 from 1000:", confience_interval_approach(525, 1000))
    test_fit_p_value()
    a_b_test_run()
