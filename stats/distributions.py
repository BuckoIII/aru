# script thats codes out from scratch a numer of statistical distributions inclufing
# - binomial
# - hypergeometric
# - poisson


import math as m
import numpy as np
from scipy.stats import binom
from scipy.stats import hypergeom
from scipy.stats import poisson
from scipy.stats import norm


# Binomial Distribution in Python
questions = []

answers = []
# Q1
k, n, p = 4, 5, 0.5
ans1 = binom.pmf(k=k, n=n, p=p)
answers.append(round(ans1, 2))
questions.append('Q' + str(len(questions) + 1))

# Q2
k, n, p = 4, 5, 1 / 3
ans2 = binom.pmf(k=k, n=n, p=p)
answers.append(round(ans2, 3))
questions.append('Q' + str(len(questions) + 1))


# Example of cumulative probabailty
def bi_cumulative_probability_for_multiple_events(n, k, p):
    '''
    input:
    n: trials
    k: successes - list or array
    p: probability of outcome

    output:
    ans: cumulative probability of outcome
    '''
    outputs = []

    for k in inputs:
        ## with math
        outputs.append(m.factorial(n) / (m.factorial(k) * m.factorial(n - k)) * p ** k * (1 - p) ** (n - k))
        # # with scipy
        # outputs.append(binom.pmf(k,n,p))
    ans = np.sum(outputs)

    if ans > 1:
        print('Error: Probabilies sum to greater then 1')
    else:
        return ans


# set vars
# trials & probability
n, p = 7, 0.5
inputs = [5, 6, 7]
ansx = bi_cumulative_probability_for_multiple_events(n, inputs, p)
answers.append(round(ansx, 3))
questions.append('exp')

# Q3
k, n, p = 3, 4, 0.5
inputs = [3, 4]
ans3 = bi_cumulative_probability_for_multiple_events(n, inputs, p)
answers.append(round(ans3, 3))
questions.append('Q' + str(len(questions)))

# Q4
n, p = 10, 0.7
inputs = [10, 9, 8, 7, 6, 5]
ans4 = 1 - bi_cumulative_probability_for_multiple_events(n, inputs, p)
answers.append(round(ans4, 3))
questions.append('Q' + str(len(questions)))

# mean and variance of a binomial distribution

# Q5
# mean = n trials * probability of success
n, p = 20, 0.3
u = n * p
answers.append(u)
questions.append('Q5')

# Q6
# mean = n trials * probability of success
n, p = 15, 0.6
var = n * p * (1 - p)
# var = binom.var(n,p)
answers.append(var)
questions.append('Q6')

# Q7
n, p = 50, 0.5
var = n * p * (1 - p)
std = var ** (1 / 2)
# alternate method
std = binom.std(n, p)

answers.append(round(std, 3))
questions.append('Q7')

print('binomial distribution questions:',dict(zip(questions, answers)))

# hypergeometric distributions
# a hypergeometric distribution is a distribution of outcomes in a binary outcome event where there is no replaceent
# e.g king or not king in a deck of cards

hy_answers = []
hy_questions = []
def hypergeo_dist_single_probability(k, N, n, x, show_workings=False):
    '''
    descr: function to calculate the probability of an outcome occurring which a hypergeometric distribution

    inputs:
    k: number of successes
    N: population size in the trial
    n: number of trials
    x: number of successes you are testing for or that there could be in the population

    ouputs:
    ans: probability of event occuring
    '''

    C_k_n = m.factorial(x) / (m.factorial(k) * m.factorial(x - k))
    C_N_x_n_k = m.factorial(N - x) / (m.factorial(n - k) * m.factorial((N - x) - (n - k)))
    C_N_n = m.factorial(N) / (m.factorial(n) * m.factorial(N - n))

    ans = (C_k_n * C_N_x_n_k) /C_N_n

    if show_workings == True:
        print('k:', k, 'n:', n, 'N:', N, 'x:', x)
        print(f'C_k_n = m.factorial({x}) / (m.factorial({k}) * m.factorial({x - k}))')
        print(f'C_N_x_n_k = m.factorial({N - x}) / (m.factorial({n - k}) * m.factorial({(N - x) - (n - k)}))')
        print(f'C_N_n = m.factorial({N}) / (m.factorial({n}) * m.factorial({N - n}))')

        print('C_k_n:',C_k_n)
        print('C_N_x_n_k:',C_N_x_n_k)
        print('C_N_n:', C_N_n)

    return ans

def hypergeom_dist_multiple_probability(k, N, n, x, show_probs=False, show_workings=False):

    probabilities = []
    for k in range(k+1):
        probabilities.append(hypergeo_dist_single_probability(k=k, N=N, n=n, x=x, show_workings=show_workings))

    if show_probs == True:
        print(probabilities)
    ans = sum(probabilities)

    return ans


# example
N = 52
n = 20
k = 3
x = 4

ans = hypergeom_dist_multiple_probability(k=k, N=N, n=n, x=x)

hy_answers.append(round(ans,3))
hy_questions.append('exp 1')

# Q1
N = 52
n = 7
x = 26
k = 7

# ans using my function
# ans = round(hypergeo_dist_single_probability(k=k, N=N, n=n, x=x), 3)

# ans using scipy hypergeom library
ans = round(hypergeom(M=N, n=x, N=k).pmf(n),3)

# confusingly the scipy version of the function follows a differetn paramter naming convention from my stats book
# - M is the total number of objects. Otherwise, 'N' (population size)
# - n is total number of Type I objects. Otherwise, 'x' (n possible success combinations)
# - N is the random variate represents the number of Type I objects in N drawn without replacement from the...
#   ... total population. Otherwise, 'k' (number of successes looked for)
# - the input to the pmf function is the number of trails ('n' in my function)

hy_answers.append(ans)
hy_questions.append('Q1')

# Q2
N = 62
n = 15
x = 31
k = 11

ans =round(1 - hypergeom_dist_multiple_probability(k=k, N=N, n=n, x=x), 3)

hy_answers.append(ans)
hy_questions.append('Q2')

# Q3
N = 50
n = 10
x = round(50*0.6)
k = 2


ans =hypergeom_dist_multiple_probability(k=k, N=N, n=n, x=x, show_probs=False, show_workings=False)
# using the scipy cdf function
ans = hypergeom.cdf(k=k, M=N, n=x, N=n)


hy_answers.append(round(ans, 3))
hy_questions.append('Q3')


# Example 2
N = 100
x = 30
n = 20

u = (n*x) / N
# use hypergeom to get mean
u = hypergeom.stats(M=N, N=n, n=x)[1]
var = (n*x * (N-x) * (N-n)) / (N**2*(N-1))
# use scipy to get std
# std = hypergeom.std(M=N, N=n, n=x)
std = var**(1/2)

hy_answers.append(round(std, 3))
hy_questions.append('exp 2')

# Q4

N = 75
x = 15
n = 30

u = (n*x) / N
var = hypergeom.var(M=N, N=n, n=x)
ans = var**(1/2)
hy_answers.append(round(ans, 4))
hy_questions.append('Q4')

print('hypergeometric distribution questions:', dict(zip(hy_questions, hy_answers)))


# Poisson distribution

p_questions = []
p_answers = []


def single_poisson_probability(lam:int, k:int):
    '''
    :param:
    k : n trials
    lam : baseilne probability of event occuring
    '''
    p = np.exp(-lam) * (lam**(k) / m.factorial(k))
    return p

k = 15
lam = 10

ans = single_poisson_probability(lam=lam, k=k)

p_answers.append(round(ans, 4))
p_questions.append('exp 1')


# Q1
k = 7
lam = 8

# ans = single_poisson_probability(lam=lam, k=k)
# ans with scipy
ans = poisson.pmf(k=k, mu=lam)

p_answers.append(round(ans, 4))
p_questions.append('Q1')


# Q2
k = 3
lam = 1

ans = single_poisson_probability(lam=lam, k=k)

p_answers.append(round(ans, 4))
p_questions.append('Q2')

# Q3
k = 0
lam = 59/20
ans = single_poisson_probability(lam=lam, k=k)

p_answers.append(round(ans, 4))
p_questions.append('Q3')

# poisson & cumulative probabilities
# Q4
lam = 11
k = 3

def cumulative_probability(lam:int, k:int):
    probs = []
    for k in range(k+1):
        probs.append(single_poisson_probability(lam=lam, k=k))
    ans = 1 - np.sum(probs)

    return ans

ans = cumulative_probability(lam=lam, k=k)

p_answers.append(round(ans, 4))
p_questions.append('Q4')

# Q5
lam = 5
k = 1
ans = 1 - cumulative_probability(lam=lam, k=k)

p_answers.append(round(ans, 4))
p_questions.append('Q5')

# Q6
lam = 5
k = 1
ans = cumulative_probability(lam=lam, k=k)

p_answers.append(round(ans, 4))
p_questions.append('Q6')

print('poisson distribution questions:', dict(zip(p_questions, p_answers)))


# normal distribution

# earnings surprise example
surprise = [11.36,7.89,1.96,0,-3.12,-9.52]
def std(arr:list) -> list:
    print(arr)

print(std(surprise))
# n_questions = []
# n_answers = []
#
#
# mu = np.mean(surprise)
# std = np.std(surprise)
# z = (mu - 11.36) / std
#
# print(mu, std, z)
#
# n_answers.append(round(ans, 4))
# n_questions.append('exp')
#
#
# # calculating z-scores
#
# print('normal distribution questions:', dict(zip(n_questions, n_answers)))
#
