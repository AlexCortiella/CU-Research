#Duffing oscillator
def duffing(x, t, gamma=0.1, kappa=1, epsilon=5):
    return [
            x[1],
            -gamma * x[1] - kappa * x[0] - epsilon * x[0] ** 3
            ]

gamma=0.1
kappa=1
epsilon=5


#Van der Pol oscillator
def vanderpol(x, t, gamma=1, kappa=1, epsilon=2):
    return [
            x[1],
            -kappa * x[0] - gamma * x[1] - epsilon * x[1] * x[0] ** 2
            ]

gamma=1
kappa=1
epsilon=2

#Lorenz 63
def lorenz63(x, t, sigma=10, rho=28, beta=8/3):
    return [
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2]
    ]

sig=10
rho=28
beta=8/3