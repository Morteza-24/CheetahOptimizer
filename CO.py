import numpy as np


def cheetah_optimizer(n, lb, ub, dim, fintess_function):
    if len(lb) == 1:
        ub = np.full(dim, ub)  # Lower Bound of Decision Variables
        lb = np.full(dim, lb)  # Upper Bound of Decision Variables

    m = 1+np.random.randint(n//2)

    # Generate initial population of cheetahs
    best_solution = {'cost': np.inf}
    population = [{} for _ in range(n)]
    for i in range(n):
        population[i]['position'] = lb+(np.random.rand(dim)*(ub-lb))
        population[i]['cost'] = fintess_function(population[i]['position'])
        if population[i]['cost'] < best_solution['cost']:
            best_solution = population[i].copy()  # Initial leader position

    population_1 = population.copy()    # Populations' initial home position
    best_cost = []                      # Leader fittnes value in a current hunting period
    X_best = best_solution.copy()       # Prey solution sofar
    glo_best = best_cost.copy()         # Prey fittnes value sofar

    t = 0                       # Hunting time counter
    it = 1                      # Iteration counter
    max_it = dim * 10000        # Maximum number of iterations
    T = np.ceil(dim / 10) * 60  # Hunting time
    FEs = 0                     # Counter for function evaluations

    while FEs <= max_it:
        # select m random members of cheetahs
        i0 = np.random.randint(n, size=m)

        for k in range(m):
            i = i0[k]

            # agent selection
            if k == len(i0)-1:
                agent = i0[k-1]
            else:
                agent = i0[k+1]

            # The current position of i-th cheetah
            X = population[i]['position']
            X1 = population[agent]['position']  # The neighbor position
            Xb = best_solution['position']  # The leader position
            Xbest = X_best['position']  # The prey position

            kk = 0
            if i <= 2 and t > 2 and t > int(np.ceil(0.2 * T) + 1) and abs(best_cost[t-2] - best_cost[t-int(np.ceil(0.2 * T) + 1)]) <= 0.0001 * glo_best[t-1]:
                X = X_best['position']
                kk = 0
            elif i == 3:
                X = best_solution['position']
                kk = -0.1 * np.random.rand() * t / T
            else:
                kk = 0.25

            if it % 100 == 0 or it == 1:
                xd = np.random.permutation(X.size)
            Z = X.copy()

            for j in xd:  # select arbitrary set of arrangements
                r_hat = np.random.randn()  # Randomization parameter, Equation (1)
                r1 = np.random.rand()
                # The leader's step length (it is assumed that k==0 is associated with the leader number)
                if k == 0:
                    alpha = 0.0001 * t / T * (ub[j] - lb[j])  # Step length
                else:  # The member's step length
                    alpha = 0.0001 * t / T * abs(Xb[j] - X[j]) + 0.001 * np.round(
                        float(np.random.rand() > 0.9))  # Member step length

                r = np.random.randn()
                # Turning factor, Equation (3) (This can be updated by desired equation)
                r_check = abs(r) ** np.exp(r / 2) * np.sin(2 * np.pi * r)
                beta = X1[j] - X[j]  # Interaction factor

                h0 = np.exp(2 - 2 * t / T)

                H = abs(2 * r1 * h0 - h0)

                r2 = np.random.rand()
                r3 = kk + np.random.rand()

                # Strategy selection mechanism
                if r2 <= r3:
                    r4 = 3 * np.random.rand()
                    if H > r4:
                        Z[j] = X[j] + r_hat ** -1 * alpha  # Search
                    else:
                        Z[j] = Xbest[j] + r_check * beta  # Attack
                else:
                    Z[j] = X[j]  # Sit and wait

            # Update the solutions of member i
            # Check the limits
            xx1 = np.where(Z < lb)
            Z[xx1] = lb[xx1] + \
                np.random.rand(1, np.size(xx1)) * (ub[xx1] - lb[xx1])
            xx1 = np.where(Z > ub)
            Z[xx1] = lb[xx1] + \
                np.random.rand(1, np.size(xx1)) * (ub[xx1] - lb[xx1])
            # Evaluate the new position
            new_sol = {'position': Z}
            new_sol['cost'] = fintess_function(new_sol['position'])
            if new_sol['cost'] < population[i]['cost']:
                population[i] = new_sol
                if population[i]['cost'] < best_solution['cost']:
                    best_solution = population[i]
            FEs += 1

        t += 1

        # Leave the prey and go back home
        if t > T and t - round(T) - 1 >= 1 and t > 2:
            if abs(best_cost[-1] - best_cost[t - round(T) - 1]) <= abs(0.01 * best_cost[-1]):
                # Change the leader position
                best = X_best['position'].copy()
                j0 = np.random.randint(dim, size=(1, np.ceil(
                    dim / 10 * np.random.rand()).astype(int)))
                best[j0] = lb[j0] + \
                    np.random.rand(1, np.size(j0)) * (ub[j0] - lb[j0])
                best_solution['cost'] = fintess_function(best)
                best_solution['position'] = best  # Leader's new position
                FEs += 1

                i0 = np.random.randint(
                    n-1, size=(1, np.round(1 * n).astype(int)))
                # Go back home
                for _ in range(m):
                    # Some members back to their initial positions
                    population[i0[0][n-1-_]] = population_1[i0[0][_]].copy()

                # Substitute the member i by the prey
                population[i] = best_solution

                t = 1  # Reset the hunting time

        it += 1

        # Update the prey (global best) position
        if best_solution['cost'] < X_best['cost']:
            X_best = best_solution
        best_cost.append(best_solution['cost'])
        glo_best.append(X_best['cost'])

        # Display
        if it % 500 == 0:
            print('FEs >>', FEs, '|| BestCost =', glo_best[-1])

    return glo_best[-1], it

