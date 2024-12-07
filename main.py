from CO import cheetah_optimizer
from fitness_funcitons import fitness_function

POPULATION_SIZE = [10, 10, 10, 10, 22, 10, 40, 10, 40, 30, 40, 30, 30, 50, 100, 50, 50, 50, 100, 70, 300, 200, 200]

for i in range(1, 24):
    print("-"*28, f"Function #{i}", "-"*28)
    f_min, it = cheetah_optimizer(POPULATION_SIZE[i-1], *fitness_function(f'F{i}'))
    print("-"*10, f"F#{i} min: {f_min}", f"|| iterations: {it}", "-"*10)
    print()

