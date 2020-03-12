from ortools.linear_solver import pywraplp
import numpy as np


def lp_three(constraints,alpha,d):
    solver = pywraplp.Solver('LinearProgrammingExample',
                             pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    # constraints are [a,b,c] corresponding to x,y,z peak link demands

    x = solver.NumVar(0, solver.infinity(), 'x')
    y = solver.NumVar(0, solver.infinity(), 'y')
    z = solver.NumVar(0, solver.infinity(), 'z')

    # Constraint 0: x + y >= max(a,b)
    constraint0 = solver.Constraint(float(np.maximum(constraints[0],constraints[1])), solver.infinity())
    constraint0.SetCoefficient(x, 1)
    constraint0.SetCoefficient(y, 1)

    # Constraint 1: y + z >= max(b,c)
    constraint1 = solver.Constraint(float(np.maximum(constraints[1],constraints[2])), solver.infinity())
    constraint1.SetCoefficient(y, 1)
    constraint1.SetCoefficient(z, 1)

    # Constraint 2: x + z >= max(a,c)
    constraint2 = solver.Constraint(float(np.maximum(constraints[0],constraints[2])), solver.infinity())
    constraint2.SetCoefficient(x, 1)
    constraint2.SetCoefficient(z, 1)

    # Objective function: sum (i<j) C_ij * (1 + alpha d_ij)
    objective = solver.Objective()
    objective.SetCoefficient(x, 1+alpha*d["x"])
    objective.SetCoefficient(y, 1+alpha*d["y"])
    objective.SetCoefficient(z, 1+alpha*d["z"])
    objective.SetMinimization()

    # Solve the system.
    solver.Solve()
    # The value of each variable in the solution.
    return x.solution_value(), y.solution_value(), z.solution_value()

def lp_four(alpha,d):
    solver = pywraplp.Solver('LinearProgrammingExample',
                             pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    n_vars = 6
    ab = solver.NumVar(0, solver.infinity(), 'AB') # 4 nodes, all connected
    ac = solver.NumVar(0, solver.infinity(), 'AC') 
    ad = solver.NumVar(0, solver.infinity(), 'AD')
    bc = solver.NumVar(0, solver.infinity(), 'BC')
    bd = solver.NumVar(0, solver.infinity(), 'BD')
    cd = solver.NumVar(0, solver.infinity(), 'CD')

    # Constraint 0: AC + AB + CD >= 7
    constraint0 = solver.Constraint(7, solver.infinity())
    constraint0.SetCoefficient(ac, 1)
    constraint0.SetCoefficient(ab, 1)
    constraint0.SetCoefficient(cd, 1)

    # Constraint 1: AC + AB + AD >= 7
    constraint1 = solver.Constraint(7, solver.infinity())
    constraint1.SetCoefficient(ac, 1)
    constraint1.SetCoefficient(ab, 1)
    constraint1.SetCoefficient(ad, 1)

    # Constraint 2: BC + AC + CD >= 13
    constraint2 = solver.Constraint(13, solver.infinity())
    constraint2.SetCoefficient(ac, 1)
    constraint2.SetCoefficient(bc, 1)
    constraint2.SetCoefficient(cd, 1)

    # Constraint 3: BC + AC + BD >= 13
    constraint3 = solver.Constraint(13, solver.infinity())
    constraint3.SetCoefficient(bc, 1)
    constraint3.SetCoefficient(ac, 1)
    constraint3.SetCoefficient(bd, 1)

    # Constraint 4: BC + AB + CD >= 13
    constraint4 = solver.Constraint(13, solver.infinity())
    constraint4.SetCoefficient(bc, 1)
    constraint4.SetCoefficient(ab, 1)
    constraint4.SetCoefficient(cd, 1)

    # Constraint 5: BC + AB + BD >= 13
    constraint5 = solver.Constraint(13, solver.infinity())
    constraint5.SetCoefficient(bc, 1)
    constraint5.SetCoefficient(ab, 1)
    constraint5.SetCoefficient(bd, 1)

    # Constraint 6: AB + AD + BC >= 5
    constraint6 = solver.Constraint(5, solver.infinity())
    constraint6.SetCoefficient(ab, 1)
    constraint6.SetCoefficient(ad, 1)
    constraint6.SetCoefficient(bc, 1)

    # Constraint 7: AB + BD + AC >= 5
    constraint7 = solver.Constraint(5, solver.infinity())
    constraint7.SetCoefficient(ab, 1)
    constraint7.SetCoefficient(bd, 1)
    constraint7.SetCoefficient(ac, 1)

    # Constraint 8: BD + CD + AB >= 5
    constraint8 = solver.Constraint(5, solver.infinity())
    constraint8.SetCoefficient(bd, 1)
    constraint8.SetCoefficient(cd, 1)
    constraint8.SetCoefficient(ab, 1)

    # Constraint 9: BD + BC +AD >= 5
    constraint9 = solver.Constraint(5, solver.infinity())
    constraint9.SetCoefficient(bd, 1)
    constraint9.SetCoefficient(bc, 1)
    constraint9.SetCoefficient(ad, 1)

    # Constraint 10: BD + CD + AD >= 5
    constraint10 = solver.Constraint(5, solver.infinity())
    constraint10.SetCoefficient(bd, 1)
    constraint10.SetCoefficient(cd, 1)
    constraint10.SetCoefficient(ad, 1)

    # Objective function: sum (i<j) C_ij * (1 + alpha d_ij)
    objective = solver.Objective()
    objective.SetCoefficient(ab, 1+alpha*d["AB"])
    objective.SetCoefficient(ac, 1+alpha*d["AC"])
    objective.SetCoefficient(ad, 1+alpha*d["AD"])
    objective.SetCoefficient(bc, 1+alpha*d["BC"])
    objective.SetCoefficient(bd, 1+alpha*d["BD"])
    objective.SetCoefficient(cd, 1+alpha*d["CD"])
    objective.SetMinimization()

    # Solve the system.
    solver.Solve()
    # The value of each variable in the solution.
    return ab.solution_value(), ac.solution_value(), ad.solution_value(), bc.solution_value(), bd.solution_value(), cd.solution_value()


interesting = [[.689,.463,.190,.538,.391,.106]]
# traffic matrix
nodes = ["A","B","C","D"]
T = [[None,5,7,3],[5,None,13,5],[7,13,None,4],[3,5,4,None]]
T_dict = {}
for i,n in enumerate(nodes):
    for j,n_ in enumerate(nodes):
        if i >= j:
            continue
        T_dict[n+n_] = T[i][j]
keys = ["AB","AC","AD","BC","BD","CD"]
distances = interesting[0]#np.random.uniform(size=(len(keys)))
d = {k:v for k,v in zip(keys,distances)}
print("{} : {}".format(keys, distances))
x_arr = np.linspace(0,100,num=100)
import matplotlib.pyplot as plt


# Find optimal solution at various alphas
tups = []
for alpha in x_arr:
    ret = lp_four(alpha, d)
    tups.append(ret)


[plt.plot(x_arr, [el[i] + np.random.normal(scale=.1) for el in tups ],label=lab) for i,lab in enumerate(keys)]
seen = []
for sol in tups:
    if sol in seen:
        continue
    seen.append(sol)
    [print("{}: {}".format(k,v)) for k,v in zip(keys,sol)]

plt.legend()
plt.xlabel("Alpha"); plt.ylabel("Capacity")
plt.title("Optimal Capacities as a Function of Distance Cost")
plt.savefig("optimal_capacities.pdf")

# Calculating Shapley values
# these are in "letter order"
sub_coalitions = [["AB", "AC", "BC"], ["AC", "AD", "CD"], ["AB","AD","BD"], ["BC","BD","CD"]]
constraints =  [[T_dict[k] for k in sub_coalition] for sub_coalition in sub_coalitions]
shapleys = {k:np.zeros(len(x_arr)) for k in nodes}
values = {}
for i, alpha in enumerate(x_arr):
    for sub_coalition, constraint in zip(sub_coalitions, constraints):
        cost_without_coalition = sum([T_dict[k] * (1 + alpha*d[k]) for k in sub_coalition])
        coalition_distances = {l:d[k] for l,k in zip(["x","y","z"],sub_coalition)}
        optimal_coalition_capacities = lp_three(constraint, alpha, coalition_distances)
        cost_of_coalition = sum([el * (1 + alpha * d[k]) for el,k in zip(optimal_coalition_capacities,sub_coalition)])
        # print("Sub coalition: {} cost without: {} cost with: {} value: {}".format(
        #     sub_coalition,cost_without_coalition,cost_of_coalition,cost_without_coalition-cost_of_coalition))
        value = cost_without_coalition - cost_of_coalition
        nodes_of_interest = set("".join(sub_coalition))
        values[tuple(sorted(nodes_of_interest))] = value # save for later
        for node in nodes_of_interest:
            shapleys[node][i] += (value / 12)
  #  print(values)
    # now calculate the terms when adding each of the nodes, to form the full coalition
    cost_without_coalition = sum([T_dict[k] * (1+alpha*d[k]) for k in keys])
    optimal_coalition_capacities = lp_four(alpha,d)
    cost_with_coalition = sum([el * (1 + alpha * d[k]) for el,k in zip(optimal_coalition_capacities, keys)])
    for node in nodes:
        # calculate marginal value when adding this node
        marginal_value = cost_without_coalition - cost_with_coalition - values[tuple(sorted(set(nodes) - set(node)))]
        shapleys[node][i] += marginal_value / 4
        # print("Node {} cost_without: {} cost with: {} marginal value: {}".format(
        #     node,cost_without_coalition, cost_with_coalition,marginal_value))

for node in nodes:
    plt.plot(np.linspace(.1,1,100), shapleys[node], label="Node {}".format(node))
plt.xlabel("Alpha"); plt.ylabel("Shapley Value")
plt.title("Shapley Values as a Function of Distance Cost")
plt.legend()
plt.savefig("shapley_values.pdf")