import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


def draw (G):
    pos = nx.kamada_kawai_layout(G)
    nodelist = G.nodes()
    edgelist = G.edges()
    labellist = nx.get_node_attributes(G,'type')
    edge_labels = nx.get_edge_attributes(G,'angle')
    plt.figure(figsize=(12,8))
    nx.draw_networkx_nodes(G,pos,
                        nodelist=nodelist,
                        node_size=1000,
                        node_color='black')
    nx.draw_networkx_edges(G,pos,
                        edgelist = edgelist,
                        edge_color='blue')
    nx.draw_networkx_labels(G, pos=pos,labels=labellist,
                            font_color='white',
                            font_size=6)
    nx.draw_networkx_edge_labels(G, pos, edge_labels,font_size=6)
    plt.show()
    
def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results


def function (G):
    result = []
    ## detect rooms with length K
    rooms = []
    cycles = sorted(nx.minimum_cycle_basis(G))
    for c in cycles:
        edges = []
        if len(c) >= 4 and len(c)<= 6: # consider a room with K=[4,6] walls 
            for e in G.edges(c):
                if e[0] in c and e[1] in c:
                    edges.append(e)
                g = nx.Graph()
                g.add_edges_from(edges)
            rooms.append(list(nx.find_cycle(g)))
    #print("Rooms: ", rooms)

    ## obtain [angle, connection type] for all edges in a room
    angle_conn_list = []

    for r in rooms:
        room=[]
        g= nx.Graph()
        g.add_edges_from(r)
        for e in list(nx.find_cycle(g)):  
            room.append((G[e[0]][e[1]]["angle"],G[e[0]][e[1]]["connection"]))
        angle_conn_list.append(room)
    #print(angle_conn_list)


    ## find illegal list (assign a large cost)
    for i in range(len(angle_conn_list)):
        if angle_conn_list[i] == [(90, 1), (90, 1), (90, 1), (90, 2)] or angle_conn_list[i] == [(90, 2), (90, 1), (90, 1), (90, 1)] or angle_conn_list[i] == [(90, 1), (90, 1),(0, 2), (90, 1), (90, 1)]:
            result.append(100000)
            result.append(100000)
            return result
        i = i + 1
    
    print(angle_conn_list)

    ## detect module (including type 1 & 2 & 3)

    type_1 = [(90,1),(90,1)]
    type_2 = [(90,1),(90,1),(90,1)]
    type_3 = [(90,1),(90,1),(90,1),(90,1)]
    type_4 = [(90,1),(90,1),(0,1),(90,1),(90,1)]

    modules = []
    i = 0
    for l in angle_conn_list:
        findType4 = find_sub_list(type_4,l)
        if findType4 != []:
            for sublist in findType4:
                module = []
                for e in range(sublist[0],sublist[1]+1):
                    module.append(rooms[i][e])
                modules.append(module)
                del l[sublist[0]:sublist[1]]

        findType3 = find_sub_list(type_3,l)
        if findType3 != []:
            for sublist in findType3:
                module = []
                for e in range(sublist[0],sublist[1]+1):
                    module.append(rooms[i][e])
                modules.append(module)
                del l[sublist[0]:sublist[1]]
        
        findType2 = find_sub_list(type_2,l)
        if findType2 != []:
            for sublist in findType2:
                module = []
                for e in range(sublist[0],sublist[1]+1):
                    module.append(rooms[i][e])
                modules.append(module)
                del l[sublist[0]:sublist[1]]
        
        findType1 = find_sub_list(type_1,l)
        if findType1 != []:
            for sublist in findType1:
                module = []
                for e in range(sublist[0],sublist[1]+1):
                    module.append(rooms[i][e])
                modules.append(module)
                del l[sublist[0]:sublist[1]]
        
        i= i + 1
                    
    print("Modules: ", modules)

    Module_wall = []
    for l in modules:
        Module_wall.append(set().union(*l))

    print("Module Wall Set: ", Module_wall)

    ## duplication check

    Panel_wall_pre = set(G.nodes())
    for w in set().union(*Module_wall):
        Panel_wall_pre.remove(w)
        
    print("Panel_wall Set (pre): ", Panel_wall_pre)

    ## merge walls
    H = G.copy()

    nodes = list(Panel_wall_pre)
    g = H.subgraph(nodes)
    while i < g.number_of_edges():
        for e in g.edges():
            if g[e[0]][e[1]]["angle"] == 0 and g.nodes[e[0]]["length"] + g.nodes[e[1]]["length"] <= 13000:
                g.nodes[e[0]]["length"] = g.nodes[e[0]]["length"] + g.nodes[e[1]]["length"]
                g = nx.contracted_nodes(g, e[0],e[1],self_loops=False)
                H = nx.contracted_nodes(H, e[0],e[1],self_loops=False)
                print("merged nodes: ",e[0], e[1])
                break
        i = i + 1

    print("Panel Wall Set (pro): ", g.nodes())
    Panel_wall = g.nodes()

    #____________________________________________
    # objective functions

    Panel2Panel = []
    Panel2Module = []
    Module2Module = []
    Panel2Panel = g.edges()
    for e in H.edges():
        if e[0] in list(Panel_wall) and e[1] in set().union(*Module_wall):
            Panel2Module.append(e)
        elif e[1] in list(Panel_wall) and e[0] in set().union(*Module_wall):
            Panel2Module.append(e)
        elif e[0] in set().union(*Module_wall) and e[1] in set().union(*Module_wall):
            ModuleConnection = True
            for i in range(len(Module_wall)):
                if e[0] in Module_wall[i] and e[1] in Module_wall[i]:
                    ModuleConnection = False
                    break
            if ModuleConnection:
                Module2Module.append(e)
    

    
    print("All connections: ", H.edges())
    print("Panel2Panel Connection: ", Panel2Panel)
    print("Panel2Module Connection: ", Panel2Module)
    print("Module2Module Connection: ", Module2Module)

    ## Time rough estimates
    # Connection capacity
    p2p = 2
    p2m = 2
    m2m = 1
    # Production capacity  
    panel_prod = 2
    module_prod = 0.5
    # Finishing capacity
    module_finish = 1
    panel_finish = 2
    # Lifting capacity
    panel_lift = 0.6
    module_lift = 1

    Time_prod = len(Panel_wall) / panel_prod + len(modules) / module_prod
    print("Time_prod", Time_prod)
    Time_finish = len(modules) * module_finish + (len(rooms)-len(modules))*panel_finish
    print("Time_finish", Time_finish)
    Time_assembly = len(Panel_wall) * panel_lift + len(modules) * module_lift + len(Panel2Panel) / p2p + len(Panel2Module) / p2m + len(Module2Module) / m2m
    print("Time_assembly", Time_assembly)
    Time = Time_prod + Time_finish + Time_assembly

    # Time = len(modules)*3 + len(Panel_wall)*5
    print("Time:", Time)
    
    total_length = 0
    for p in Panel_wall_pre:
        total_length = total_length + G.nodes[p]["length"]

    Cost = len(modules)*10 + total_length/1000*5
    print("Cost:", Cost)

    result.append(Cost)
    result.append(Time)

    return result

## read the data from input
df = pd.read_csv ('data/test01/graph.csv',header=None)*1
df = df.to_numpy()
prop = pd.read_csv('data/test01/property.csv',header=None)
prop = prop.to_numpy()
length = pd.read_csv('data/test01/length.csv',header=None)
length = length.to_numpy()
angle = pd.read_csv('data/test01/angle.csv',header=None)
angle = angle.to_numpy()


## create graph
G = nx.from_numpy_array(df)
G.remove_edges_from(nx.selfloop_edges(G))
for n in range(0,G.number_of_nodes()):
    G.nodes[n]["type"]=np.array2string(prop[n])[2:-2]
    G.nodes[n]["length"]=length[n][0]
for e in G.edges():
    G[e[0]][e[1]]["angle"] = angle[e[0],e[1]]



from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

class MyProblem(Problem):

    def __init__(self, G, **kwargs):
        super().__init__(n_var=G.number_of_edges(), n_obj=2, n_ieq_constr=0, xl=1, xu=2, vtype=int,**kwargs)
        self.G = G

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = []
        f2 = []
        for c in x:
            i = 0
            for e in self.G.edges():
                self.G[e[0]][e[1]]["connection"] = c[i]
                i = i + 1
            f1.append(function(self.G)[0])
            f2.append(function(self.G)[1])

        out["F"] = [f1, f2]


problem = MyProblem(G)

method = NSGA2(pop_size=30,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
            mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True,
            )

res = minimize(problem,
               method,
               termination=('n_gen', 40),
               seed=1,
               save_history=True
               )

print("Best solution found: %s" % res.X)
print("Function value: %s" % res.F)
print("Constraint violation: %s" % res.CV)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()