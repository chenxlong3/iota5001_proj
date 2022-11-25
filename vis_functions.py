from IMGraph import IMGraph
import matplotlib.pyplot as plt

method_marker = {
    "pagerank": 'o',
    "outdegree": "*",
    "betweenness": "x",
    "greedy": "+",
    "CELF": "s",
    "RIS": "D",
    "TIM": "p"
}
font = {'family': 'serif',
        'weight': 'normal',
        }

def vis_methods_spread(IM_G:IMGraph, **kwargs) -> None:
    fig_size = (8, 6)
    if "figsize" in kwargs.keys():
        fig_size= kwargs["figsize"]
    plt.figure(figsize=fig_size)

    for method in IM_G.method_spread_map.keys():
        plt.plot(IM_G.k_list, IM_G.method_spread_map[method],  marker=method_marker[method], label=method, markersize=8, mfc="None", linewidth=1)
        
    plt.legend()
    plt.title("Resulting Influence Spread of Different Methods", size=15, fontdict=font)
    plt.xticks(IM_G.k_list)
    plt.xlabel(r"$k$", size=10)
    plt.ylabel("Expected Spread", size=10, fontdict=font)
    
    plt.show()

def vis_methods_time(IM_G:IMGraph, **kwargs) -> None:
    fig_size = (8, 6)
    if "figsize" in kwargs.keys():
        fig_size= kwargs["figsize"]
    plt.figure(figsize=fig_size)

    for method in IM_G.method_time_map.keys():
        plt.plot(IM_G.k_list, IM_G.method_time_map[method],  marker=method_marker[method], label=method, markersize=8, mfc="None", linewidth=1)
        
    plt.legend()
    plt.title("Time Cost of Different Methods", size=15)
    plt.xticks(IM_G.k_list)
    plt.xlabel(r"$k$", size=10)
    plt.ylabel("Expected Spread", size=10)
    plt.show()