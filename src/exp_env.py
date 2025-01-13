import model as my_model
from pyexpat import model
import copy
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

"""
Experiment Installations
------------------------------------------------------------------------------------------------------------------------
"""

def choose_influencers(model, num_influencers, method='random'):
    if method == 'random':
        return model.random_influencers(num_influencers)
    elif method == 'hill_climbing':
        return model.greedy_hill_climbing(num_influencers, steps=10, R=10)
    else:
        raise ValueError("Unsupported method for selecting influencers")

def run_influence_cascade(graph, num_influencers, steps, selection_method='random'):
    # Initialize the model
    model = my_model.InfluenceDeinfluenceModel(graph, selection_method)
    # Choose influencers
    influencers = choose_influencers(model, num_influencers, method=selection_method)
    model.set_influencers(influencers)
    model.selected_influencers = influencers
    # Run the cascade
    model.run_cascade(steps)
    # Return the updated graph and model
    return model

def run_cascade_with_recording(model, num_deinfluencers, steps):
  
    deinfluencers = model.select_deinfluencers_random(num_deinfluencers)
    model.set_deinfluencers(deinfluencers)

    # Evaluate the influence and deinfluence
    num_influenced = model.evaluate_influence()
    num_deinfluenced = model.evaluate_deinfluence()
    num_susceptible = model.evaluate_susceptible()
    
    # Record the numbers of influencers, deinfluencers, and susceptibles at each step
    influencer_counts = [num_influenced]
    deinfluencer_counts = [num_deinfluenced]
    susceptible_counts = [num_susceptible]

    for step in range(steps):
        model.pre_determine_active_edges()
        model.spread_influence()
        
        influencer_counts.append(model.evaluate_influence())
        deinfluencer_counts.append(model.evaluate_deinfluence())
        susceptible_counts.append(model.evaluate_susceptible())

    return influencer_counts, deinfluencer_counts, susceptible_counts

def run_simple_cascade(steps):
    model.set_influencers(model.selected_influencers)
    model.run_cascade(steps)
    return model

def shuffle_deinfluencers(model, k, deinfluencers_dict):
    methods_to_shuffle = ['Random', 'ExIniInf', 'ExAllInf', 'IniInf', 'AllInf', 'RkIniInf', 'RkAllInf']
    for method in methods_to_shuffle:
        if method in deinfluencers_dict:
            if method == 'Random':
                deinfluencers_dict[method] = model.select_deinfluencers_random(k)
            elif method == 'ExIniInf':
                deinfluencers_dict[method] = model.select_deinfluencers_from_not_ini_influencers(k)
            elif method == 'ExAllInf':
                deinfluencers_dict[method] = model.select_deinfluencers_from_not_influencers(k)
            elif method == 'IniInf':
                deinfluencers_dict[method] = model.select_deinfluencers_from_ini_influencers(k)
            elif method == 'AllInf':
                deinfluencers_dict[method] = model.select_deinfluencers_from_influencers(k)
            elif method == 'RkIniInf':
                deinfluencers_dict[method] = model.select_deinfluencers_from_ini_influencers_degree_centrality(k)
            elif method == 'RkAllInf':
                deinfluencers_dict[method] = model.select_deinfluencers_from_influencers_degree_centrality(k)
    return deinfluencers_dict


# Define the combined count function
def count_deinfluenced(model, deinfluencers, num_runs, steps):
    total_deinfluenced = 0
    total_influenced = 0
    total_transition_counts = {'I->S': 0, 'D->S': 0, 'D->I': 0}
    # Create a deep copy of the model to ensure initial influencers remain the same
    initial_model = copy.deepcopy(model)
    
    for _ in range(num_runs):
        # Reset the model to the initial state with the same influencers
        model = copy.deepcopy(initial_model)
        model.reset_transition_counts()
        model.set_deinfluencers(deinfluencers)
        model.run_cascade(steps)
        
        total_deinfluenced += model.evaluate_deinfluence()
        total_influenced += model.evaluate_influence()
        
        for key in total_transition_counts:
            total_transition_counts[key] += model.transition_counts[key]
            
    return total_deinfluenced / num_runs, total_influenced / num_runs, {key: total / num_runs for key, total in total_transition_counts.items()}



def average_results(deinfluencers_list, model, num_runs, steps):
    cumulative_results = {}
    for k, deinfluencers_methods in deinfluencers_list:
        if k not in cumulative_results:
            cumulative_results[k] = {method: (0, 0, {'I->S': 0, 'D->S': 0, 'D->I': 0}) for method in deinfluencers_methods.keys()}
        
        for _ in range(num_runs):
            shuffled_deinfluencers_methods = {method: shuffle_deinfluencers(model, k, deinfluencers) if method in ['Random', 'ExIniInf', 'ExAllInf', 'IniInf', 'AllInf', 'RkIniInf', 'RkAllInf'] else deinfluencers for method, deinfluencers in deinfluencers_methods.items()}
            results = {
                method: count_deinfluenced(model, deinfluencers, num_runs, steps)
                for method, deinfluencers in shuffled_deinfluencers_methods.items()
            }
            
            for method, result in results.items():
                cumulative_results[k][method] = (
                    cumulative_results[k][method][0] + result[0],
                    cumulative_results[k][method][1] + result[1],
                    {key: cumulative_results[k][method][2][key] + result[2][key] for key in result[2]}
                )
    average_results = {
        k: {
            method: (
                cumulative_results[k][method][0] / num_runs,
                cumulative_results[k][method][1] / num_runs,
                {key: cumulative_results[k][method][2][key] / num_runs for key in cumulative_results[k][method][2]}
            )
            for method in cumulative_results[k]
        }
        for k in cumulative_results
    }
    
    return average_results


def average_results_simple(deinfluencers_list, model, num_runs, steps):
    cumulative_results = {}
    for k, deinfluencers_methods in deinfluencers_list:
        if k not in cumulative_results:
            cumulative_results[k] = {method: (0, 0, {'I->S': 0, 'D->S': 0, 'D->I': 0}) for method in deinfluencers_methods.keys()}
        
        for _ in range(num_runs):
            shuffled_deinfluencers_methods = {method: shuffle_deinfluencers(model, k, deinfluencers) if method in ['Random','High Degree', 'Low Degree'] else deinfluencers for method, deinfluencers in deinfluencers_methods.items()}
            results = {
                method: count_deinfluenced(model, deinfluencers, num_runs, steps)
                for method, deinfluencers in shuffled_deinfluencers_methods.items()
            }
            
            for method, result in results.items():
                cumulative_results[k][method] = (
                    cumulative_results[k][method][0] + result[0],
                    cumulative_results[k][method][1] + result[1],
                    {key: cumulative_results[k][method][2][key] + result[2][key] for key in result[2]}
                )
    average_results = {
        k: {
            method: (
                cumulative_results[k][method][0] / num_runs,
                cumulative_results[k][method][1] / num_runs,
                {key: cumulative_results[k][method][2][key] / num_runs for key in cumulative_results[k][method][2]}
            )
            for method in cumulative_results[k]
        }
        for k in cumulative_results
    }
    return average_results

def average_results_without_shuffle(deinfluencers_list, model, num_runs, steps):
    cumulative_results = {}
    for k, deinfluencers_methods in deinfluencers_list:
        if k not in cumulative_results:
            cumulative_results[k] = {method: (0, 0, {'I->S': 0, 'D->S': 0, 'D->I': 0}) for method in deinfluencers_methods.keys()}
        
        for _ in range(num_runs):
            results = {
                method: count_deinfluenced(model, deinfluencers, num_runs, steps)
                for method, deinfluencers in deinfluencers_methods.items()
            }
            
            for method, result in results.items():
                cumulative_results[k][method] = (
                    cumulative_results[k][method][0] + result[0],
                    cumulative_results[k][method][1] + result[1],
                    {key: cumulative_results[k][method][2][key] + result[2][key] for key in result[2]}
                )
    average_results = {
        k: {
            method: (
                cumulative_results[k][method][0] / num_runs,
                cumulative_results[k][method][1] / num_runs,
                {key: cumulative_results[k][method][2][key] / num_runs for key in cumulative_results[k][method][2]}
            )
            for method in cumulative_results[k]
        }
        for k in cumulative_results
    }
    return average_results


"""
Selection Schemes
------------------------------------------------------------------------------------------------------------------------
"""

def select_deinfluencers(k_deinfluencers_ls, model):
    deinfluencers_list = []
    for k in k_deinfluencers_ls:
        deinfluencers_dict = {}
        # Sample function calls to model object methods
        deinfluencers_dict['Random'] = model.select_deinfluencers_random(k)
        deinfluencers_dict['ExIniInf'] = model.select_deinfluencers_from_not_ini_influencers(k)
        deinfluencers_dict['ExAllInf'] = model.select_deinfluencers_from_not_influencers(k)
        deinfluencers_dict['IniInf'] = model.select_deinfluencers_from_ini_influencers(k)
        deinfluencers_dict['AllInf'] = model.select_deinfluencers_from_influencers(k)
        deinfluencers_dict['RkIniInf'] = model.select_deinfluencers_from_ini_influencers_degree_centrality(k)
        deinfluencers_dict['RkAllInf'] = model.select_deinfluencers_from_influencers_degree_centrality(k)
        deinfluencers_dict['Degree'] = model.select_deinfluencers_degree_centrality(k)
        deinfluencers_dict['Closeness'] = model.select_deinfluencers_closeness_centrality(k)
        deinfluencers_dict['Betweenness'] = model.select_deinfluencers_betweenness_centrality(k)
        deinfluencers_dict['Eigenvector'] = model.select_deinfluencers_eigenvector_centrality(k, max_iter=1000, tol=1e-06)
        deinfluencers_dict['PageRank'] = model.select_deinfluencers_pagerank_centrality(k)
    
        deinfluencers_list.append((k, deinfluencers_dict))
    return deinfluencers_list

def select_deinfluencers_cen(k_deinfluencers_ls, model):
    deinfluencers_list = []
    for k in k_deinfluencers_ls:
        deinfluencers_dict = {}
        # Sample function calls to model object methods
        deinfluencers_dict['Random'] = model.select_deinfluencers_random(k)
        deinfluencers_dict['Degree'] = model.select_deinfluencers_degree_centrality(k)
        deinfluencers_dict['Closeness'] = model.select_deinfluencers_closeness_centrality(k)
        deinfluencers_dict['Betweenness'] = model.select_deinfluencers_betweenness_centrality(k)
        deinfluencers_dict['Eigenvector'] = model.select_deinfluencers_eigenvector_centrality(k, max_iter=1000, tol=1e-06)
        deinfluencers_dict['PageRank'] = model.select_deinfluencers_pagerank_centrality(k)
    
        deinfluencers_list.append((k, deinfluencers_dict))
    return deinfluencers_list

def select_deinfluencers_cri(k_deinfluencers_ls, model):
    deinfluencers_list = []
    for k in k_deinfluencers_ls:
        deinfluencers_dict = {}
        # Sample function calls to model object methods
        deinfluencers_dict['Random'] = model.select_deinfluencers_random(k)
        deinfluencers_dict['Degree'] = model.select_deinfluencers_degree_centrality(k)
        deinfluencers_dict['AllInf'] = model.select_deinfluencers_from_influencers(k)
        deinfluencers_dict['IniInf'] = model.select_deinfluencers_from_ini_influencers(k)
        deinfluencers_dict['ExAllInf'] = model.select_deinfluencers_from_not_influencers(k)
        deinfluencers_dict['ExIniInf'] = model.select_deinfluencers_from_not_ini_influencers(k)
        deinfluencers_dict['RkAllInf'] = model.select_deinfluencers_degree_centrality(k)
        deinfluencers_dict['RkIniInf'] = model.select_deinfluencers_from_ini_influencers_degree_centrality(k)
        #deinfluencers_dict['GreedyDeinfAllInf'] = model.select_deinfluencers_from_influencers_greedy(k)
    
        deinfluencers_list.append((k, deinfluencers_dict))
    return deinfluencers_list

def select_deinfluencers_2(k_deinfluencers_ls, model):
    deinfluencers_list = []
    for k in k_deinfluencers_ls:
        deinfluencers_dict = {}
        # Sample function calls to model object methods
        deinfluencers_dict['RkIniInf'] = model.select_deinfluencers_from_ini_influencers_degree_centrality(k)
        deinfluencers_dict['RkAllInf'] = model.select_deinfluencers_from_influencers_degree_centrality(k)
        deinfluencers_dict['Degree'] = model.select_deinfluencers_degree_centrality(k)
    
        deinfluencers_list.append((k, deinfluencers_dict))
    return deinfluencers_list

def select_deinfluencers_greedy(k_deinfluencers_ls, model):
    deinfluencers_list = []
    for k in k_deinfluencers_ls:
        deinfluencers_dict = {}
        # Sample function calls to model object methods
        deinfluencers_dict['Random'] = model.select_deinfluencers_random(k)
        deinfluencers_dict['Degree'] = model.select_deinfluencers_degree_centrality(k)
        deinfluencers_dict['GreedyDeinf'] = model.greedy_hill_climbing_deinf(k)
        deinfluencers_dict['GreedyInf'] = model.greedy_hill_climbing_deinf_reduce_influence(k)
    
        deinfluencers_list.append((k, deinfluencers_dict))
    return deinfluencers_list

def select_deinfluencers_cgn(k_deinfluencers_ls, model, epochs):
    deinfluencers_list = []
    for k in k_deinfluencers_ls:
        deinfluencers_dict = {}
        # Sample function calls to model object methods
        deinfluencers_dict['Random'] = model.select_deinfluencers_random(k)
        deinfluencers_dict['Degree'] = model.select_deinfluencers_degree_centrality(k)
        deinfluencers_dict['CGN1'] = model.select_deinfluencers_gnn_cgn1(k, epochs)
        deinfluencers_dict['CGN2'] = model.select_deinfluencers_gnn_cgn2(k, epochs)
        deinfluencers_list.append((k, deinfluencers_dict))
    return deinfluencers_list

def select_deinfluencers_cgn_fea(k_deinfluencers_ls, model, node_id, dataset_folder, epochs):
    deinfluencers_list = []
    for k in k_deinfluencers_ls:
        deinfluencers_dict = {}
        # Sample function calls to model object methods
        deinfluencers_dict['Random'] = model.select_deinfluencers_random(k)
        deinfluencers_dict['Degree'] = model.select_deinfluencers_degree_centrality(k)
        deinfluencers_dict['CGN_fea'] = model.select_deinfluencers_gnn_fea(k, node_id, dataset_folder, epochs)
        deinfluencers_list.append((k, deinfluencers_dict))
    return deinfluencers_list

def select_deinfluencers_budget(budget_ls, model, type):
    deinfluencers_list = []
    for budget in budget_ls:
        deinfluencers_dict = {}
        # Sample function calls to model object methods
        deinfluencers_dict['Random'] = choose_random_nodes_until_budget(model.graph,budget,type)
        deinfluencers_dict['High Degree'] = choose_highest_degree_nodes_until_budget(model.graph,budget,type)
        deinfluencers_dict['Low Degree'] = choose_lowest_degree_nodes_until_budget(model.graph,budget,type)

        deinfluencers_list.append((budget, deinfluencers_dict))
    return deinfluencers_list

def select_deinfluencers_budget_naive(budget_ls, model, type):
    deinfluencers_list = []
    for budget in budget_ls:
        deinfluencers_dict = {}
        # Sample function calls to model object methods
        deinfluencers_dict['Random'] = choose_random_nodes_until_budget_naive(model.graph,budget,type)
        deinfluencers_dict['High Degree'] = choose_highest_degree_nodes_until_budget_naive(model.graph,budget,type)
        deinfluencers_dict['Low Degree'] = choose_lowest_degree_nodes_until_budget_naive(model.graph,budget,type)

        deinfluencers_list.append((budget, deinfluencers_dict))
    return deinfluencers_list

"""
Selection Schemes for Costs
------------------------------------------------------------------------------------------------------------------------
"""

def choose_highest_degree_nodes_until_budget_naive(graph, budget, type):
    selected_nodes = set()
    sorted_nodes = sorted(graph.nodes, key=lambda node: graph.degree(node), reverse=True)
    current_budget = 0
    
    for node in sorted_nodes:
        node_budget = graph.nodes[node][type]
        if current_budget + node_budget > budget:
            break
        selected_nodes.add(node)
        current_budget += node_budget
    
    return selected_nodes

def choose_random_nodes_until_budget_naive(graph, budget, type):
    selected_nodes = set()
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    current_budget = 0
    for node in nodes:
        node_budget = graph.nodes[node][type]
        if current_budget + node_budget > budget:
            break
        selected_nodes.add(node)
        current_budget += node_budget
    return selected_nodes

def choose_lowest_degree_nodes_until_budget_naive(graph, budget, type):
    selected_nodes = set()
    sorted_nodes = sorted(graph.nodes, key=lambda node: graph.degree(node), reverse=False)
    current_budget = 0
    for node in sorted_nodes:
        node_budget = graph.nodes[node][type]
        if current_budget + node_budget > budget:
            break
        selected_nodes.add(node)
        current_budget += node_budget
    return selected_nodes


def choose_highest_degree_nodes_until_budget(graph, budget, type):
    selected_nodes = set()
    sorted_nodes = sorted(graph.nodes, key=lambda node: graph.degree(node), reverse=True)
    current_budget = 0
    for node in sorted_nodes:
        node_budget = graph.nodes[node][type]
        if current_budget + node_budget <= budget:
            selected_nodes.add(node)
            current_budget += node_budget
    # Check if there is remaining budget
    if current_budget < budget:
        for node in sorted_nodes:
            if node not in selected_nodes:
                node_budget = graph.nodes[node][type]
                if current_budget + node_budget <= budget:
                    selected_nodes.add(node)
                    current_budget += node_budget
                if current_budget == budget:
                    break
    return selected_nodes

def choose_random_nodes_until_budget(graph, budget, type):
    selected_nodes = set()
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    current_budget = 0
    for node in nodes:
        node_budget = graph.nodes[node][type]
        if current_budget + node_budget <= budget:
            selected_nodes.add(node)
            current_budget += node_budget
    # Check if there is remaining budget
    if current_budget < budget:
        for node in nodes:
            if node not in selected_nodes:
                node_budget = graph.nodes[node][type]
                if current_budget + node_budget <= budget:
                    selected_nodes.add(node)
                    current_budget += node_budget
                if current_budget == budget:
                    break
    return selected_nodes

def choose_lowest_degree_nodes_until_budget(graph, budget, type):
    selected_nodes = set()
    sorted_nodes = sorted(graph.nodes, key=lambda node: graph.degree(node))
    current_budget = 0
    for node in sorted_nodes:
        node_budget = graph.nodes[node][type]
        if current_budget + node_budget <= budget:
            selected_nodes.add(node)
            current_budget += node_budget
    # Check if there is remaining budget
    if current_budget < budget:
        for node in sorted_nodes:
            if node not in selected_nodes:
                node_budget = graph.nodes[node][type]
                if current_budget + node_budget <= budget:
                    selected_nodes.add(node)
                    current_budget += node_budget
                if current_budget == budget:
                    break
    return selected_nodes

"""
Plotting Functions
------------------------------------------------------------------------------------------------------------------------
"""

def plot_deinfluencer_results_new(results, G, graph_type, num_nodes, num_edges, num_influencers, influencers_cascade_steps, general_cascade_steps, num_avg_runs):
    """
    Plot the effectiveness of deinfluencers by selection method and budget, with an info box.

    Parameters:
    - results: A dictionary where keys are budgets (k values) and values are dictionaries
               of methods with their corresponding deinfluenced and influenced nodes counts.
    - G: The graph object containing the nodes.
    - graph_type: Type of the graph.
    - num_nodes: Number of nodes in the graph.
    - num_edges: Number of edges in the graph.
    - num_influencers: Number of influencers.
    - influencers_cascade_steps: Number of cascade steps for influencers.
    - general_cascade_steps: Number of general cascade steps.
    - num_avg_runs: Number of average runs.
    """

    # Define different marker styles and colors for each method
    marker_styles = {
        'Random': 'o',
        'ExIniInf': 's',
        'ExAllInf': 'D',
        'IniInf': '*',
        'AllInf': 'h',
        'RkIniInf': 'X',
        'RkAllInf': 'd',
        'Degree': 'v',
        'Closeness': '^',
        'Betweenness': '<',
        'Eigenvector': '>',
        'PageRank': 'P',
        'GreedyDeinf': '+',
        'GreedyInf': 'x',
        'GreedyDeinfAllInf': 'x',
        'CGN1': 'o',
        'CGN2': 's',
        'CGN_fea': 'D'
    }

    color_styles = {
        'Random': 'tab:blue',
        'ExIniInf': 'tab:orange',
        'ExAllInf': 'tab:green',
        'IniInf': 'tab:red',
        'AllInf': 'tab:purple',
        'RkIniInf': 'tab:brown',
        'RkAllInf': 'tab:pink',
        'Degree': 'tab:gray',
        'Closeness': 'tab:olive',
        'Betweenness': 'tab:cyan',
        'Eigenvector': 'tab:blue',
        'PageRank': 'tab:orange',
        'GreedyDeinf': 'tab:green',
        'GreedyInf': 'tab:purple',
        'GreedyDeinfAllInf': 'tab:purple',
        'CGN1': 'tab:orange',
        'CGN2': 'tab:green',
        'CGN_fea': 'tab:red'
    }

    # Create subplots, including an additional one for the info box
    fig, axs = plt.subplots(4, 1, figsize=(9, 15))

    # Create an info box in the fourth subplot
    axs[3].axis('off')  # Hide the axis
    info_text = (f"Graph Type: {graph_type}\n"
                 f"Nodes: {num_nodes}\n"
                 f"Edges: {num_edges}\n"
                 f"Influencers: {num_influencers}\n"
                 f"Influencer Cascade Steps: {influencers_cascade_steps}\n"
                 f"General Cascade Steps: {general_cascade_steps}\n"
                 f"Average Runs: {num_avg_runs}\n")

    # Display the info text in the last subplot
    axs[3].text(0.5, 0.5, info_text, fontsize=12, va='center', ha='center', bbox=dict(facecolor='white', edgecolor='black'))

    # Adjust layout to make it look nice
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Create line plots
    methods = results[next(iter(results))].keys()  # Get all methods from the first key
    k_values = sorted(results.keys())  # Sort k values for plotting

    total_nodes = len(G.nodes)

    for method in methods:
        deinfluenced_nodes = [results[k][method][0] for k in k_values]
        influenced_nodes = [results[k][method][1] for k in k_values]
        susceptible_nodes = [total_nodes - (influenced + deinfluenced) for influenced, deinfluenced in zip(influenced_nodes, deinfluenced_nodes)]

        marker = marker_styles.get(method, 'o')  # Default to 'o' if method is not in marker_styles
        color = color_styles.get(method, 'tab:blue')  # Default to 'tab:blue' if method is not in color_styles

        axs[0].plot(k_values, deinfluenced_nodes, label=method, marker=marker, color=color)
        axs[1].plot(k_values, influenced_nodes, label=method, marker=marker, color=color)
        axs[2].plot(k_values, susceptible_nodes, label=method, marker=marker, color=color)

    # Set font size for axis labels and tick labels

    for ax in axs[:3]:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.tick_params(axis='both', which='major', labelsize=13)  # Increase tick label font size
        ax.set_xlabel('Number of Initial Deinfluencers Selected', fontsize=14)
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)
        
        # Set the x and y axes to show integer ticks only
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axs[0].set_ylabel('Final Deinfluencers Count', fontsize=13)
    axs[1].set_ylabel('Final Influencers Count', fontsize=13)
    axs[2].set_ylabel('Final Susceptible Nodes', fontsize=13)

    plt.tight_layout()
    plt.show()



def plot_deinfluencer_results_exp2(results, G):
    """
    Plot the effectiveness of deinfluencers by selection method and budget.
    
    Parameters:
    - results: A dictionary where keys are budgets (k values) and values are dictionaries
               of methods with their corresponding deinfluenced and influenced nodes counts.
    - G: The graph object containing the nodes.
    """
    # Plotting results
    fig, axs = plt.subplots(3, figsize=(15, 15))
    
    # Set titles for individual subplots
    axs[0].set_title('Effectiveness of Deinfluencers by Selection Method and Budget')
    axs[1].set_title('Influence Reduction by Deinfluencer Selection Method and Budget')
    axs[2].set_title('Remaining Susceptible Nodes by Deinfluencer Selection Method and Budget')

    # Create line plots
    methods = results[next(iter(results))].keys()  # Get all methods from the first key
    k_values = sorted(results.keys())  # Sort k values for plotting
    total_nodes = len(G.nodes)

    for method in methods:
        deinfluenced_nodes = [results[k][method][0] for k in k_values]
        influenced_nodes = [results[k][method][1] for k in k_values]
        susceptible_nodes = [total_nodes - (influenced + deinfluenced) for influenced, deinfluenced in zip(influenced_nodes, deinfluenced_nodes)]

        axs[0].plot(k_values, deinfluenced_nodes, label=method, marker="o")
        axs[1].plot(k_values, influenced_nodes, label=method, marker="o")
        axs[2].plot(k_values, susceptible_nodes, label=method, marker="o")

    # Set y-axis limits
    for ax in axs:
        ax.set_ylim(0, 2000)
    
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[0].set_xlabel('Number of Deinfluencers')
    axs[0].set_ylabel('Average Number of Final Deinfluenced Nodes')

    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[1].set_xlabel('Number of Deinfluencers')
    axs[1].set_ylabel('Average Number of Final Influenced Nodes')

    axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[2].set_xlabel('Number of Deinfluencers')
    axs[2].set_ylabel('Average Number of Final Susceptible Nodes')

    plt.tight_layout()
    plt.show()

def plot_cascade_results(influencer_counts, deinfluencer_counts, susceptible_counts):
    steps = range(len(influencer_counts))
    plt.figure(figsize=(6, 4))
    plt.plot(steps, influencer_counts, label='Influencers', marker='o')
    plt.plot(steps, deinfluencer_counts, label='Deinfluencers', marker='s')
    plt.plot(steps, susceptible_counts, label='Susceptibles', marker='^')
    plt.xticks(range(len(steps)), [int(step) for step in steps])  # Show integer steps on x-axis
    plt.xlabel('Steps')
    plt.ylabel('Count')
    plt.title('General Cascade Dynamics Over Time')
    plt.legend()
    plt.grid(False)
    plt.ylim(0, 2000)  # Set y-axis range to 2000
    plt.show()