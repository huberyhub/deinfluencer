import networkx as nx
import random
import math
import copy

class InfluenceDeinfluenceModel:

    def __init__(self, graph, edge_weights_type='random', c=1, p=0.5, seed=None):
        """
        Initialize the InfluenceDeinfluenceModel.
        
        :param graph: A NetworkX graph
        :param edge_weights_type: How to assign edge probabilities ('random', 'fixed', 'dominate')
        :param c: Parameter for dominate_edge_weights
        :param p: Probability for tie-breaking in spread_influence
        :param seed: Optional random seed for reproducibility
        """
        self.graph = graph
        self.rng = random.Random(seed)  # local RNG with optional seed
        self.edge_weights(edge_weights_type, c)
        self.set_initial_states()
        self.activated_edges = set()
        self.selected_influencers = set()
        self.selected_deinfluencers = set()
        self.transition_counts = {'I->S': 0, 'D->S': 0, 'D->I': 0}
        self.p = p
        self.attempted_influence = set()
        self.attempted_deinfluence = set()
        self.assign_node_budgets_linear()
        self.assign_node_budgets_sqrt()

    def edge_weights(self, type, c):
        if type == 'random':
            self.random_edge_weights()
        elif type == 'fixed':
            self.fixed_edge_weights(p_is=1, p_ds=1, p_di=1)
        elif type == 'dominate':
            self.dominate_edge_weights(c)
        else:
            print("Invalid edge weights type. Using random edge weights.")
            self.random_edge_weights()
    
    def reset_transition_counts(self):
        self.transition_counts = {'I->S': 0, 'D->S': 0, 'D->I': 0}

    def assign_node_budgets_sqrt(self):
        for node in self.graph.nodes:
            degree = self.graph.degree(node)
            budget = math.sqrt(degree)
            self.graph.nodes[node]['budget_sqrt'] = budget
    
    def assign_node_budgets_linear(self):
        for node in self.graph.nodes:
            degree = self.graph.degree(node)
            budget = degree
            self.graph.nodes[node]['budget_linear'] = budget


    def random_edge_weights(self):
        for u, v in self.graph.edges:
            p_is = self.rng.uniform(0, 1)     # was random.uniform(0, 1)
            p_ds = self.rng.uniform(0, 1)
            p_di = self.rng.uniform(0, 1)
            self.graph[u][v]['p_is'] = p_is
            self.graph[u][v]['p_ds'] = p_ds
            self.graph[u][v]['p_di'] = p_di


    def fixed_edge_weights(self, p_is, p_ds, p_di):
        for u, v in self.graph.edges:
            self.graph[u][v]['p_is'] = p_is
            self.graph[u][v]['p_ds'] = p_ds
            self.graph[u][v]['p_di'] = p_di
    
    def dominate_edge_weights(self, c):
        for u, v in self.graph.edges:
            p_is = random.uniform(0, 1)
            p_ds = 1 - (1 - p_is)**c
            p_di = 1 - (1 - p_is)**c
            self.graph[u][v]['p_is'] = p_is
            self.graph[u][v]['p_ds'] = p_ds
            self.graph[u][v]['p_di'] = p_di

    def set_initial_states(self):
        nx.set_node_attributes(self.graph, 'S', 'state')

    def set_influencers(self, influencers):
        for node in influencers:
            self.graph.nodes[node]['state'] = 'I'

    def set_deinfluencers(self, deinfluencers):
        for node in deinfluencers:
            self.graph.nodes[node]['state'] = 'D'

    def pre_determine_active_edges(self):
        """
        Identifies and marks edges that are active in the current iteration.

        For each node:
        - If the node is 'I' (influencer), it attempts to influence its 'S' neighbors
            with probability p_is (stored in graph[u][v]['p_is']).
        - If the node is 'D' (deinfluencer), it attempts to:
            * deinfluence 'S' neighbors with probability p_ds, or
            * convert 'I' neighbors to 'D' with probability p_di.

        This method updates:
        - self.active_edges: set of edges that will attempt influence or deinfluence
        - self.attempted_influence / self.attempted_deinfluence: track which edges
            have already attempted an action, preventing repeated attempts.
        """
        self.active_edges = set()
        
        for node in self.graph.nodes:
            node_state = self.graph.nodes[node]['state']

            if node_state == 'I':
                # Node is an influencer
                for neighbor in self.graph.neighbors(node):
                    edge = tuple(sorted((node, neighbor)))  # canonical form
                    if (self.graph.nodes[neighbor]['state'] == 'S'
                        and self.rng.random() < self.graph[node][neighbor]['p_is']
                        and edge not in self.attempted_influence):
                        self.active_edges.add(edge)
                        self.attempted_influence.add(edge)

            elif node_state == 'D':
                # Node is a deinfluencer
                for neighbor in self.graph.neighbors(node):
                    edge = tuple(sorted((node, neighbor)))  # canonical form
                    neighbor_state = self.graph.nodes[neighbor]['state']

                    # Deinfluence an 'S' neighbor
                    if neighbor_state == 'S' and self.rng.random() < self.graph[node][neighbor]['p_ds']:
                        if edge not in self.attempted_deinfluence:
                            self.active_edges.add(edge)
                            self.attempted_deinfluence.add(edge)
                    # Convert an 'I' neighbor to 'D'
                    elif neighbor_state == 'I' and self.rng.random() < self.graph[node][neighbor]['p_di']:
                        if edge not in self.attempted_deinfluence:
                            self.active_edges.add(edge)
                            self.attempted_deinfluence.add(edge)

    def spread_influence(self):
        new_influenced = set()
        new_deinfluenced = set()
        simultaneous_influence = set()

        for edge in self.active_edges:
            node, neighbor = edge
            if self.graph.nodes[node]['state'] == 'I' and self.graph.nodes[neighbor]['state'] == 'S':
                if neighbor in new_deinfluenced:
                    simultaneous_influence.add(neighbor)
                else:
                    new_influenced.add(neighbor)
                    self.transition_counts['I->S'] += 1
            elif self.graph.nodes[node]['state'] == 'D':
                if self.graph.nodes[neighbor]['state'] == 'S':
                    if neighbor in new_influenced:
                        simultaneous_influence.add(neighbor)
                    else:
                        new_deinfluenced.add(neighbor)
                        self.transition_counts['D->S'] += 1
                elif self.graph.nodes[neighbor]['state'] == 'I':
                    new_deinfluenced.add(neighbor)
                    self.transition_counts['D->I'] += 1

        for node in new_influenced:
            if node not in simultaneous_influence:
                self.graph.nodes[node]['state'] = 'I'

        for node in new_deinfluenced:
            if node not in simultaneous_influence:
                self.graph.nodes[node]['state'] = 'D'

        for node in simultaneous_influence:
            # Resolve conflict using parameter p
            if random.random() < self.p:
                self.graph.nodes[node]['state'] = 'I'
            else:
                self.graph.nodes[node]['state'] = 'D'

    def influencer_spread_influence(self):
        new_influenced = set()
        for edge in self.active_edges:
            node, neighbor = edge
            if self.graph.nodes[node]['state'] == 'I' and self.graph.nodes[neighbor]['state'] == 'S':
                new_influenced.add(neighbor)

        for node in new_influenced:
            self.graph.nodes[node]['state'] = 'I'

    def run_cascade(self, steps):
        #self.pre_determine_active_edges()
        for _ in range(steps):
            self.pre_determine_active_edges()
            self.spread_influence()
    
    def run_cascade_until_stable(self):
        while True:
            previous_attempted_influence = len(self.attempted_influence)
            previous_attempted_deinfluence = len(self.attempted_deinfluence)
            
            self.pre_determine_active_edges()
            self.spread_influence()
            
            current_attempted_influence = len(self.attempted_influence)
            current_attempted_deinfluence = len(self.attempted_deinfluence)
            
            # Check if new attempts can be made
            if (current_attempted_influence == previous_attempted_influence and
                current_attempted_deinfluence == previous_attempted_deinfluence):
                break

    def run_cascade_influencer(self, steps):
        #self.pre_determine_active_edges()
        for _ in range(steps):
            self.pre_determine_active_edges()
            self.influencer_spread_influence()

    def evaluate_influence(self):
        """Evaluate the number of influenced nodes."""
        return sum(1 for node in self.graph.nodes if self.graph.nodes[node]['state'] == 'I')
    
    def evaluate_deinfluence(self):
        return sum(1 for node in self.graph.nodes if self.graph.nodes[node]['state'] == 'D')
    
    def evaluate_susceptible(self):
        return sum(1 for node in self.graph.nodes if self.graph.nodes[node]['state'] == 'S')
    
    def random_influencers(self, k):
        return set(self.rng.sample(list(self.graph.nodes), k))
    
    def random_deinfluencers(self, k):
        return set(self.rng.sample(list(self.graph.nodes), k))
    
    def reset_graph(self):
        self.set_initial_states()
        self.activated_edges = set()
        self.attempted_influence = set()  # Reset the attempted influence tracker
        self.attempted_deinfluence = set()  # Reset the attempted deinfluence tracker


    def select_deinfluencers_by_centrality(self, k, centrality_func, **kwargs):
        """
        Select k deinfluencers based on a centrality function.

        :param k: number of nodes to select
        :param centrality_func: a NetworkX centrality function (e.g., nx.degree_centrality)
        :param kwargs: additional named arguments for the centrality function
        """
        centrality = centrality_func(self.graph, **kwargs)
        # Sort nodes by descending centrality score
        return sorted(centrality, key=centrality.get, reverse=True)[:k]
    
    
    def select_deinfluencers_degree_centrality(self, k):
        return self.select_deinfluencers_by_centrality(k, nx.degree_centrality)

    def select_deinfluencers_closeness_centrality(self, k):
        return self.select_deinfluencers_by_centrality(k, nx.closeness_centrality)

    def select_deinfluencers_betweenness_centrality(self, k):
        return self.select_deinfluencers_by_centrality(k, nx.betweenness_centrality)

    def select_deinfluencers_eigenvector_centrality(self, k, max_iter=1000, tol=1e-6):
        try:
            return self.select_deinfluencers_by_centrality(
                k, nx.eigenvector_centrality, max_iter=max_iter, tol=tol
            )
        except nx.PowerIterationFailedConvergence:
            print(f"Power iteration failed to converge within {max_iter} iterations")
            return []

    def select_deinfluencers_pagerank_centrality(self, k):
        return self.select_deinfluencers_by_centrality(k, nx.pagerank)

    def select_deinfluencers_random(self, k):
        """Select k random deinfluencers."""
        population = sorted(self.graph.nodes)
        return random.sample(population, k)
    

    def greedy_hill_climbing(self, k, steps=3, R=5):
        """Select k initial influencers using the improved greedy algorithm."""
        best_influencers = set()

        for _ in range(k):
            best_candidate = None
            best_score = -1

            for node in self.graph.nodes:
                if node in best_influencers:
                    continue
                
                # Temporarily add the candidate node to the set of influencers
                current_influencers = best_influencers | {node}
                total_score = 0

                for _ in range(R):
                    self.activated_edges.clear()  # Reset activated edges
                    self.set_initial_states()
                    self.set_influencers(current_influencers)
                    self.run_cascade_influencer(steps)
                    total_score += self.evaluate_influence()

                avg_score = total_score / R

                if avg_score > best_score:
                    best_score = avg_score
                    best_candidate = node

            if best_candidate is not None:
                best_influencers.add(best_candidate)

        self.selected_influencers = best_influencers
        return best_influencers
    
    def greedy_hill_climbing_deinf(self, j, steps=3, R=3):
        """Select j de-influencers using greedy algorithm."""
        optimized_deinfluencer = set()

        # Create a deep copy of the original model
        original_model = copy.deepcopy(self)

        for _ in range(j):
            best_candidate = None
            best_score = -1

            for node in original_model.graph.nodes:
                if node in optimized_deinfluencer:
                    continue

                # Temporarily add the candidate node to the set of deinfluencers
                current_deinfluencers = optimized_deinfluencer | {node}
                total_score = 0

                for _ in range(R):
                    # Create a fresh copy of the original model for each run
                    model_copy = copy.deepcopy(original_model)
                    model_copy.set_deinfluencers(current_deinfluencers)
                    model_copy.run_cascade(steps)
                    total_score += model_copy.evaluate_deinfluence()

                avg_score = total_score / R

                if avg_score > best_score:
                    best_score = avg_score
                    best_candidate = node

            if best_candidate is not None:
                optimized_deinfluencer.add(best_candidate)

        # Reset the original model to its initial state
   
        return optimized_deinfluencer

    def greedy_hill_climbing_deinf_reduce_influence(self, j, steps=3, R=3):
        """Select j de-influencers using greedy algorithm to reduce most influence."""
        optimized_deinfluencer = set()

        # Create a deep copy of the original model
        original_model = copy.deepcopy(self)

        for _ in range(j):
            best_candidate = None
            best_score = float('inf')  # Minimize influence

            for node in original_model.graph.nodes:
                if node in optimized_deinfluencer:
                    continue

                # Temporarily add the candidate node to the set of deinfluencers
                current_deinfluencers = optimized_deinfluencer | {node}
                total_score = 0

                for _ in range(R):
                    # Create a fresh copy of the original model for each run
                    model_copy = copy.deepcopy(original_model)
                    model_copy.set_deinfluencers(current_deinfluencers)
                    model_copy.run_cascade(steps)
                    total_score += model_copy.evaluate_influence()  # We want to minimize influence

                avg_score = total_score / R

                if avg_score < best_score:  # Looking for minimum influence
                    best_score = avg_score
                    best_candidate = node

            if best_candidate is not None:
                optimized_deinfluencer.add(best_candidate)
                
        return optimized_deinfluencer
    

    def greedy_hill_climbing_deinf_restricted(self, j, node_list, steps=3, R=3):
        """Select j de-influencers using greedy algorithm from the provided node_list."""
        optimized_deinfluencer = set()
        original_model = copy.deepcopy(self)

        for _ in range(j):
            best_candidate = None
            best_score = -1

            for node in node_list:
                if node in optimized_deinfluencer:
                    continue
                current_deinfluencers = optimized_deinfluencer | {node}
                total_score = 0

                for _ in range(R):
                    model_copy = copy.deepcopy(original_model)
                    model_copy.set_deinfluencers(current_deinfluencers)
                    model_copy.run_cascade(steps)
                    total_score += model_copy.evaluate_deinfluence()

                avg_score = total_score / R

                if avg_score > best_score:
                    best_score = avg_score
                    best_candidate = node

            if best_candidate is not None:
                optimized_deinfluencer.add(best_candidate)

        return optimized_deinfluencer
    
    def select_deinfluencers_from_ini_influencers(self, j):
        influencers = list(self.selected_influencers)  # Convert set to list
        deinfluencers = random.sample(influencers, j)  # Select j deinfluencers randomly from the selected influencers
        return deinfluencers
    
    def select_deinfluencers_from_ini_influencers_degree_centrality(self, j):
        influencers = list(self.selected_influencers)
        return sorted(influencers, key=lambda node: self.graph.degree(node), reverse=True)[:j]
    
    def select_deinfluencers_from_not_ini_influencers(self, j):
        not_influencers = [node for node in self.graph.nodes if node not in self.selected_influencers]
        deinfluencers = random.sample(not_influencers, j)
        return deinfluencers

    def select_deinfluencers_from_influencers(self, j):
        influencers = [node for node in self.graph.nodes if self.graph.nodes[node]['state'] == 'I']  # Convert set to list
        deinfluencers = random.sample(influencers, j)  # Select j deinfluencers randomly from the selected influencers
        return deinfluencers
    
    def select_deinfluencers_from_influencers_degree_centrality(self, j):
        influencers = [node for node in self.graph.nodes if self.graph.nodes[node]['state'] == 'I']
        return sorted(influencers, key=lambda node: self.graph.degree(node), reverse=True)[:j]
    
    def select_deinfluencers_from_not_influencers(self, j):
        not_influencers = [node for node in self.graph.nodes if self.graph.nodes[node]['state'] != 'I']
        deinfluencers = random.sample(not_influencers, j)
        return deinfluencers
    
    def select_deinfluencers_from_influencers_greedy(self, j):
        influencers = [node for node in self.graph.nodes if self.graph.nodes[node]['state'] == 'I']
        return self.greedy_hill_climbing_deinf(j, influencers)