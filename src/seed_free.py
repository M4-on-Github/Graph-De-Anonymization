# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:49:32 2024

@author: thsou
"""

# from google.colab import drive
# drive.mount('/content/drive')

### import important libraries
import os
import argparse

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent



### text colors
RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"


print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")




### Index by index evaluation 

def get_seed_pairs(dictionary_G1, dictionary_G2, validation_seed_pairs, top_k = 100):

    top_nodes_score_G1 = sorted(dictionary_G1.items(), key = lambda x:x[1], reverse=True)[:top_k]
    top_nodes_score_G2 = sorted(dictionary_G2.items(), key = lambda x:x[1], reverse=True)[:top_k]

    correct_pairs = list()
    all_pairs = list() 
    
    count_right_dict = dict() 
    index_based_accuracy = dict() 

    count = 0

    for i in range(top_k):
        tup = (top_nodes_score_G1[i][0], top_nodes_score_G2[i][0])
        all_pairs.append(tup)
        
        if tup in validation_seed_pairs:
            print(tup, i, count, end=',')
            count += 1
            correct_pairs.append(tup)
        
        if i%25 == 0: 
            count_right_dict[i] = count

            if i == 0:
                index_based_accuracy[i] = 0
            else: 
                index_based_accuracy[i] = (count/(i+1))*100

    accuracy = len(correct_pairs) / len(validation_seed_pairs)

    return all_pairs, correct_pairs, count, accuracy, count_right_dict, index_based_accuracy

### Plot first k elements 

# Function to plot the first k elements of two dictionaries
def plot_first_k_elements_two_dicts(dict1, dict2, k, method = ''):
    # Get the first k items for each dictionary
    first_k_items1 = list(dict1.items())[:k]
    first_k_items2 = list(dict2.items())[:k]
    
    keys1 = [item[0] for item in first_k_items1]
    values1 = [item[1] for item in first_k_items1]
    values2 = [item[1] for item in first_k_items2]
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(values1)), values1, marker='o', label='Count')
    plt.plot(range(len(values2)), values2, marker='s', label='Accuracy', linestyle='--')
    plt.title(f'Index Matching Between Two Graphs ({str(method)})')
    plt.xlabel('Compaerd Index')
    plt.ylabel('Matching Value ( Count, Accuracy )')
    plt.xticks(range(len(keys1)), keys1, rotation=45, ha='right')
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()



def get_all_edges_trussness_and_decomposition(G):

    """
        Compute and assign the trussness score to each edge in graph G.

        This function iteratively finds the k-truss subgraph for increasing values
        of k, starting from k=2. For each value of k, it determines the edges that
        belong to the (k-1)-truss but not the k-truss. For these edges, it assigns
        the trussness score as k-1 and also calculates and assigns their support.
        The trussness score and support are stored in the 'weight' attribute of the
        edges in a copy of the original graph G.

        Parameters:
        - G (NetworkX graph): The input graph.

        Returns:
        - G_main (NetworkX graph): A copy of G where each edge has an assigned
                'weight' attribute containing a dictionary with the trussness and
                support.
        - trussness_dict (dict): A dictionary where the keys are trussness scores
                and the values are lists of edges that have the corresponding trussness.
    """

    i = 2
    trussness_dict = dict()
    G.remove_edges_from(nx.selfloop_edges(G))
    G_main = G.copy()

    G1 = None
    
    total_edges = 0
    while (1):
        G = nx.k_truss(G, i)

        if i > 2:
            edges = set(list(G1.edges())) - set(list(G.edges())) # get_trussness_edges(edge_list1, edge_list2) #
            edges = list(edges)

            # edges = get_trussness_edges(list(G1.edges()), list(G.edges()))
            for u, v in edges:
                if G_main.has_edge(u, v):
                    G_main[u][v]['weight'] = i-1

            if len(G.edges()) == 0 and len(edges) == 0:
                break

            trussness_dict[i-1] = edges
            # edgeSubG = G_main.edge_subgraph(edges)
            # print(f"Edge Subgraph: , {edgeSubG}, Trussness: {i-1}")
            # nx.draw(edgeSubG , with_labels = True)
            # plt.show()
        print(i, end='-')
        G1 = G.copy()
        i += 1

    return G_main, trussness_dict

def get_all_edges_coreness_and_decomposition(G):

    G.remove_edges_from(nx.selfloop_edges(G))
    G_main = G.copy()
    # print(G_main,'--------Main Graph Info-------')
    coreness_dict = dict()
    # nodes_coresness_dict = dict()

    prev_Core_Graph = nx.Graph()
    edge_count = 0

    i = 0
    while(1):
        Gk = nx.k_core(G, i)

        deducted_nodes =  set(list(prev_Core_Graph.nodes())).difference(set(list(Gk.nodes())))
        deducted_edges = set([tuple(sorted([u,v])) for u, v in prev_Core_Graph.edges()]) - (set([ tuple(sorted([u, v])) for u,v in Gk.edges()]))
        # nodes_set = set(list(G_main.edge_subgraph(deducted_edges).nodes()))

        # edge_count += len(deducted_edges)

        ###print(edge_count)


        if i <= 1:
            pass
        # elif i == 1:
        #     coreness_dict[i-1] = list(deducted_nodes)
        else:
            for u, v in deducted_edges:
                if G_main.has_edge(u, v):
                    G_main[u][v]['weight'] = i-1

            coreness_dict[i-1] = list(deducted_edges)
            # nodes_coresness_dict[i-1] = nodes_set
        # nodeSubG = prev_Core_Graph.subgraph(deducted_nodes)
        # edgeSubG = prev_Core_Graph.edge_subgraph(deducted_edges)


        # if i > 0 and len(edgeSubG.nodes()) > 0:
        #     print(f"Core: {i-1}")
        #     print(f"Edge Subgraph: , {edgeSubG}")
        #     # nx.draw(edgeSubG , with_labels = True)
        #     # plt.show()

        if len(Gk.nodes())== 0 and len(Gk.edges()) == 0:
            break
        print(i, end="-")
        i += 1
        prev_Core_Graph = Gk


    unused_key_list = list()
    for key, value in coreness_dict.items():
        if len(coreness_dict[key]) == 0:
            unused_key_list.append(key)
    for key in unused_key_list:
        del coreness_dict[key]

    return G_main, coreness_dict

### Node score

def node_score_based_on_ego_network(graph, weight="weight", radius = 1):
    """
    Measure node scores based on their ego networks, considering edge weights.

    Parameters:
        graph (networkx.Graph): Input graph with edge weights.
        weight (str): Attribute name for edge weights.

    Returns:
        dict: Dictionary with node scores {node: score}.
    """
    node_scores = {}

    for node in graph.nodes:
        # Get the ego network (subgraph containing node and its neighbors)
        ego_network = nx.ego_graph(graph, node, radius= radius)

        # Calculate the total weight of edges in the ego network
        total_edge_weight = sum(data[weight] for _, _, data in ego_network.edges(data=True))
        # # Calculate weighted degree centrality for the node in the ego network
        # weighted_degree = sum(data[weight] for _, _, data in ego_network.edges(node, data=True))

        # Example scoring function: Combine edge weights and degree centrality
        node_score = total_edge_weight #+ weighted_degree

        # Store the score
        node_scores[node] = node_score

    return node_scores





def determine_seed_pairs(G1, G2, validation_seed_pairs, top_k, method_name = 'pagerank'):
    
    correct_pairs = []
    count = 0 
    accuracy = 0.0 
    

    if method_name == 'pagerank':
        ### page rank centrality
        pagerank_centrality_G1 =  nx.pagerank(G1, alpha= 1)
        pagerank_centrality_G2 = nx.pagerank(G2, alpha= 1)
        
        ### Measure with PageRank Centrality
        all_pairs, correct_pairs, count, accuracy, count_right_dict, index_based_accuracy = get_seed_pairs(pagerank_centrality_G1, pagerank_centrality_G2, \
                                                     validation_seed_pairs, top_k = top_k )
        
        print('\nPage Rank:')
        print(f"Number of Extracted Seed Pairs: {count}\t Accuracy: {accuracy*100}")
        
        plot_first_k_elements_two_dicts(count_right_dict, index_based_accuracy,  k = 21, method= 'PageRank Centrality')
    
    
    elif method_name == 'eigen':
        ### eigenvector similarity
        eigenvector_centrality_G1 = nx.eigenvector_centrality(G1)
        eigenvector_centrality_G2 = nx.eigenvector_centrality(G2)
        
        ### Measure with EigenVector Centrality
        all_pairs, correct_pairs, count, accuracy, count_right_dict, index_based_accuracy = get_seed_pairs(eigenvector_centrality_G1, eigenvector_centrality_G2, \
                                                     validation_seed_pairs, top_k = top_k)
        print('\nEigen Vecotr:')
        print(f"Number of Extracted Seed Pairs: {count}\t Accuracy: {accuracy*100}")
        
        plot_first_k_elements_two_dicts(count_right_dict, index_based_accuracy,  k = 21, method='Eigenvector Centrality')
    
    elif method_name == 'degree':
        ### degree centrality
        degree_centrality_G1 = nx.degree_centrality(G1)
        degree_centrality_G2 = nx.degree_centrality(G2)
        
        ### Measure with Degree Centrality
        all_pairs, correct_pairs, count, accuracy , count_right_dict, index_based_accuracy = get_seed_pairs(degree_centrality_G1, degree_centrality_G2, \
                                                     validation_seed_pairs, top_k = top_k)
        print('\nDegree:')
        print(f"Number of Extracted Seed Pairs: {count}\t Accuracy: {accuracy*100}")
    
        plot_first_k_elements_two_dicts(count_right_dict, index_based_accuracy,  k = 21, method='Degree centrality')

    elif method_name == 'k-truss':
        
        G1_main, trussness_dict_G1 = get_all_edges_trussness_and_decomposition(G1)
        print()
        G2_main, trussness_dict_G2 = get_all_edges_trussness_and_decomposition(G2)
        print()

        node_scores_G1 = node_score_based_on_ego_network(G1_main, radius = 1)
        node_scores_G2 = node_score_based_on_ego_network(G2_main, radius = 1)

        ### Measure with trussness
        all_pairs, correct_pairs, count, accuracy, count_right_dict, index_based_accuracy = get_seed_pairs(node_scores_G1, node_scores_G2, \
                                                     validation_seed_pairs, top_k = top_k)
        print('\nEigen Vecotr:')
        print(f"Number of Extracted Seed Pairs: {count}\t Accuracy: {accuracy*100}")

        plot_first_k_elements_two_dicts(count_right_dict, index_based_accuracy,  k = 21, method='k-truss')
        
    elif method_name == 'k-core':
        G1_main, _ = get_all_edges_coreness_and_decomposition(G1)
        print()
        G2_main, _ = get_all_edges_coreness_and_decomposition(G2)
        print()
        
        node_scores_G1 = node_score_based_on_ego_network(G1_main, radius = 1)
        node_scores_G2 = node_score_based_on_ego_network(G2_main, radius = 1)
        
        ### Measure with trussness
        all_pairs, correct_pairs, count, accuracy, count_right_dict, index_based_accuracy = get_seed_pairs(node_scores_G1, node_scores_G2, \
                                                     validation_seed_pairs, top_k = top_k)
        print('\nEigen Vecotr:')
        print(f"Number of Extracted Seed Pairs: {count}\t Accuracy: {accuracy*100}")
        
        plot_first_k_elements_two_dicts(count_right_dict, index_based_accuracy,  k = 21, method='k-core')
        
    return  all_pairs, correct_pairs, count, accuracy 

def process_files_and_get_initial_data(file_path):
    print(file_path)
    # files = os.listdir(file_path)
    # files = sorted(files)
    

    
    # files
    validation_pair_file = ''
    validation_seed_pairs = list()
    
    ### Declare two graphs
    G1 = None
    G2 = None 
    
    if 'labeled_dev' in str(file_path):
        ### Declare two graphs
        files = os.listdir(file_path)
 
        
        for file in files:
            # print(file)
            if 'G1.edgelist' in file:
                G1 = nx.read_edgelist(file_path+'//'+ file)
            elif 'G2.edgelist' in file:
                G2 = nx.read_edgelist(file_path+'//'+ file)
            elif 'mapping' in file: 
                validation_pair_file = file 
    
        
        with open(file_path+'//'+ validation_pair_file, 'r') as seed_mapping_file:
            for line in seed_mapping_file:
                line_data = line.rstrip('\n')
                line_data = tuple(line_data.split(' '))
                
                u, v = line_data
                
                if u in G1 and u in G2:
                    validation_seed_pairs.append(line_data)
            
        return G1, G2, validation_seed_pairs
    

def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Seed Free Graph Deanonymization Script")
    parser.add_argument('--method_name', type=str, default='pagerank', \
                        help="Method name for centrality or strength measures. Options: 'eigen', 'pagerank', 'degree', 'k-truss', 'k-core'")
    parser.add_argument('--integrate_with_Seed_Based', type=int, choices=[0, 1], default= 1, \
                        help="Set 1 to integrate Seed Based Graph Deanonymization, otherwise 0 for only centrality measures.")
    parser.add_argument('--path', type=str, default= str(REPO_ROOT / 'data'), \
                        help="Path to the data directory holding labeled_dev/.")
    parser.add_argument('--out', type=str, default= str(REPO_ROOT / 'results'), \
                        help="Directory for result and final-mapping output files.")
    parser.add_argument('--number_of_top_index_pairs', type=int, default= 50, \
                        help="Before integrating Seed Based method how many top indices pairs you want to provide as given seed pairs")

    args = parser.parse_args()

    # Extract arguments
    method_name = args.method_name
    integrate_with_Seed_Based = args.integrate_with_Seed_Based
    path = args.path
    number_of_top_index_pairs = args.number_of_top_index_pairs
    # Connect with the file paths
    os.chdir(path)
    path = os.getcwd()
    print(f"Current Working Directory: {path}")
    file_path = os.path.join(path, 'labeled_dev')
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # Process files and get initial data
    G1, G2, validation_seed_pairs = process_files_and_get_initial_data(file_path)

    total_extracted_pairs = set()
    accuracy = 0.0
    

    if integrate_with_Seed_Based == 0:
        top_k = min([len(G1), len(G2)])
        print(f"Top-K: {top_k}")

        all_pairs, correct_pairs, count, method_accuracy = determine_seed_pairs(
            G1, G2, validation_seed_pairs, top_k, method_name=method_name
        )
        method_accuracy = method_accuracy * 100
        accuracy = method_accuracy

        common = correct_pairs  # Only for writing in final file
        total_extracted_pairs = set(all_pairs)
        number_of_top_index_pairs = 0

    elif integrate_with_Seed_Based == 1:
        
        ### Import Seed Free Graph Deanonymization from specific Directory
        import sys 
        sys.path.append(path)
        from  PAC_Project_1_Seed_Based import SeedFreeD 
        
        top_k = number_of_top_index_pairs

        all_pairs, correct_pairs, count, method_accuracy = determine_seed_pairs(
            G1, G2, validation_seed_pairs, top_k, method_name=method_name
        )
        method_accuracy = method_accuracy * 100
        training_seed_pairs = all_pairs[:number_of_top_index_pairs]

        # ECCE Threshold
        ecce_threshold = 0.5
        seedFreeAnon = SeedFreeD(G1, G2, training_seed_pairs, ecce_threshold=ecce_threshold)

        new_seed_pairs, seed_nodes_G1 = seedFreeAnon.iterativeOperation()
        total_extracted_pairs = set(new_seed_pairs).union(set(all_pairs[:number_of_top_index_pairs]))

        if len(total_extracted_pairs) == len(seed_nodes_G1):
            print(f"Correct Calculation: {len(seed_nodes_G1)}")
        else:
            print("Please Check Carefully")

        common = set(total_extracted_pairs).intersection(set(validation_seed_pairs))

        accuracy = len(common) * 100 / len(validation_seed_pairs)

    print(f"Accuracy: {accuracy}")
    
    with open(os.path.join(out_dir, 'result_'+str(number_of_top_index_pairs)+'_.txt'), 'w') as result_file: 
        result_file.write("Accuracy: "+str(accuracy)+"\n")
        result_file.write("# of Correct pairs: "+str(len(common))+"\n")
        result_file.write("Method Accuracy ({method_name}): "+str(method_accuracy)+"\n")
        result_file.close() 
    ### Write newly added seed pairs with given seed pairs in a new file
    with open(os.path.join(out_dir, 'seed_free_final_mapping_'+str(integrate_with_Seed_Based)+'.txt'), 'w') as final_seed_file: 
        for u, v in list(total_extracted_pairs):
            final_seed_file.write(u+' '+ v +'\n')
        final_seed_file.close()

if __name__ == "__main__":
    main()
    

    
    
