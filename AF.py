import math
import subprocess
import sys
import os
import shutil
import time
import networkx as nx
import pickle

import numpy as np
from numpy.linalg import LinAlgError
from sklearn import preprocessing
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def write_features(file, solution_list):
    with open(file + '_features.pkl', 'wb') as f:
        pickle.dump(solution_list, f)


def read_features(file):
    with open(file + '_features.pkl', 'rb') as f:
        solution_list = pickle.load(f)
    return solution_list


def write_solutions(file, solution_list):
    with open(file + '.pkl', 'wb') as f:
        pickle.dump(solution_list, f)


def read_solutions(file):
    with open(file + '.pkl', 'rb') as f:
        solution_list = pickle.load(f)
    return solution_list


def build_di_graph(file):
    di_graph = nx.DiGraph()
    with open(file) as f:
        af = f.read()
    f.close()
    graph = af.splitlines()

    for node in graph:
        node = node.replace('.', '')
        if 'arg(' in node:
            argument = node.replace('arg(', '').replace(')', '')
            di_graph.add_node(argument)
        elif 'att(' in node:
            attack = node.replace('att(', '').replace('', '').replace(')', '')
            attacking_node, attacked_node = attack.split(',')
            di_graph.add_edge(attacking_node, attacked_node)
    return di_graph


def parse_solutions(file, problem, file_solution):
    try:
        # parse labels from file
        f = open(file_solution)
        sol_text = f.read()
        sol_yes = sol_text.split("\n")
        solution = {}
        # build complete solution
        af = build_di_graph(file)
        for node in af.nodes:
            if node in sol_yes:
                solution[node] = "YES"
            else:
                solution[node] = "NO"
        return solution
    except FileNotFoundError:
        return None


def calculate_features(file, di_graph):
    # add features for training
    feature_dict = {
        'degree': {},
        'katz_centrality': {},
        'page_rank': {},
        'closeness_centrality': {},
        'betweenness_centrality': {},
        'no_of_sccs': {},
        'scc_size': {},
        'strong_connectivity': {},
        'symmetry': {},
        'asymmetry': {},
        'irreflexive': {},
        'attacks_all_others': {},
        'avg_degree': {},
        'aperiodicity': {}
    }
    try:
        feature_dict = read_features(file)
        #raise FileNotFoundError
        # print('read features')

    except (OSError, IOError, FileNotFoundError):
        # add degree centrality
        in_deg = di_graph.in_degree
        out_deg = di_graph.out_degree

        # add katz centrality
        # Calculate largest eigenvalue of di_graph
        largest_eigval =  max(nx.adjacency_spectrum(di_graph))
        lambda_max = np.real(largest_eigval)
        # Set alpha smaller than 1/largest_eigval or 0.1 if all eigenvals are 0
        if lambda_max == 0 :
            alpha = 0.1
        else:
            alpha = (1 / lambda_max)*0.9
        try:
            in_katz = nx.katz_centrality_numpy(di_graph, alpha=alpha)
        except LinAlgError:
            print("execption alpha =", alpha)
            in_katz = nx.katz_centrality_numpy(di_graph, alpha=0.9)
        try:
            out_katz = nx.katz_centrality_numpy(di_graph.reverse(), alpha=alpha)
        except LinAlgError:
            print("execption alpha =", alpha)
            out_katz = nx.katz_centrality_numpy(di_graph.reverse(), alpha=0.9)

        has_nan = any(math.isnan(score) for score in in_katz.values())

        if has_nan:
            # If there are NaN values, set all scores to 0
            print("There are NaN values in the katz centrality scores. Setting all values to 0")

        # add page_rank
        in_page = nx.pagerank_numpy(di_graph)
        out_page = nx.pagerank_numpy(di_graph.reverse())

        # add closeness centrality
        in_closeness = nx.closeness_centrality(di_graph)
        out_closeness = nx.closeness_centrality(di_graph.reverse())

        # add Betweenness centrality
        betweenness = nx.betweenness_centrality(di_graph)
        has_nan = any(math.isnan(score) for score in betweenness.values())

        if has_nan:
            # If there are NaN values, set all scores to 0
            betweenness = {node: 0 for node in di_graph.nodes()}
            print("There are NaN values in the betweenness centrality scores. Setting all values to 0")

        # add SCC membership
        # Extract all strongly connected components from the di_graph
        scc_list = list(nx.strongly_connected_components(di_graph))

        # Assign SCC membership and size values to each node
        for i, scc in enumerate(scc_list):
            for node in scc:
                feature_dict['scc_size'][node] = [len(scc)]

        # Strong connectivity
        strong_connectivity = nx.is_strongly_connected(di_graph)

        # Doumbouya

        # Symmetry
        symmetry = is_symmetric(di_graph)

        # Asymmetry
        asymmetry = is_asymmetric(di_graph)

        # Irreflexitivity
        irreflexive = is_irreflexive(di_graph)

        # Attacks all others
        for node in di_graph.nodes():
            feature_dict["attacks_all_others"][node] = [
                int((di_graph.out_degree(node) == di_graph.number_of_nodes() - 1))]

        # Vallati

        # number of SCCs
        no_of_sccs = nx.number_strongly_connected_components(di_graph)

        # average degree
        avg_degree = calculate_average_degree(di_graph)
        if (avg_degree) < 0:
            print("avg unter 0: " + file)

        # aperiodicity
        aperiodicity = nx.is_aperiodic(di_graph)

        # add features to feature_dict
        for node in di_graph:
            if (in_deg[node] < 0) or (out_deg[node] < 0):
                print("degree< 0: " + file)
            feature_dict["degree"][node] = (in_deg[node], out_deg[node])
            feature_dict["katz_centrality"][node] = (in_katz[node], out_katz[node])
            feature_dict["page_rank"][node] = (in_page[node], out_page[node])
            if in_closeness[node] < 0 or out_closeness[node] < 0:
                print("closeness< 0: " + file)
            feature_dict["closeness_centrality"][node] = (in_closeness[node], out_closeness[node])
            if betweenness[node] < 0:
                print("betweenness< 0: " + file)
            feature_dict["betweenness_centrality"][node] = [betweenness[node]]
            if int(strong_connectivity) < 0:
                print("strong_connectivity< 0: " + file)
            feature_dict["strong_connectivity"][node] = [int(strong_connectivity)]
            if int(symmetry) < 0:
                print("symmetry< 0: " + file)
            feature_dict["symmetry"][node] = [int(symmetry)]
            if int(asymmetry) < 0:
                print("asymmetry < 0: " + file)
            feature_dict["asymmetry"][node] = [int(asymmetry)]
            if int(irreflexive) < 0:
                print("irreflexive< 0: " + file)
            feature_dict["irreflexive"][node] = [int(irreflexive)]
            feature_dict["no_of_sccs"][node] = [no_of_sccs]
            feature_dict["avg_degree"][node] = [avg_degree]
            if int(aperiodicity) < 0:
                print("aperiodicity< 0: " + file)
            feature_dict["aperiodicity"][node] = [int(aperiodicity)]
        write_features(file, feature_dict)
        #print('wrote features for ' + file)
    return feature_dict


def is_symmetric(G):

    # Check if the graph is weakly connected
    if not nx.is_weakly_connected(G):
        return False

    # Check if the graph is undirected
    for edge in G.edges():
        if not G.has_edge(edge[1], edge[0]):
            return False

    # If all edges in the graph are bidirectional, the graph is symmetric
    return True


def is_asymmetric(G):

    # Check if the graph is weakly connected
    if not nx.is_weakly_connected(G):
        return False

    # Check if for every pair of nodes (u, v) in the graph, if there is an edge from u to v,
    # then there shouldn't be an edge from v to u, and vice versa.
    for u, v in G.edges():
        if G.has_edge(v, u) or G.has_edge(u, v):
            return False

    # graph is asymmetric.
    return True


def is_irreflexive(G):

    # Check if the graph has any self-loops
    for node in G.nodes():
        if G.has_edge(node, node):
            return False

    # If the graph has no self-loops, it is irreflexive
    return True


def calculate_average_degree(G):

    degrees = dict(G.degree())
    sum_degrees = sum(degrees.values())
    num_nodes = len(G.nodes())

    # Calculate the average degree
    if num_nodes > 0:
        avg_degree = sum_degrees / num_nodes
        return avg_degree
    else:
        return 0


def tgf_to_apx(file):
    f_ext = 'apx' if '.apx' in file else 'tgf'
    if f_ext == "tgf":
        with open(file) as f:
            af = f.read()
        f.close()
        graph = af.splitlines()
        graph = [(f"arg({n})." if ' ' not in n else f"att({n.split()[0]},{n.split()[1]}).") for n in graph if n != '#']
        head, tail = os.path.split(file)
        tail = tail.replace("tgf", 'apx')
        f_name = head + '/' + tail
        f_apx = open(f_name, "w")
        for item in graph:
            f_apx.write("%s\n" % item)
        f_apx.close
        # remove tgf file
        os.remove(file)
        return f_apx.name
    else:
        return file


def solve(file, di_graph, timeout, solver='ArgSemSAT'):
    problems = ['DC-PR', 'DC-CO', 'DC-ST', 'DC-GR', 'DS-PR', 'DS-CO', 'DS-ST', 'DS-GR']
    solutions = {}
    solution_parsed = False
    head, tail = os.path.split(file)
    timeout_timer = timeout
    timeout = False

    # get file extension
    f_ext = 'apx' if '.apx' in file else 'tgf'
    # solve AF for every problem
    try:
        solutions = read_solutions(file)
        # print('read solutions')
    except (OSError, IOError, FileNotFoundError) as e:
        for problem in problems:
            solution = {}
            # try to parse solution
            head_solution = head + '/solutions'
            file_solution = head_solution + "/" + tail + "_" + problem + ".txt"
            parsed_solution = parse_solutions(file, problem, file_solution)
            if parsed_solution is not None:
                solution = parsed_solution
                print("existing solution parsed for: " + tail + " " + problem)
                solution_parsed = True
            cpu_time_nodes = {}
            time_to_solve_graph = {}

            if not solution_parsed:
                print('solving ', file, ' for problem ', problem)
                print("\n")
                time_to_solve_graph = 0
                i = 1
                for node in di_graph.nodes:
                    # check if timeout conditions are met
                    if time_to_solve_graph >= timeout_timer:
                        timeout = True
                        break
                    # clear previous percentage output
                    sys.stdout.write("\033[F")
                    sys.stdout.write("\033[F")
                    sys.stdout.write("\033[K")
                    print('solving node ', node)
                    start = time.perf_counter_ns()
                    out = subprocess.run(
                        './' + solver + ' -p ' + problem + ' -a ' + node + ' -fo ' + f_ext + ' -f ' + file,
                        shell=True, capture_output=True, text=True)
                    end = time.perf_counter_ns()
                    solution[node] = out.stdout.strip()
                    progress = int(i / len(di_graph.nodes) * 100)
                    print(progress, '% solved')
                    i = i + 1
                    time_to_solve_node = end - start
                    # computation time in millisec
                    cpu_time_nodes[node] = round(time_to_solve_node / 1000000, 5)
                    time_to_solve_graph += round(time_to_solve_node / 1000000, 5)
                if timeout:
                    break
                # write solution to txt file
                try:
                    os.mkdir(head + "/solutions")
                except (OSError) as e:
                    print("")
                    # folder exists already
                f_name = head + "/solutions/" + tail + "_" + problem + ".txt"
                f = open(f_name, "x")
                for node in solution:
                    if solution[node] == "YES":
                        f.write(node)
                        f.write("\n")
                f.close()
            if timeout:
                break
            # add solution to all solutions for graph
            solutions[problem] = [solution, cpu_time_nodes, time_to_solve_graph]
            # reset solution_parsed for next problem
            solution_parsed = False
        if timeout:
            print("timeout for graph: " + tail)
            # move file to timeout folder
            try:
                os.mkdir(head + "/timeout")
            except (OSError) as e:
                print("")
                # folder exists already
            src = head + "/" + tail
            dest = head + "/timeout/" + tail
            shutil.move(src, dest)
            return
        write_solutions(file, solutions)
        print('wrote solutions')
    # solutions_overview(solutions)
    return solutions


class AF:

    def __init__(self, file, timeout, solutions=None):
        file_apx = tgf_to_apx(file)
        self.file_path = file_apx
        self.di_graph = build_di_graph(file_apx)
        self.solutions_dict = solve(file_apx, self.di_graph, timeout)
        self.feature_dict = calculate_features(file_apx, self.di_graph)

    def print_solutions(self):
        sols = self.solutions_dict
        for sem in sols:
            print('solutions for semantics: ', sem)
            print(sols[sem][0])

    def solutions_overview(self):
        overview = {}
        sems = {}
        sol_dict = self.solutions_dict
        overview['path'] = self.file_path
        for sem in sol_dict:
            details = {}
            try:
                # CPU-Time solution
                details['cpu_time_solution'] = sol_dict[sem][2]
                # avg CPU-Time per node
                time_per_node = dict(sol_dict[sem][1])
                details['average_time'] = sum(time_per_node.values()) / len(time_per_node)
                # fastest node
                details['min_time_node'] = [min(time_per_node.values()), min(time_per_node, key=time_per_node.get)]
                # slowest node
                details['max_time_node'] = [max(time_per_node.values()), max(time_per_node, key=time_per_node.get)]
            except ZeroDivisionError:
                # set all solution calculation times to zero for provided solutions
                details['cpu_time_solution'] = 0
                details['average_time'] = 0
                details['min_time_node'] = [0, 0]
                details['max_time_node'] = [0, 0]
            # Yes and no nodes
            sol_list = list(sol_dict[sem][0].values())
            no = sol_list.count('NO')
            yes = sol_list.count('YES')
            details['yes_no_nodes'] = [yes, no]
            sems[sem] = details
        overview['semantics'] = sems
        # Total nodes
        overview['nodes'] = len(self.di_graph.nodes)
        # Total attacks
        overview['attacks'] = len(self.di_graph.edges)
        return overview

    def print_solutions_overview(self):
        sol_dict = self.solutions_overview()
        print('Information for: ', sol_dict['path'])
        for sem in sol_dict['semantics']:
            # CPU-Time solution
            details = sol_dict['semantics'][sem]
            print("Time to solve for ", sem, ': ', details['cpu_time_solution'], " milliseconds")
            # avg CPU-Time per node
            print("Average time to solve node for ", sem, ': ', details['average_time'], " milliseconds")
            # fastest node
            print("Minimum time to solve node for ", sem, ': ', details['min_time_node'][0], " milliseconds. Node: ",
                  details['min_time_node'][1])
            # slowest node
            print("Maximum time to solve node for ", sem, ': ', details['max_time_node'][0], " milliseconds. Node: ",
                  details['max_time_node'][1])
            # Yes and no nodes
            print(sem, ': YES nodes: ', details['yes_no_nodes'][0], ' NO nodes: ', details['yes_no_nodes'][1])
        # Total nodes
        print("Number of nodes: ", sol_dict['nodes'])
        # Total attacks
        print("Number of attacks: ", sol_dict['attacks'])
