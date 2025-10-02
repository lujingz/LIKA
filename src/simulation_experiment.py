'''
Do simulation experiments for the method
'''
from method import *
from pipeline import pipeline
from utils import *

def KSEA(intensity_df, log_transform=False, network_df=None):
    # Get p-values
    logFC = {}
    intensity_df.columns = intensity_df.columns.str.lower()
    intensity_cols = get_intensity_columns(intensity_df)

    if log_transform:
        # 0 means missing
        intensity_df['mean'] = intensity_df[intensity_cols].apply(lambda x: x[x != 0].mean(), axis=1)
        intensity_df['mean'] = np.log2(intensity_df['mean']+1e-6)
    else:
        intensity_df['mean'] = intensity_df[intensity_cols].mean(axis=1)
    

    for index, row in intensity_df.iterrows():
        if row['name'] in logFC:
            logFC[row['name']] += row['mean'] * (2 * row['group'] - 1)
        else:
            logFC[row['name']] = row['mean'] * (2 * row['group'] - 1)
    logFC = {k:v for k,v in logFC.items() if not np.isnan(v)}

    # Get network data # TODO: allow the user to include more netwokrs than kinase-phosphosite network
    if network_df is not None: # self-defined network
        network_df.columns = network_df.columns.str.lower()
        network = nx.from_pandas_edgelist(network_df, source="from", target="to", create_using=my_network)
        for index, row in network_df.iterrows():
            network.nodes[row['to']]['Description'] = 'Phosphosite'
            network.nodes[row['from']]['Description'] = 'Kinase'
            network.nodes[row['to']]['p_value'] = logFC[row['to']]
        network.set_kinase_index()
        network.set_unmissing_neighbors_and_children()
    else: # use default network
        df = pd.read_csv('data/KSEA_dataset_processed.csv')
        df = df[df['to'].isin(logFC.keys())].reset_index(drop=True)
        network = nx.from_pandas_edgelist(df, source="from", target="to", create_using=my_network)
        for index, row in df.iterrows():
            network.nodes[row['to']]['Description'] = 'Phosphosite'
            network.nodes[row['to']]['p_value'] = logFC[row['to']]
            network.nodes[row['from']]['Description'] = 'Kinase'
        network.set_kinase_index()
        network.set_unmissing_neighbors_and_children()

    # get background mean and std
    logFC = list(logFC.values())
    background_mean = np.mean(logFC)
    background_std = np.std(logFC)

    kinase_scores = {}
    p_values = {}
    for kinase in network.get_kinase():
        kinase_scores[kinase] = 0
        num = 0
        for node in network.get_unmissing_children(kinase):
            kinase_scores[kinase] += network.nodes[node]['p_value']
            num += 1
        kinase_scores[kinase] /= num
        kinase_scores[kinase] = (kinase_scores[kinase] - background_mean) / (background_std/np.sqrt(num))
        p_value = 2 * (1 - stats.norm.cdf(abs(kinase_scores[kinase])))
        p_values[kinase] = p_value

    # get top 10 kinases
    top_kinases = sorted(kinase_scores.items(), key=lambda x: x[1], reverse=False)[:10]
    top_kinases = [kinase for kinase, score in top_kinases]
    results_df = pd.DataFrame({
        'Name': list(p_values.keys()),
        'p_value': list(p_values.values()),
        'Lower Bound': list(map(lambda x:-x, list(p_values.values())))
    })

    # print the kinases with p_value < 0.05
    # print(BH_(p_values, 0.05))

    return results_df, top_kinases

def simulation_experiment_1():
    '''
    Example 1: all substrates are only affected by one kinase
    '''
    from_ = list(map(lambda x:'K'+str(x//10), [i for i in range(100)])) # 10 kinases in total, each have 10 substrates
    to_ = list(map(lambda x:str(x), [i for i in range(100)])) # 100 substrates in total
    network_df = pd.DataFrame({
        'from': from_,
        'to': to_
    })
    # Generate simulation data
    n = 5  # number of intensity columns
    np.random.seed(42)  # for reproducibility
    data = []
    for name in range(100):
        # Group 0 line - always sample from N(0,1)
        row_group0 = {'name': str(name), 'group': 0}
        intensities_group0 = np.random.normal(0, 1, n)
        for i in range(n):
            row_group0[f'intensity_{i}'] = intensities_group0[i]
        data.append(row_group0)
        
        # Group 1 line - sample based on name value
        row_group1 = {'name': str(name), 'group': 1}
        if int(name) < 20:
            # Sample from N(1,1) for names < 20
            intensities_group1 = np.random.normal(2, 1, n)
        else:
            # Sample from N(0,1) for names >= 20
            intensities_group1 = np.random.normal(0, 1, n)
        
        for i in range(n):
            row_group1[f'intensity_{i}'] = intensities_group1[i]
        data.append(row_group1)
    print('abnormal kinases: ', [0,1])
    intensity_df = pd.DataFrame(data)
    results_df, top_kinases = KSEA(intensity_df, log_transform=False, network_df=network_df)
    results_df.to_csv('results/simulation_experiment_1_KSEA.csv', index=False)
    # plot_top_kinases(results_df, top_kinases)
    # Run pipeline
    network, rejection_set, results_df, top_kinases, _ = pipeline(intensity_df, log_transform=False, network_df=network_df, CI=0.2)
    visualize_rejection(network, rejection_set, 'Simulation Experiment 1', None)
    plot_top_kinases(results_df, top_kinases)
    results_df.to_csv('results/simulation_experiment_1_LIKA.csv', index=False)
    
def simulation_experiment_3():
    '''
    Example 3: substrates can be affected by multiple kinases (shared nodes)
    - Parents (kinases) have different numbers of children (substrates)
    - Children (substrates) have different numbers of parents (kinases)
    - Some parents are randomly selected to be 'abnormal'
    '''
    np.random.seed(42)  # for reproducibility
    
    # Create a more complex network structure
    # 8 kinases with varying numbers of substrates
    kinase_substrate_counts = [20, 20, 15, 15, 12, 11, 10, 8, 6, 4, 3, 2, 1]  # number of substrates per kinase
    total_substrates = 40  # total number of unique substrates
    
    # Create network connections
    from_ = []
    to_ = []
    
    # For each kinase, randomly assign substrates (allowing overlap)
    substrate_pool = list(range(total_substrates))
    
    for kinase_idx, substrate_count in enumerate(kinase_substrate_counts):
        kinase_name = f'K{kinase_idx}'
        # Randomly select substrates for this kinase (with replacement possible across kinases)
        selected_substrates = np.random.choice(substrate_pool, size=substrate_count, replace=False)
        
        for substrate_idx in selected_substrates:
            from_.append(kinase_name)
            to_.append(str(substrate_idx))
    
    network_df = pd.DataFrame({
        'from': from_,
        'to': to_
    })
    
    abnormal_kinases = ['K0', 'K2', 'K5', 'K8']
    print(f"Abnormal kinases: {abnormal_kinases}")
    
    # Get all substrates affected by abnormal kinases
    abnormal_substrates = set()
    for substrate in substrate_pool:
        father_kinases = set(network_df[network_df['to'] == str(substrate)]['from'])
        if len(father_kinases) > 0:
            p = len(father_kinases & set(abnormal_kinases)) / len(father_kinases)
            if np.random.uniform(0, 1) < p:
                abnormal_substrates.add(str(substrate))
    
    # print(f"Substrates affected by abnormal kinases: {sorted([int(x) for x in abnormal_substrates])}")
    
    # Generate simulation data
    n = 5  # number of intensity columns
    data = []
    
    for substrate_idx in range(total_substrates):
        substrate_name = str(substrate_idx)
        is_abnormal = substrate_name in abnormal_substrates
        
        # Group 0 line - always sample from N(0,1)
        row_group0 = {'name': substrate_name, 'group': 0}
        intensities_group0 = np.random.normal(0, 1, n)
        for i in range(n):
            row_group0[f'intensity_{i}'] = intensities_group0[i]
        data.append(row_group0)
        
        # Group 1 line - sample based on whether substrate is affected by abnormal kinases
        row_group1 = {'name': substrate_name, 'group': 1}
        if is_abnormal:
            # Sample from N(2,1) for substrates affected by abnormal kinases
            intensities_group1 = np.random.normal(2, 1, n)
        else:
            # Sample from N(0,1) for normal substrates
            intensities_group1 = np.random.normal(0, 1, n)
        
        for i in range(n):
            row_group1[f'intensity_{i}'] = intensities_group1[i]
        data.append(row_group1)

    intensity_df = pd.DataFrame(data)
    
    print("\n=== KSEA Method Results ===")
    results_df, top_kinases = KSEA(intensity_df, log_transform=False, network_df=network_df)
    # plot_top_kinases(results_df, top_kinases)
    results_df.to_csv('results/simulation_experiment_3_KSEA.csv', index=False)

    
    print("\n=== Pipeline Method Results ===")
    # Run pipeline
    network, rejection_set, results_df, top_kinases, _ = pipeline(intensity_df, log_transform=False, network_df=network_df, CI=0.2)
    visualize_rejection(network, rejection_set, 'Simulation Experiment 2', None)
    plot_top_kinases(results_df, top_kinases)
    results_df.to_csv('results/simulation_experiment_3_LIKA.csv', index=False)

    
    return network_df, intensity_df, abnormal_kinases 

def simulation_experiment_2():
    '''
    Example 2: kinases have non-overlapping substrates but varying numbers of substrates
    - Each substrate belongs to exactly one kinase (no sharing)
    - Different kinases have different numbers of substrates
    - Some kinases are randomly selected to be 'abnormal'
    '''
    np.random.seed(42)  # for reproducibility
    
    # Create kinases with varying numbers of substrates (non-overlapping)
    kinase_substrate_counts = [20, 20, 15, 15, 12, 11, 10, 8, 6, 4, 3, 2, 1]  # number of substrates per kinase
    total_substrates = sum(kinase_substrate_counts)  # total number of unique substrates
    
    print(f"Total kinases: {len(kinase_substrate_counts)}")
    print(f"Substrates per kinase: {kinase_substrate_counts}")
    print(f"Total substrates: {total_substrates}")
    
    # Create network connections (non-overlapping assignment)
    from_ = []
    to_ = []
    
    substrate_idx = 0
    for kinase_idx, substrate_count in enumerate(kinase_substrate_counts):
        kinase_name = f'K{kinase_idx}'
        
        # Assign consecutive substrates to this kinase (no overlap)
        for _ in range(substrate_count):
            from_.append(kinase_name)
            to_.append(str(substrate_idx))
            substrate_idx += 1
    
    network_df = pd.DataFrame({
        'from': from_,
        'to': to_
    })
    
    abnormal_kinases = ['K0', 'K2', 'K5', 'K8']
    print(f"\nAbnormal kinases: {abnormal_kinases}")
    
    # Get all substrates affected by abnormal kinases
    abnormal_substrates = set()
    for substrate in range(total_substrates):
        father_kinases = set(network_df[network_df['to'] == str(substrate)]['from'])
        if len(father_kinases) > 0:
            p = len(father_kinases & set(abnormal_kinases)) / len(father_kinases)
            if np.random.uniform(0, 1) < p:
                abnormal_substrates.add(str(substrate))
    print(f"Number of substrates affected by abnormal kinases: {len(abnormal_substrates)}")
    
    # Show which kinase sizes are abnormal
    abnormal_kinase_sizes = []
    for kinase in abnormal_kinases:
        size = len(network_df[network_df['from'] == kinase])
        abnormal_kinase_sizes.append(size)
    print(f"Sizes of abnormal kinases: {abnormal_kinase_sizes}")
    
    # Generate simulation data
    n = 5  # number of intensity columns
    data = []
    
    for substrate_idx in range(total_substrates):
        substrate_name = str(substrate_idx)
        is_abnormal = substrate_name in abnormal_substrates
        
        # Group 0 line - always sample from N(0,1)
        row_group0 = {'name': substrate_name, 'group': 0}
        intensities_group0 = np.random.normal(0, 1, n)
        for i in range(n):
            row_group0[f'intensity_{i}'] = intensities_group0[i]
        data.append(row_group0)
        
        # Group 1 line - sample based on whether substrate is affected by abnormal kinases
        row_group1 = {'name': substrate_name, 'group': 1}
        if is_abnormal:
            # Sample from N(2,1) for substrates affected by abnormal kinases
            intensities_group1 = np.random.normal(2, 1, n)
        else:
            # Sample from N(0,1) for normal substrates
            intensities_group1 = np.random.normal(0, 1, n)
        
        for i in range(n):
            row_group1[f'intensity_{i}'] = intensities_group1[i]
        data.append(row_group1)

    intensity_df = pd.DataFrame(data)
    
    print("\n=== KSEA Method Results ===")
    results_df, top_kinases = KSEA(intensity_df, log_transform=False, network_df=network_df)
    plot_top_kinases(results_df, top_kinases)
    results_df.to_csv('results/simulation_experiment_2_KSEA.csv', index=False)
    
    print("\n=== Pipeline Method Results ===")
    # Run pipeline
    network, rejection_set, results_df, top_kinases, _ = pipeline(intensity_df, log_transform=False, network_df=network_df, CI=0.2)
    visualize_rejection(network, rejection_set, 'Simulation Experiment 3', None)
    plot_top_kinases(results_df, top_kinases)
    results_df.to_csv('results/simulation_experiment_2_LIKA.csv', index=False)
    
    return network_df, intensity_df, abnormal_kinases


if __name__ == "__main__":
    print("=== Running Simulation Experiment 1 ===")
    simulation_experiment_1()
    
    print("\n" + "="*50)
    print("=== Running Simulation Experiment 2 ===")
    network_df, intensity_df, abnormal_kinases = simulation_experiment_2()


    print("\n" + "="*50)
    print("=== Running Simulation Experiment 3 ===")
    network_df3, intensity_df3, abnormal_kinases3 = simulation_experiment_3()

