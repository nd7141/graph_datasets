## Graph Classification Data Sets

This repo contains manually curated list of graph datasets for evaluation graph classification methods. 
These data sets are results of removing isomorphic copies of graphs from the original data sets. There are at the moment 54 data sets. The code to generate data sets is available here (https://github.com/nd7141/iso_bias).

### Citation
If you found our work useful, please consider citing our work. 

@misc{ivanov2019understanding,
    title={Understanding Isomorphism Bias in Graph Data Sets},
    author={Sergei Ivanov and Sergei Sviridov and Evgeny Burnaev},
    year={2019},
    eprint={1910.12091},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}

### Getting graphs for a data set
All datasets are zipped. There is a class ``GraphDataset`` that extracts, transforms, and save graphs to necessary formats.


    dataset = GraphDataset()
    
    # extract dataset
    dataset_path = 'datasets/'
    d = 'MUTAG'
    output = 'compact/'
    dataset.extract_folder(dataset_path + d + '.zip', output)
  
After the dataset was extracted locally, we can read graphs as a list of ``GraphStruct`` object, where
each graph is a collection of nodes, edges, labels/attributes in simple python data structures (list of dict). 
``GraphStruct`` also allows creating networkx graphs.  
    
    # read graphs
    graphs = dataset.read_graphs(output + d + '/')
    
We can additionally save graphs in ``graphml`` format, which will preserve node/edge labels/attributes.
 
    # save graphml
    output = 'graphml/'
    dataset.save_graphs_graphml(graphs, output + d + '/')
    
We can also save graphs in ``edgelist`` format, which purely keeps topology of a graph. 

    # save edgelist
    output = 'edgelist/'
    dataset.save_graphs_edgelist(graphs, output + d + '/') 

## Data Set Stats
![image](https://user-images.githubusercontent.com/4607709/67501980-7c6b4980-f685-11e9-850a-eb7fcaa5d605.png)
