

from dataset.parallel_bench import ParallelBench


def load_dataset(dataset_name, **kwargs):
    """
    Load a dataset by name with an optional number of samples.
    
    Args:
        dataset_name (str): Name of the dataset to load.
        num_samples (int, optional): Number of samples to load from the dataset.
        
    Returns:
        BaseDataset: An instance of the loaded dataset.
    """
    if dataset_name == "parallel_bench":
        return ParallelBench(**kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")