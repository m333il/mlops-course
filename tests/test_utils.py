import pytest
import logging
import random
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from src.utils import set_global_seed

def test_set_global_seed_sets_random_seed():
    seed = 42
    set_global_seed(seed)
    
    random_val1 = random.random()
    
    set_global_seed(seed)
    random_val2 = random.random()
    
    assert random_val1 == random_val2


def test_set_global_seed_sets_numpy_seed():
    seed = 123
    set_global_seed(seed)
    
    array1 = np.random.rand(5)
    
    set_global_seed(seed)
    array2 = np.random.rand(5)
    
    np.testing.assert_array_equal(array1, array2)


def test_set_global_seed_sets_torch_seed():
    seed = 789
    set_global_seed(seed)
    
    tensor1 = torch.rand(5)
    
    set_global_seed(seed)
    tensor2 = torch.rand(5)
    
    assert torch.equal(tensor1, tensor2)


def test_set_global_seed_reproducibility_across_operations():
    seed = 999

    set_global_seed(seed)
    random_vals1 = [random.random() for _ in range(3)]
    numpy_array1 = np.random.rand(5)
    torch_tensor1 = torch.rand(5)
    
    set_global_seed(seed)
    random_vals2 = [random.random() for _ in range(3)]
    numpy_array2 = np.random.rand(5)
    torch_tensor2 = torch.rand(5)
    
    assert random_vals1 == random_vals2
    np.testing.assert_array_equal(numpy_array1, numpy_array2)
    assert torch.equal(torch_tensor1, torch_tensor2)
