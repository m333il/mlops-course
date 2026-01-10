import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
import hashlib


def test_get_dvc_data_hash_no_file():
    from train import get_dvc_data_hash
    
    with patch('builtins.open', side_effect=FileNotFoundError()):
        result = get_dvc_data_hash()
        assert result == 'no_dvc_lock'


def test_get_git_commit_success():
    from train import get_git_commit
    
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.stdout = 'abc12345678\n'
        mock_run.return_value.returncode = 0
        
        result = get_git_commit()
        assert result == 'abc12345'
        assert len(result) == 8


def test_get_git_commit_failure():
    from train import get_git_commit
    
    with patch('subprocess.run', side_effect=Exception('git not found')):
        result = get_git_commit()
        assert result == 'unknown'


def test_log_params_from_config(mock_config):
    from train import log_params_from_config
    
    with patch('mlflow.log_params') as mock_log_params:
        log_params_from_config(mock_config)
        
        mock_log_params.assert_called_once()
        logged_params = mock_log_params.call_args[0][0]
        
        assert logged_params['model_name'] == mock_config['model']['name']
        assert logged_params['learning_rate'] == mock_config['training']['learning_rate']
        assert logged_params['batch_size'] == mock_config['training']['batch_size']
        assert logged_params['num_epochs'] == mock_config['training']['num_train_epochs']
        assert logged_params['seed'] == mock_config['run']['seed']


def test_log_tags():
    from train import log_tags
    
    with patch('train.get_dvc_data_hash', return_value='abc123'):
        with patch('train.get_git_commit', return_value='def456'):
            with patch('mlflow.set_tag') as mock_set_tag:
                log_tags()
                
                calls = mock_set_tag.call_args_list
                tag_names = [call[0][0] for call in calls]
                
                assert 'dvc_data_hash' in tag_names
                assert 'git_commit' in tag_names
                assert 'mlflow.runName' in tag_names
