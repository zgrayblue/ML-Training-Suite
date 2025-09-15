from unittest.mock import patch, MagicMock
from ml_suite.train.utils.wandb_logger import WandbLogger


class TestWandbLogger:
    @patch('vphysics.train.utils.wandb_logger.wandb')
    def test_wandb_logger_initialization_success(self, mock_wandb):
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        config = {
            "project": "test_project",
            "entity": "test_entity",
            "id": "test_id",
            "tags": ["test"],
            "notes": "test notes"
        }

        logger = WandbLogger(config)

        assert logger.run is mock_run
        mock_wandb.login.assert_called_once()
        mock_wandb.init.assert_called_once()

    @patch('vphysics.train.utils.wandb_logger.wandb')
    def test_wandb_logger_initialization_failure(self, mock_wandb):
        mock_wandb.init.side_effect = Exception("Connection failed")

        config = {"project": "test_project"}

        logger = WandbLogger(config)

        assert logger.run is None

    @patch('vphysics.train.utils.wandb_logger.wandb')
    def test_log_data_success(self, mock_wandb):
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        logger = WandbLogger({"project": "test"})

        data = {"metric1": 1.0, "metric2": 2.0}
        logger.log(data, folder="train")

        expected_data = {"train/metric1": 1.0, "train/metric2": 2.0}
        mock_run.log.assert_called_once_with(expected_data, commit=True)

    @patch('vphysics.train.utils.wandb_logger.wandb')
    def test_log_data_no_run(self, mock_wandb):
        mock_wandb.init.side_effect = Exception("Failed")

        logger = WandbLogger({"project": "test"})
        logger.log({"metric": 1.0}, "train")

        # Should not raise error when run is None