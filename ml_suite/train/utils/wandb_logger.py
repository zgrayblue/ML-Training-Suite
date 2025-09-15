"""
Wandb logger for handling all wandb-related logging functionality.

Author: Florian Wiesner
Date: 2025-04-07
"""

from typing import Any, Dict, Optional, Literal
from pathlib import Path

import wandb
from wandb.sdk.wandb_run import Run

from ml_suite.train.utils.logger import setup_logger


class WandbLogger:
    """A class to handle all wandb logging functionality with error handling.

    This class is responsible for initializing wandb, logging metrics, and handling
    any errors that might occur during logging. It ensures that training can continue
    even if wandb logging fails.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing wandb settings
    log_dir : Optional[Path], optional
        Directory to save wandb logs, by default None
        Usually to store the logs in the same directory as checkpoints
    """

    def __init__(
        self,
        wandb_config: dict,
        log_dir: Optional[Path] = None,
    ):
        self.config = wandb_config
        self.config["dir"] = log_dir
        self.logger = setup_logger(
            "WandbLogger",
        )
        self.run: Optional[Run] = None
        self._initialize_wandb()

    def _initialize_wandb(self) -> None:
        """Initialize wandb with error handling."""
        try:
            wandb_id = self.config.get("id", "test")
            project = self.config.get("project")
            entity = self.config.get("entity")
            tags = self.config.get("tags", [])
            notes = self.config.get("notes", "")
            dir = self.config.get("dir", None)
            wandb.login()
            self.run = wandb.init(
                project=project,
                entity=entity,
                config=self.config,
                id=wandb_id,
                dir=dir,
                tags=tags,
                notes=notes,
                resume="allow",
                settings=wandb.Settings(init_timeout=120),
            )
            self.logger.info("Successfully initialized wandb")
        except Exception as e:
            self.logger.error(f"Failed to initialize wandb: {str(e)}")
            self.run = None

    def log(self, data: Dict[str, Any], folder: str, commit: bool = True) -> None:
        """Log data to wandb with error handling.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary of data to log
        folder : str
            Wandb folder to log the data to
        commit : bool, optional
            Whether to commit the data immediately, by default True
        """
        if self.run is None:
            return

        try:
            # add folder to the dict strings
            data = {f"{folder}/{k}": v for k, v in data.items()}
            self.run.log(data, commit=commit)
        except Exception as e:
            self.logger.error(f"Failed to log data to wandb: {str(e)}")

    def watch(
        self,
        model: Any,
        criterion: Any,
        log: Literal["gradients", "parameters"] = "gradients",
        log_freq: int = 100,
    ) -> None:
        """Watch model parameters with error handling.

        Parameters
        ----------
        model : Any
            Model to watch
        criterion : Any
            Loss function
        log : str, optional
            What to log, by default "gradients"
        log_freq : int, optional
            How often to log, by default 100
        """
        if self.run is None:
            return

        try:
            self.run.watch(
                model,
                criterion=criterion,
                log=log,
                log_freq=log_freq,
            )
        except Exception as e:
            self.logger.error(f"Failed to watch model in wandb: {str(e)}")

    def update_config(self, data: Dict[str, Any]) -> None:
        """Update wandb config with error handling.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary of config values to update
        """
        if self.run is None:
            return

        try:
            self.run.config.update(data, allow_val_change=True)
        except Exception as e:
            self.logger.error(f"Failed to update wandb config: {str(e)}")

    def finish(self) -> None:
        """Finish the wandb run with error handling."""
        if self.run is None:
            return

        try:
            self.run.finish()
            self.logger.info("Successfully finished wandb run")
        except Exception as e:
            self.logger.error(f"Failed to finish wandb run: {str(e)}")

    # def log_predictions(
    #     self,
    #     image_path: Path,
    #     name_prefix: str,
    # ) -> None:
    #     """Log predictions to wandb with error handling.

    #     Parameters
    #     ----------
    #     image_path : Path
    #         Path to the images
    #     name_prefix : str
    #         Prefix for the image names
    #     """
    #     if self.run is None:
    #         return

    #     try:
    #         data = {}
    #         for image in image_path.glob("**/*.png"):
    #             img = Image.open(image)
    #             data[f"{name_prefix}/{image.name}"] = wandb.Image(
    #                 img, file_type="png", mode="RGB"
    #             )
    #         self.run.log(data)
    #     except Exception as e:
    #         self.logger.error(f"Failed to log predictions to wandb: {str(e)}")

