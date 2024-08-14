from pathlib import Path


class AbsSave:
    """
    A base class for managing directory creation and handling file saving/loading.
    """

    def __init__(self, output_dir: str) -> None:
        """
        Initializes the AbsSave object and ensures the output directory exists.

        Args:
            output_dir (str): The path to the directory where output files will be saved.
        """
        self.output_dir = Path(output_dir)
        self.create_directory(self.output_dir)

    @staticmethod
    def create_directory(dir_name: Path) -> None:
        """
        Creates a directory at the specified path if it does not already exist.

        Args:
            dir_name (Path): The path to the directory to be created.

        Notes:
            If the directory already exists, this method will not raise an exception.
            It will simply indicate that the directory already exists.
        """
        print(
            f"\n{dir_name} directory {'created' if not dir_name.exists() else 'already exists'}"
        )
        dir_name.mkdir(parents=True, exist_ok=True)
