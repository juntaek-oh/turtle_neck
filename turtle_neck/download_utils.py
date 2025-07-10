from os import PathLike
from pathlib import Path


class DownloadHelper:
    def download_file(
        self,
        url: PathLike,
        filename: PathLike = None,
        directory: PathLike = None,
        show_progress: bool = True,
    ) -> PathLike:
        """
        Download a file from a url and save it to the local filesystem. The file is saved to the
        current directory by default, or to `directory` if specified. If a filename is not given,
        the filename of the URL will be used.

        :param url: URL that points to the file to download
        :param filename: Name of the local file to save. Should point to the name of the file only,
                        not the full path. If None the filename from the url will be used
        :param directory: Directory to save the file to. Will be created if it doesn't exist
                        If None the file will be saved to the current working directory
        :param show_progress: If True, show an TQDM ProgressBar
        :param silent: If True, do not print a message if the file already exists
        :param timeout: Number of seconds before cancelling the connection attempt
        :return: path to downloaded file
        """
        import urllib.parse

        import requests
        from tqdm import tqdm

        filename = filename or Path(urllib.parse.urlparse(url).path).name
        chunk_size = 16384  # make chunks bigger so that not too many updates are triggered for Jupyter front-end

        filename = Path(filename)
        if len(filename.parts) > 1:
            raise ValueError(
                "`filename` should refer to the name of the file, excluding the directory. "
                "Use the `directory` parameter to specify a target directory for the downloaded file."
            )

        filepath = Path(directory) / filename if directory is not None else filename
        if filepath.exists():
            return filepath.resolve()

        # create the directory if it does not exist, and add the directory to the filename
        if directory is not None:
            Path(directory).mkdir(parents=True, exist_ok=True)

        try:
            response = requests.get(
                url=url, headers={"User-agent": "Mozilla/5.0"}, stream=True
            )
            response.raise_for_status()
        except (
            requests.exceptions.HTTPError
        ) as error:  # For error associated with not-200 codes. Will output something like: "404 Client Error: Not Found for url: {url}"
            raise Exception(error) from None
        except requests.exceptions.Timeout:
            raise Exception(
                "Connection timed out. If you access the internet through a proxy server, please "
                "make sure the proxy is set in the shell from where you launched Jupyter."
            ) from None
        except requests.exceptions.RequestException as error:
            raise Exception(f"File downloading failed with error: {error}") from None

        # download the file if it does not exist
        filesize = int(response.headers.get("Content-length", 0))
        if not filepath.exists():
            with tqdm(
                total=filesize,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=str(filename),
                disable=not show_progress,
            ) as progress_bar:
                with open(filepath, "wb") as file_object:
                    for chunk in response.iter_content(chunk_size):
                        file_object.write(chunk)
                        progress_bar.update(len(chunk))
                        progress_bar.refresh()
        else:
            print(f"'{filepath}' already exists.")

        response.close()

        return filepath.resolve()
