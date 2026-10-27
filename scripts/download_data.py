from pathlib import Path
from immune_response.data import download

if __name__ == "__main__":
    download(Path(__file__).resolve().parents[1] / "data")
