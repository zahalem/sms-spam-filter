import csv
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATA_DIR = Path(__file__).parent / "data"
DATA_FILE = DATA_DIR / "SMSSpamCollection"


def download(force: bool = False) -> Path:
    """Скачивает датасет SMS Spam Collection (UCI), если его ещё нет."""
    if DATA_FILE.exists() and not force:
        return DATA_FILE
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Скачиваю {URL}")
    with urlopen(URL, timeout=60) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    archive.extract("SMSSpamCollection", DATA_DIR)
    return DATA_FILE


def load(path: Path = DATA_FILE) -> pd.DataFrame:
    """Файл в формате "метка<TAB>текст". Возвращает label (1 = спам) и text."""
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"],
                     quoting=csv.QUOTE_NONE, encoding="utf-8")
    df["label"] = (df["label"] == "spam").astype(int)
    return df.dropna().drop_duplicates(subset="text").reset_index(drop=True)
