from pathlib import Path

from src.data.yahoo import download_ticker


def main():
    output_dir = Path("data/raw/yahoo")

    result = download_ticker(
        ticker="AAPL",
        output_dir=output_dir,
        start="2008-01-01",
    )

    print("\nDownload result:")
    print(result)


if __name__ == "__main__":
    main()