from src.data.yahoo import download_universe


def main():
    status = download_universe(
        force=False 
    )

    print("\n=== DOWNLOAD SUMMARY ===")
    print(
        status["status"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()