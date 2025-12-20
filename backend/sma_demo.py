from .signals import load_price_history, add_sma_crossover_signal


def main():
    df = load_price_history("AAPL")
    df = add_sma_crossover_signal(df, short_window=50, long_window=200)
    print(df.head(40))  

if __name__ == "__main__":
    main()