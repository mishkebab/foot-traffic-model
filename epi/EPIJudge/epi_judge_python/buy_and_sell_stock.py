from typing import List

from test_framework import generic_test


def buy_and_sell_stock_once(prices: List[float]) -> float:
    # TODO - you fill in here.
    min_price = prices[0]
    max_price = prices[0]
    max_profit = 0
    
    for right in range(1, len(prices)):
        if prices[right] > max_price:
            max_price = prices[right]
            max_profit = max(max_profit, (max_price - min_price))
        elif prices[right] < min_price:
            min_price = prices[right]
            max_price = prices[right]
    
    return max_profit


if __name__ == '__main__':
    exit(
        generic_test.generic_test_main('buy_and_sell_stock.py',
                                       'buy_and_sell_stock.tsv',
                                       buy_and_sell_stock_once))
