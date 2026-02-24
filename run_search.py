#!/usr/bin/env python3
"""
Entrypoint: natural language query → find problem → print integer program.

  python run_search.py "minimize cost of opening warehouses and assigning customers"
  python run_search.py "knapsack" 2
"""
from retrieval.search import main

if __name__ == "__main__":
    main()
