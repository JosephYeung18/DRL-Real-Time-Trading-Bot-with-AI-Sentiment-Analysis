"""
Trading GUI Launcher
Launch the trading visualization GUI
"""

import sys
import os
import subprocess

def main():
    print("=" * 60)
    print("QCS Trading GUI Launcher")
    print("=" * 60)
    print("\nStarting Trading Visualization...")
    print("\nMake sure you have run a backtest first to generate results.")
    print()

    # Try to import GUI
    try:
        from trading_gui import TradingVisualizationGUI
        print("GUI module loaded successfully!")
    except ImportError as e:
        print(f"Error: Could not load GUI module: {e}")
        print("\nPlease ensure trading_gui.py is in the same directory.")
        return

    # Launch GUI
    try:
        app = TradingVisualizationGUI()
        app.run()
    except Exception as e:
        print(f"Error running GUI: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
