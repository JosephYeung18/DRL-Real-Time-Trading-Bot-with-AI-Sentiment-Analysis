"""
QCS Trading Visualization GUI
Real-time visualization of trading backtest results
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

class TradingVisualizationGUI:
    """
    GUI application for visualizing trading backtest results
    """

    def __init__(self, root=None):
        if root is None:
            self.root = tk.Tk()
        else:
            self.root = root

        self.root.title("QCS Trading Visualization")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        self.data_dir = "eurusd_backtest_result_sentiment"
        self.results_data = None

        self.setup_ui()

        # Auto-load data if exists
        if os.path.exists(self.data_dir):
            self.load_data()

    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill=tk.X)

        title_label = tk.Label(
            title_frame,
            text="QCS Trading Visualization",
            font=("Arial", 20, "bold"),
            fg="white",
            bg='#2c3e50'
        )
        title_label.pack(pady=15)

        # Control panel
        control_frame = tk.Frame(self.root, bg='#f0f0f0')
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        # Data directory selector
        tk.Label(control_frame, text="Results Directory:", bg='#f0f0f0').grid(row=0, column=0, padx=5)

        self.dir_entry = tk.Entry(control_frame, width=40)
        self.dir_entry.insert(0, self.data_dir)
        self.dir_entry.grid(row=0, column=1, padx=5)

        tk.Button(control_frame, text="Browse", command=self.browse_directory).grid(row=0, column=2, padx=5)
        tk.Button(control_frame, text="Load Data", command=self.load_data, bg='#3498db', fg='white').grid(row=0, column=3, padx=5)

        # Summary panel
        summary_frame = tk.Frame(self.root, bg='white', bd=2, relief=tk.RIDGE)
        summary_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(summary_frame, text="Performance Summary", font=("Arial", 12, "bold"), bg='white').pack(pady=5)

        self.summary_label = tk.Label(summary_frame, text="Load data to see summary", font=("Arial", 10), bg='white')
        self.summary_label.pack(pady=5)

        # Main visualization area
        visualization_frame = tk.Frame(self.root, bg='white', bd=2, relief=tk.RIDGE)
        visualization_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Tab control for different views
        self.notebook = ttk.Notebook(visualization_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Portfolio Value
        self.portfolio_tab = tk.Frame(self.notebook)
        self.notebook.add(self.portfolio_tab, text="Portfolio Value")

        # Tab 2: Position Chart
        self.position_tab = tk.Frame(self.notebook)
        self.notebook.add(self.position_tab, text="Position")

        # Tab 3: Actions
        self.actions_tab = tk.Frame(self.notebook)
        self.notebook.add(self.actions_tab, text="Actions")

        # Tab 4: Trade Log
        self.trade_log_tab = tk.Frame(self.notebook)
        self.notebook.add(self.trade_log_tab, text="Trade Log")

        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def browse_directory(self):
        """Browse for results directory"""
        directory = filedialog.askdirectory(initialdir=".")
        if directory:
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, directory)
            self.data_dir = directory
            self.load_data()

    def load_data(self):
        """Load backtest results data"""
        self.data_dir = self.dir_entry.get()

        try:
            # Load backtest results
            csv_path = os.path.join(self.data_dir, "backtest_results.csv")
            if os.path.exists(csv_path):
                self.results_data = pd.read_csv(csv_path)
                self.status_bar.config(text=f"Loaded: {csv_path}")
                self.update_summary()
                self.plot_portfolio()
                self.plot_position()
                self.plot_actions()
                self.update_trade_log()
            else:
                messagebox.showwarning("Warning", f"File not found: {csv_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")
            self.status_bar.config(text=f"Error: {e}")

    def update_summary(self):
        """Update performance summary display"""
        if self.results_data is None:
            return

        df = self.results_data

        # Calculate metrics
        initial_value = df['portfolio_value'].iloc[0]
        final_value = df['portfolio_value'].iloc[-1]
        total_return = (final_value / initial_value - 1) * 100

        returns = df['portfolio_value'].pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        # Max drawdown
        cumulative = df['portfolio_value']
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min() * 100

        summary_text = f"""
Initial Value: ${initial_value:,.2f}
Final Value: ${final_value:,.2f}
Total Return: {total_return:+.2f}%
Sharpe Ratio: {sharpe:.2f}
Max Drawdown: {max_dd:.2f}%
Total Trades: {len(df)}
        """.strip()

        self.summary_label.config(text=summary_text)

    def plot_portfolio(self):
        """Plot portfolio value chart"""
        if self.results_data is None:
            return

        # Clear tab
        for widget in self.portfolio_tab.winfo_children():
            widget.destroy()

        df = self.results_data

        # Create figure
        fig = Figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

        ax.plot(pd.to_datetime(df['date']), df['portfolio_value'], 'b-', linewidth=2)
        ax.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=df['portfolio_value'].iloc[0], color='gray', linestyle='--', alpha=0.7)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Rotate x-axis labels
        plt.xticks(rotation=45)

        fig.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.portfolio_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_position(self):
        """Plot position chart"""
        if self.results_data is None:
            return

        # Clear tab
        for widget in self.position_tab.winfo_children():
            widget.destroy()

        df = self.results_data

        # Create figure
        fig = Figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

        ax.plot(pd.to_datetime(df['date']), df['holdings'], 'g-', linewidth=2)
        ax.set_title('Position Holdings Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Shares/Lots')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='-', alpha=0.5)

        # Color positive and negative regions
        ax.fill_between(pd.to_datetime(df['date']), df['holdings'], 0,
                       where=(df['holdings'] >= 0), color='green', alpha=0.3, label='Long')
        ax.fill_between(pd.to_datetime(df['date']), df['holdings'], 0,
                       where=(df['holdings'] < 0), color='red', alpha=0.3, label='Short')

        plt.xticks(rotation=45)
        fig.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.position_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_actions(self):
        """Plot action distribution"""
        if self.results_data is None:
            return

        # Clear tab
        for widget in self.actions_tab.winfo_children():
            widget.destroy()

        df = self.results_data

        # Create figure
        fig = Figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

        ax.plot(pd.to_datetime(df['date']), df['action'], 'purple', linewidth=1)
        ax.set_title('Actions Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Action')
        ax.set_ylim([-1.1, 1.1])
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.3)
        ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.3)

        plt.xticks(rotation=45)
        fig.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.actions_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_trade_log(self):
        """Update trade log display"""
        if self.results_data is None:
            return

        # Clear tab
        for widget in self.trade_log_tab.winfo_children():
            widget.destroy()

        df = self.results_data

        # Create treeview
        tree_scroll = tk.Scrollbar(self.trade_log_tab)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        columns = ('date', 'price', 'action', 'holdings', 'portfolio_value')
        tree = ttk.Treeview(self.trade_log_tab, yscrollcommand=tree_scroll.set, columns=columns, show='headings')

        for col in columns:
            tree.heading(col, text=col.capitalize())
            tree.column(col, width=120)

        # Add data
        for _, row in df.iterrows():
            tree.insert('', tk.END, values=[
                row['date'],
                f"{row['price']:.5f}",
                f"{row['action']:.4f}",
                int(row['holdings']),
                f"${row['portfolio_value']:,.2f}"
            ])

        tree_scroll.config(command=tree.yview)
        tree.pack(fill=tk.BOTH, expand=True)

    def run(self):
        """Run the GUI application"""
        self.root.mainloop()


def main():
    """Main entry point"""
    app = TradingVisualizationGUI()
    app.run()


if __name__ == "__main__":
    main()
