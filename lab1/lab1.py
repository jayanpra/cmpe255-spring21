import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Solution:
    def __init__(self) -> None:
        # TODO: 
        # Load data from data/chipotle.tsv file using Pandas library and 
        # assign the dataset to the 'chipo' variable.
        file = 'data/chipotle.tsv'
        self.chipo = pd.read_csv(file, delimiter="\t")
    
    def top_x(self, count) -> None:
        # TODO
        # Top x number of entries from the dataset and display as markdown format.
        topx = self.chipo.head(count)
        print(topx.to_markdown())
        
    def count(self) -> int:
        # TODO
        # The number of observations/entries in the dataset.
        return self.chipo.shape[0]
    
    def info(self) -> None:
        # TODO
        # print data info.
        print (self.chipo.info())
    
    def num_column(self) -> int:
        # TODO return the number of columns in the dataset
        return self.chipo.shape[1]
    
    def print_columns(self) -> None:
        # TODO Print the name of all the columns.
        print(self.chipo.columns.to_list())
    
    def most_ordered_item(self):
        # TODO
        max_index  = self.chipo["quantity"].idxmax()
        item_name = self.chipo['item_name'][max_index]
        order_id = self.chipo['order_id'][max_index]
        quantity = self.chipo['quantity'][max_index]
        return item_name, order_id, quantity

    def total_item_orders(self) -> int:
       # TODO How many items were orderd in total?
       return self.chipo['quantity'].sum()
   
    def total_sales(self) -> float:
        # TODO 
        # 1. Create a lambda function to change all item prices to float.
        # 2. Calculate total sales.
        con = lambda x : float(x.replace("$",""))
        total = 0
        for i in range(0,len(self.chipo['item_price'])):
            self.chipo['item_price'][i] = con(self.chipo['item_price'][i])
            total += self.chipo['item_price'][i] * self.chipo['quantity'][i]
        total = float(format(total,".2f"))
        return total
   
    def num_orders(self) -> int:
        # TODO
        # How many orders were made in the dataset?
        return len(self.chipo['order_id'].unique())
    
    def average_sales_amount_per_order(self) -> float:
        # TODO
        total = 0
        for i in range(0,len(self.chipo['item_price'])):
            total += self.chipo['item_price'][i] * self.chipo['quantity'][i]
        total = float(format(total,".2f"))
        avg = total/len(self.chipo['order_id'].unique())
        import pdb;pdb.set_trace()
        return avg

    def num_different_items_sold(self) -> int:
        # TODO
        # How many different items are sold?
        return len(self.chipo['item_name'].unique())
    
    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        # TODO
        # 1. convert the dictionary to a DataFrame
        # 2. sort the values from the top to the least value and slice the first 5 items
        # 3. create a 'bar' plot from the DataFrame
        # 4. set the title and labels:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        # 5. show the plot. Hint: plt.show(block=True).
        pass
        
    def scatter_plot_num_items_per_order_price(self) -> None:
        # TODO
        # 1. create a list of prices by removing dollar sign and trailing space.
        # 2. groupby the orders and sum it.
        # 3. create a scatter plot:
        #       x: orders' item price
        #       y: orders' quantity
        #       s: 50
        #       c: blue
        # 4. set the title and labels.
        #       title: Numer of items per order price
        #       x: Order Price
        #       y: Num Items
        pass
    
        

def test() -> None:
    solution = Solution()
    solution.top_x(10)
    count = solution.count()
    print(count)
    assert count == 4622
    solution.info()
    count = solution.num_column()
    assert count == 5
    item_name, order_id, quantity = solution.most_ordered_item()
    #assert item_name == 'Chicken Bowl'
    #assert order_id == 713926	
    #assert quantity == 159
    total = solution.total_item_orders()
    assert total == 4972
    assert 39237.02 == solution.total_sales()
    assert 1834 == solution.num_orders()
    assert 21.39 == solution.average_sales_amount_per_order()
    assert 50 == solution.num_different_items_sold()
    '''solution.plot_histogram_top_x_popular_items(5)
    solution.scatter_plot_num_items_per_order_price()'''

    
if __name__ == "__main__":
    # execute only if run as a script
    test()
    
    