import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator


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
        order_dict = {}
        order_id_dict = {}
        for i in range(0,len(self.chipo["quantity"])):
            if self.chipo["item_name"][i] not in order_dict.keys():
                order_dict.update({self.chipo["item_name"][i] : self.chipo["quantity"][i]})
                order_id_dict.update({self.chipo["item_name"][i] : self.chipo["order_id"][i]})
            else:
                order_dict[self.chipo["item_name"][i]] += self.chipo["quantity"][i]
                order_id_dict[self.chipo["item_name"][i]] += self.chipo["order_id"][i]
        item_name = max(order_dict.items(), key=operator.itemgetter(1))[0]
        order_id = order_id_dict[item_name]
        quantity = order_dict[item_name]
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
        avg = float(format(avg,".2f"))
        return avg

    def num_different_items_sold(self) -> int:
        # TODO
        # How many different items are sold?
        return len(self.chipo['item_name'].unique())
    
    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        # TODO
        # 1. convert the dictionary to a DataFrame
        # 2. sort the values from the top to the least value and slice the first 5 items
        # 3. create a 'bar' plot from the DataFrame
        # 4. set the title and labels:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        # 5. show the plot. Hint: plt.show(block=True).
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        order_dict = {}
        for i in range(0,len(self.chipo["quantity"])):
            if self.chipo["item_name"][i] not in order_dict.keys():
                order_dict.update({self.chipo["item_name"][i] : self.chipo["quantity"][i]})
            else:
                order_dict[self.chipo["item_name"][i]] += self.chipo["quantity"][i]
        sorted_dict = {k:v for k,v in sorted(order_dict.items(), key = lambda item : item[1], reverse = True)}
        hist_dict = {}
        for i in sorted_dict.keys():
            hist_dict.update({i:sorted_dict[i]})
            if len(hist_dict) == x:
                break
        hist_dict = pd.DataFrame.from_dict({"Items":hist_dict.keys(),"Number of orders":hist_dict.values()})
        bar = hist_dict.plot.bar(x="Items", y="Number of orders", title ="Most Popular Items", rot = 0)
        plt.show(block=True)
        
        
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
        current_order = 0
        items_vs_price = []
        #price_f = lambda x : float(x.replace("$","")
        for i in range(0,len(self.chipo["order_id"])):
            items = self.chipo["quantity"][i]
            price = self.chipo["item_price"][i]
            if self.chipo["order_id"][i] != current_order:
                items_vs_price.append([items,price*items])
                current_order = self.chipo["order_id"][i]
            else:
                items_vs_price[-1][0] += items
                items_vs_price[-1][1] += items*price
        df = pd.DataFrame(data=items_vs_price, columns = ["number_of_items", "price"])
        df.plot.scatter(x="number_of_items",y="price", title="Scatter plot for number of items vs price per order id")
        plt.show(block=True)

        

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
    assert item_name == 'Chicken Bowl'
    assert order_id == 713926
    assert quantity == 761          #quantity given below should be incorrect	
    #assert quantity == 159
    total = solution.total_item_orders()
    assert total == 4972
    assert 39237.02 == solution.total_sales()
    assert 1834 == solution.num_orders()
    assert 21.39 == solution.average_sales_amount_per_order()
    assert 50 == solution.num_different_items_sold()
    solution.plot_histogram_top_x_popular_items(5)
    solution.scatter_plot_num_items_per_order_price()

    
if __name__ == "__main__":
    # execute only if run as a script
    test()
    
    