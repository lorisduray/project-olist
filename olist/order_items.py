import pandas as pd
import numpy as np
from olist.utils import haversine_distance
from olist.data import Olist

class Order_items:
    '''
    DataFrames containing all orders as index,
    and various properties of these orders as columns
    '''
    def __init__(self):
        # Assign an attribute ".data" to all new instances of Order
        self.data = Olist().get_data()

    def get_order_items(self):

        order_items = self.data['order_items'].copy()
        return order_items
