import pandas as pd
import numpy as np
from olist.data import Olist


class Order:
    '''
    DataFrames containing all orders as index,
    and various properties of these orders as columns
    '''
    def __init__(self):
        # Assign an attribute ".data" to all new instances of Order
        self.data = Olist().get_data()

    def get_wait_time(self, is_delivered=True):
        """
        Returns a DataFrame with:
        [order_id, wait_time, expected_wait_time, delay_vs_expected, order_status]
        and filters out non-delivered orders unless specified
        """

        # create a copy
        orders = self.data['orders'].copy()

        # filter delivered orders
        if is_delivered:
            orders = orders.query("order_status=='delivered'").copy()

        # handle datetime
        orders.loc[:, 'order_delivered_customer_date'] = \
            pd.to_datetime(orders['order_delivered_customer_date'])
        orders.loc[:, 'order_estimated_delivery_date'] = \
            pd.to_datetime(orders['order_estimated_delivery_date'])
        orders.loc[:, 'order_purchase_timestamp'] = \
            pd.to_datetime(orders['order_purchase_timestamp'])

        # compute wait time
        orders.loc[:, 'wait_time'] = \
            (orders['order_delivered_customer_date'] -
             orders['order_purchase_timestamp']) / np.timedelta64(24, 'h')

        # compute expected wait time
        orders.loc[:, 'expected_wait_time'] = \
            (orders['order_estimated_delivery_date'] -
             orders['order_purchase_timestamp']) / np.timedelta64(24, 'h')

        # compute delay vs expected
        orders.loc[:, 'delay_vs_expected'] = \
            (orders['order_delivered_customer_date'] -
             orders['order_estimated_delivery_date']) / np.timedelta64(24, 'h')

        def handle_delay(x):
            if x > 0:
                return x
            else:
                return 0

        orders.loc[:, 'delay_vs_expected'] = \
            orders['delay_vs_expected'].apply(handle_delay)


        return orders[[
            'order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected',
            'order_status'
        ]]

    def get_review_score(self):
        """
        Returns a DataFrame with:
        order_id, dim_is_five_star, dim_is_one_star, dim_two_star, dim_three_star, review_score
        """

        # import data
        reviews = self.data['order_reviews']

        def dim_five_star(d):
            if d == 5:
                return 1
            else:
                return 0

        def dim_one_star(d):
            if d == 1:
                return 1
            else:
                return 0

        reviews.loc[:, 'dim_is_five_star'] =\
            reviews['review_score'].apply(dim_five_star)

        reviews.loc[:, 'dim_is_one_star'] =\
            reviews['review_score'].apply(dim_one_star)

        return reviews[[
            'order_id', 'dim_is_five_star', 'dim_is_one_star', 'review_score'
        ]]


    def get_number_products(self):
        """
        Returns a DataFrame with:
        order_id, number_of_products
        """

        data = self.data
        products = \
            data['order_items']\
            .groupby('order_id',
                     as_index=False).agg({'order_item_id': 'count'})
        products.columns = ['order_id', 'number_of_products']
        return products


    def get_number_sellers(self):
        """
        Returns a DataFrame with:
        order_id, number_of_sellers
        """

        data = self.data
        sellers = \
            data['order_items']\
            .groupby('order_id')['seller_id'].nunique().reset_index()
        sellers.columns = ['order_id', 'number_of_sellers']

        return sellers


    def get_price_and_freight(self):
        """
        Returns a DataFrame with:
        order_id, price, freight_value
        """

        data = self.data
        price_freight = \
            data['order_items']\
            .groupby('order_id',
                     as_index=False).agg({'price': 'sum',
                                          'freight_value': 'sum'})

        return price_freight


    def get_training_data(self,
                          is_delivered=True,
                          with_distance_seller_customer=False):
        """
        Returns a clean DataFrame with the all following columns:
        ['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected',
        'order_status', 'dim_is_five_star', 'dim_is_one_star', 'review_score',
        'number_of_products', 'number_of_sellers', 'price', 'freight_value',
        'distance_seller_customer']
        """

        training_set =\
            self.get_wait_time(is_delivered)\
                .merge(
                self.get_review_score(), on='order_id'
            ).merge(
                self.get_number_products(), on='order_id'
            ).merge(
                self.get_number_sellers(), on='order_id'
            ).merge(
                self.get_price_and_freight(), on='order_id'
            )

        return training_set.dropna()
