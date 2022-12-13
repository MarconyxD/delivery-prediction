# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 17:04:52 2022

@author: Marcony Montini
"""

import pandas as pd

xls = pd.ExcelFile('SupremEats_2011.xlsx')

categories = pd.read_excel(xls, 'Categories')
customers = pd.read_excel(xls, 'Customers')
employees = pd.read_excel(xls, 'Employees')
order_details = pd.read_excel(xls, 'Order_Details')
orders = pd.read_excel(xls, 'Orders')
products = pd.read_excel(xls, 'Products')
shippers = pd.read_excel(xls, 'Shippers')
supliers = pd.read_excel(xls, 'Suppliers')

orders['Days_to_Ship'] = (orders['ShippedDate'] - orders['OrderDate']).dt.days

orders['Late_Days'] = (orders['RequiredDate'] - orders['ShippedDate']).dt.days

orders['Days_to_Ship'].describe()

orders['Late_Days'].describe()

late_orders = orders.drop(orders[orders.Late_Days > 0].index)
late_orders

late_orders = late_orders.drop(columns=['CustomerID','EmployeeID','OrderDate','RequiredDate','ShippedDate','ShipCity','ShipName', 'ShipAddress', 'ShipRegion', 'ShipPostalCode', 'OrderID'])
late_orders

late_orders = late_orders[late_orders['Late_Days'].notna()]
late_orders

late_orders['ShipVia'].value_counts()

late_orders['ShipCountry'].value_counts()

late_orders.groupby(late_orders['ShipVia'])["Late_Days"].mean().plot(kind="bar",rot=25)

late_orders.groupby(late_orders['ShipCountry'])["Late_Days"].mean().plot(kind="bar",rot=60,color='r',legend=True)
late_orders['ShipCountry'].value_counts().plot(kind="bar",rot=60,color='g',legend=True)

orders['ShipCountry'].replace(['Argentina','Austria','Belgium','Brazil','Canada','Denmark','Finland','France','Germany','Ireland','Italy','Mexico','Norway','Poland','Portugal','Spain','Sweden','Switzerland','UK','USA','Venezuela'], [0, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0], inplace=True)
orders = orders.drop(columns=['CustomerID','EmployeeID','OrderDate','RequiredDate','ShippedDate','ShipCity','ShipName', 'ShipAddress', 'ShipRegion', 'ShipPostalCode', 'OrderID'])
orders = orders[orders['Late_Days'].notna()]
orders

orders.corr()

X = orders
X = X.drop(columns=['Late_Days'])
y = orders
y = y.drop(columns=['ShipVia','Freight','ShipCountry','Days_to_Ship'])
X
y

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


from sklearn import metrics
import numpy as np

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))