# Delivery Prediction

Predicting the delivery time of purchases is an important factor that directly influences the customers' purchase decision, depending on the need for the product. Therefore, a prediction application was performed for the SupremEats_2011 dataset using supervised learning models with the Python programming language.

Supervised learning models use already labeled samples to "learn", "find patterns" and "adapt" to the proposed problem, so that they can present results for different situations based on the existing history in the database.

The dataset used in the application is SupremEats_2011, which is also present in this repository. For a more iterative execution, follow the link of this application on Google Colab: https://colab.research.google.com/drive/1dytGF4pOhDj0FaDa2MKFxA3wofBYBWby?usp=sharing

The dataset may not be present in Colab due to automatically performed recycling. For that, it will be necessary to add it for execution.

SupremEats_2011 is composed by tables: Categories, Customers, Employees, Order_Details, Orders, Products, Shippers and Suppliers.

The objective then, for this problem, was to try to get as close as possible to predicting the delivery time for purchases based on the existing parameters in the data set.

First, we start by importing Pandas and defining it as pd. This means that every time we use pd, we are using Pandas. Pandas is a software library created for the Python language for data manipulation and analysis.

```
import pandas as pd
```

To use the plot() of the graphs to visualize the data, we need to change the used version of matplotlib. First, we uninstall the current version.

```
pip uninstall matplotlib
```

Now, we install the required version, which is 3.1.3.

```
pip install matplotlib==3.1.3
```

To import a file in .xlsx format we use Pandas ExcelFile() function. If you want to use another one, replace the existing one with the desired one. The imported file is saved in the xls variable. Note: as already said, Colab deletes attached files after a while, so remember to add the dataset.

```
xls = pd.ExcelFile('SupremEats_2011.xlsx')
```

The imported file has several worksheets. So that we can work properly, one option is to transfer each worksheet to a different DataFrame. DataFrames are our data tables, with information distributed in rows and columns. In most of the actions in which we use Pandas, we need our data to be in DataFrame format. Therefore, below each worksheet is transferred to a DataFrame with an indicative name.

```
categories = pd.read_excel(xls, 'Categories')
customers = pd.read_excel(xls, 'Customers')
employees = pd.read_excel(xls, 'Employees')
order_details = pd.read_excel(xls, 'Order_Details')
orders = pd.read_excel(xls, 'Orders')
products = pd.read_excel(xls, 'Products')
shippers = pd.read_excel(xls, 'Shippers')
supliers = pd.read_excel(xls, 'Suppliers')
```

Based on the ShippedDate and OrderDate columns present in the order table, we can calculate the amount of days that each order takes to ship after the request is initiated. This information is saved in the Days_to_Ship column.

```
orders['Days_to_Ship'] = (orders['ShippedDate'] - orders['OrderDate']).dt.days
```

Based on the Required Date Shipped Date columns present in the order table, we can calculate the number of days that orders were delayed to be shipped. This information is saved in the Late_Days column.

```
orders['Late_Days'] = (orders['RequiredDate'] - orders['ShippedDate']).dt.days
```

Using the describe() function it is possible to check interesting statistics regarding column values, such as sum, average, etc.

```
orders['Days_to_Ship'].describe()
```

<p align="center">
<img width="600" src="/Figures/01.png" alt="Figure 01">
</p>

Using the describe() function it is possible to check interesting statistics regarding column values, such as sum, average, etc.

```
orders['Late_Days'].describe()
```

<p align="center">
<img width="600" src="/Figures/02.png" alt="Figure 02">
</p>

Focusing only on orders that are late, we can filter our data and create a late_orders table that contains only orders where orders were sent on or after RequiredDate. Thus, we exclude all rows that have positive values for the Late_Days parameter with the drop() function.

```
late_orders = orders.drop(orders[orders.Late_Days > 0].index)
late_orders
```

<p align="center">
<img width="600" src="/Figures/03.png" alt="Figure 03">
</p>

The next step is to exclude all columns that are not useful to us for this analysis.

```
late_orders = late_orders.drop(columns=['CustomerID','EmployeeID','OrderDate','RequiredDate','ShippedDate','ShipCity','ShipName', 'ShipAddress', 'ShipRegion', 'ShipPostalCode', 'OrderID'])
late_orders
```

<p align="center">
<img width="600" src="/Figures/04.png" alt="Figure 04">
</p>

Then keep only the rows that have values in all columns, excluding those that have NaN values. The image below shows only part of this new table. Now, there are no NaN values.

```
late_orders = late_orders[late_orders['Late_Days'].notna()]
late_orders
```

<p align="center">
<img width="600" src="/Figures/05.png" alt="Figure 05">
</p>

To start the analysis, we can use the value_counts() function to count the number of occurrences of all values ​​present in a given column. In this way, we apply value_counts() to count the number of occurrences of each shipping company in the delay table. Thus, we obtain information on the total backlog of orders in each company during the analysis period.

```
late_orders['ShipVia'].value_counts()
```

<p align="center">
<img width="600" src="/Figures/06.png" alt="Figure 06">
</p>

We can do the same to identify the number of delay occurrences in each country during the analysis period.

```
late_orders['ShipCountry'].value_counts()
```

<p align="center">
<img width="600" src="/Figures/07.png" alt="Figure 07">
</p>

Now, for a better visualization of the data, it is interesting to use the graph plot. The first deals with the relationship between the ShipVia parameters, corresponding to the identification of the companies transporting the products (companies 1, 2 and 3), and the second is the number of days of delay. In this way, we use the mean() function to calculate the average number of days late for each company and plot the result through a bar graph. Thus, the average delay in company 1 is 8 days, in company 2 it is approximately 4.5 days and in company 3 it is approximately 5.5 days.

```
late_orders.groupby(late_orders['ShipVia'])["Late_Days"].mean().plot(kind="bar",rot=25)
```

<p align="center">
<img width="600" src="/Figures/08.png" alt="Figure 08">
</p>

Finally, we have the comparison plot between the number of occurrences of order delays in each country (in green) and the average of days of delay related to this amount of delays (in red).

```
late_orders.groupby(late_orders['ShipCountry'])["Late_Days"].mean().plot(kind="bar",rot=60,color='r',legend=True)
late_orders['ShipCountry'].value_counts().plot(kind="bar",rot=60,color='g',legend=True)
```

<p align="center">
<img width="600" src="/Figures/09.png" alt="Figure 09">
</p>

Now, let's try to extract some information from the correlations between the existing parameters in the orders table and try to predict how many days an order takes to be sent to the customer. First, let's convert the ShipCountry parameter to a numeric variable. We can transform each country into a value, but we will have many different values and few occurrences for each value. So let's perform this transformation by region, so 0 is equivalent to South America, 1 to North and Central America, and 2 to Europe. For this, we use the replace() function.

```
orders['ShipCountry'].replace(['Argentina','Austria','Belgium','Brazil','Canada','Denmark','Finland','France','Germany','Ireland','Italy','Mexico','Norway','Poland','Portugal','Spain','Sweden','Switzerland','UK','USA','Venezuela'], [0, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0], inplace=True)
orders = orders.drop(columns=['CustomerID','EmployeeID','OrderDate','RequiredDate','ShippedDate','ShipCity','ShipName', 'ShipAddress', 'ShipRegion', 'ShipPostalCode', 'OrderID'])
orders = orders[orders['Late_Days'].notna()]
```

<p align="center">
<img width="600" src="/Figures/10.png" alt="Figure 10">
</p>

For correlation, we use the corr() function. Values close to 1 or -1 indicate a high correction. Values ​​close to zero indicate a low correlation. As we can see, the correlation between the variables is not very good, which makes us think that there is a lack of information in this table to establish more concrete correlations between the variables or that the dataset may have been created randomly, without the pattern real between the data. In short, the parameters that have the highest correlation with each other are Days_to_Ship with Late_Days.

```
orders.corr()
```

<p align="center">
<img width="600" src="/Figures/11.png" alt="Figure 11">
</p>

To try to predict the days to ship, let's start by separating our labels in a variable y and leaving the remaining attributes in a variable X. The labels are the result we want to predict, that is, the Late_Days. We will use the information we have to train the algorithm so that it tries to establish patterns and predict the value of Late_Days for different parameter values.

```
X = orders
X = X.drop(columns=['Late_Days'])
y = orders
y = y.drop(columns=['ShipVia','Freight','ShipCountry','Days_to_Ship'])
X
```

<p align="center">
<img width="600" src="/Figures/12.png" alt="Figure 12">
</p>

```
y
```

<p align="center">
<img width="600" src="/Figures/13.png" alt="Figure 13">
</p>

Let's separate the data into two groups: training data and test data. For this, we will use train_test_split.

```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Let's use a Random Forest for the regression.

```
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
```

To validate the results, we used some metrics, such as MAE, MSE and RMSE. Through these results we were able to identify how good the forecast was. As they give us error values, the closer to zero, the better.

```
from sklearn import metrics
import numpy as np

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
```

<p align="center">
<img width="600" src="/Figures/14.png" alt="Figure 14">
</p>

Therefore, the forecast is having an absolute error of approximately 4 to 5 days when calculating how long it would take for the order to be shipped. It is worth mentioning that several different models were used to find which model best adapted to the data set. In total, 26 different models were applied. Other models presented results close to or equal to those of Random Forest, but to make this application a little more didactic and easy to understand for those who do not have much knowledge, we chose to use only Random Forest to present the results.
