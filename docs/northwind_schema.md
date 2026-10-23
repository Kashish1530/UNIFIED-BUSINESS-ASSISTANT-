# Northwind Database Schema (SQLite)

This document provides the reference schema for the Northwind SQLite database used by **Role B — Data Service**.

## Database Overview

- **Engine**: SQLite 3
- **File Location**: `data/northwind.db`
- **Total Tables**: 8
- **Total Rows**: 932
- **Foreign Key Enforcement**: `PRAGMA foreign_keys = ON;` required per connection.

---

## Tables & Relationships

### 1. `Categories` (8 rows)
- `CategoryID`: `INTEGER PRIMARY KEY AUTOINCREMENT`
- `CategoryName`: `VARCHAR(25)`
- `Description`: `VARCHAR(255)`

### 2. `Customers` (91 rows)
- `CustomerID`: `INTEGER PRIMARY KEY AUTOINCREMENT`
- `CustomerName`: `VARCHAR(50)`
- `ContactName`: `VARCHAR(50)`
- `Address`: `VARCHAR(50)`
- `City`: `VARCHAR(20)`
- `PostalCode`: `VARCHAR(10)`
- `Country`: `VARCHAR(15)`

### 3. `Employees` (10 rows)
- `EmployeeID`: `INTEGER PRIMARY KEY AUTOINCREMENT`
- `LastName`: `VARCHAR(15)`
- `FirstName`: `VARCHAR(15)`
- `BirthDate`: `DATETIME`
- `Photo`: `VARCHAR(25)`
- `Notes`: `VARCHAR(1024)`

### 4. `Shippers` (3 rows)
- `ShipperID`: `INTEGER PRIMARY KEY AUTOINCREMENT`
- `ShipperName`: `VARCHAR(25)`
- `Phone`: `VARCHAR(15)`

### 5. `Suppliers` (29 rows)
- `SupplierID`: `INTEGER PRIMARY KEY AUTOINCREMENT`
- `SupplierName`: `VARCHAR(50)`
- `ContactName`: `VARCHAR(50)`
- `Address`: `VARCHAR(50)`
- `City`: `VARCHAR(20)`
- `PostalCode`: `VARCHAR(10)`
- `Country`: `VARCHAR(15)`
- `Phone`: `VARCHAR(15)`

### 6. `Products` (77 rows)
- `ProductID`: `INTEGER PRIMARY KEY AUTOINCREMENT`
- `ProductName`: `VARCHAR(50)`
- `SupplierID`: `INTEGER` -> `REFERENCES Suppliers(SupplierID)`
- `CategoryID`: `INTEGER` -> `REFERENCES Categories(CategoryID)`
- `Unit`: `VARCHAR(25)`
- `Price`: `NUMERIC`

### 7. `Orders` (196 rows)
- `OrderID`: `INTEGER PRIMARY KEY AUTOINCREMENT`
- `CustomerID`: `INTEGER` -> `REFERENCES Customers(CustomerID)`
- `EmployeeID`: `INTEGER` -> `REFERENCES Employees(EmployeeID)`
- `OrderDate`: `DATETIME`
- `ShipperID`: `INTEGER` -> `REFERENCES Shippers(ShipperID)`

### 8. `OrderDetails` (518 rows)
- `OrderDetailID`: `INTEGER PRIMARY KEY AUTOINCREMENT`
- `OrderID`: `INTEGER` -> `REFERENCES Orders(OrderID)`
- `ProductID`: `INTEGER` -> `REFERENCES Products(ProductID)`
- `Quantity`: `INTEGER`

---

## Entity Relationship Overview

```
Categories (8) <------ Products (77) <------- OrderDetails (518)
                         ^                         |
Suppliers (29) <---------+                         v
                                              Orders (196)
                                               |   |   |
     Customers (91) <--------------------------+   |   +--------> Shippers (3)
                                                   v
                                             Employees (10)
```
