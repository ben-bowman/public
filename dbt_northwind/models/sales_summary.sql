
WITH order_details AS (
    SELECT 
        od.OrderID,
        od.ProductID,
        od.UnitPrice,
        od.Quantity,
        od.Discount,
        (od.UnitPrice * od.Quantity) - (od.UnitPrice * od.Quantity * od.Discount) AS total_price
    FROM {{ source('northwind', 'order_details') }} AS od
)
SELECT 
    o.OrderID,
    o.CustomerID,
    o.EmployeeID,
    o.OrderDate,
    o.ShippedDate,
    o.Freight,
    SUM(od.total_price) AS total_order_value
FROM {{ source('northwind', 'orders') }} AS o
JOIN order_details AS od ON o.OrderID = od.OrderID
GROUP BY o.OrderID, o.CustomerID, o.EmployeeID, o.OrderDate, o.ShippedDate, o.Freight
