{{ config(materialized='table') }}

    SELECT 
        od.OrderID,
        o.OrderDate,
        strftime('%Y', o.OrderDate) AS order_year,
        od.ProductID,
        p.ProductName,
        p.CategoryID,
        c.CategoryName,
        ROUND(od.UnitPrice * (1 - od.Discount), 2) AS net_price,
        SUM(od.Quantity) AS total_quantity_sold,
        COUNT(DISTINCT o.OrderID) AS total_orders
    FROM {{ source('northwind', 'order_details') }} od
    JOIN {{ source('northwind', 'products') }} p ON od.ProductID = p.ProductID
    JOIN {{ source('northwind', 'orders') }} o ON od.OrderID = o.OrderID
    JOIN {{ source('northwind', 'categories') }} c on p.CategoryID = c.CategoryID
    GROUP BY od.ProductID, order_year, net_price
    
    
    